import numpy as np
import random
import os
from datetime import datetime
from tqdm import tqdm

from utils.gpu_utils import xp, to_cpu
from models.base_model import BaseModel
from models.multilayer_model import MultilayerModel
from simulation.utils.simulation_utils import (
    update_pairwise_opinion,
    get_neighbors,
    is_within_trust_threshold,
    clip_opinions
)

class BasicSimulation:
    """
    基础多层仿真模型。
    通过 __init__ 方法接收所有参数和计算后端(xp)。
    """
    def __init__(self, xp, network_params, sim_params):
        # --- 1. 接收注入的依赖 ---
        self.xp = xp
        self.network_params = network_params
        self.sim_params = sim_params
        
        # --- 2. 从传入的参数中提取配置 ---
        self.num_nodes = network_params['num_nodes']
        self.num_layers = len(network_params['layers'])

        self.graphs = []
        self.opinions = None
        self.trust_thresholds = None
        self.current_iteration = 0
        self.history = []

        self._initialize_network()
        self._initialize_state()
        
        print(f"基础仿真初始化完成。节点数: {self.num_nodes}, 层数: {self.num_layers}")

    def _initialize_network(self):
        """使用传入的 network_params 初始化网络。"""
        print("正在初始化多层网络...")
        multilayer_model = MultilayerModel(num_nodes=self.num_nodes)
        for layer_cfg in self.network_params['layers']:
            multilayer_model.add_layer(layer_cfg['type'], **layer_cfg['params'])
        self.graphs = multilayer_model.build_multilayer_network()

    def _initialize_state(self):
        """使用传入的 sim_params 初始化状态。"""
        initial_config = self.sim_params['initial_state']
        opinion_range = tuple(initial_config['opinion_range'])
        epsilon_base = initial_config['epsilon_base']

        self.opinions = self.xp.random.uniform(opinion_range[0], opinion_range[1], (self.num_nodes, self.num_layers))
        self.trust_thresholds = self.xp.full((self.num_nodes, self.num_layers), epsilon_base)
    
    def _perform_interaction(self, active_node, layer_index, candidates):
        mu = self.sim_params['dw_params']['mu']
        
        target_node = random.choice(candidates)
        
        opinion_active = self.opinions[active_node, layer_index]
        opinion_target = self.opinions[target_node, layer_index]

        new_op_active, new_op_target = update_pairwise_opinion(
            opinion_active, opinion_target, mu
        )
        self.opinions[active_node, layer_index] = new_op_active
        self.opinions[target_node, layer_index] = new_op_target

    def _record_state(self):
        """记录当前仿真状态（将数据移至CPU）。"""
        opinions_cpu = to_cpu(self.opinions)
        self.history.append({
            'iteration': self.current_iteration,
            'opinions_snapshot': opinions_cpu
        })
    
    def _run_interaction_step(self, opinions_at_start_of_step):
        """
        【新增】执行一个完整迭代步中的所有【基础】节点交互。
        子类将覆写此方法来注入更复杂的交互逻辑，例如时空效应。
        """
        node_order = random.sample(range(self.num_nodes), self.num_nodes)
        for active_node in node_order:
            # 随机选择一层进行交互
            layer_index = random.randrange(self.num_layers)
            graph = self.graphs[layer_index]
            
            neighbors = get_neighbors(graph, active_node)
            if not neighbors: continue
            
            epsilon_active = self.trust_thresholds[active_node, layer_index]
            candidate_neighbors = [n for n in neighbors if is_within_trust_threshold(
                opinions_at_start_of_step[active_node, layer_index],
                opinions_at_start_of_step[n, layer_index], 
                epsilon_active)]
            
            if candidate_neighbors:
                self._perform_interaction(active_node, layer_index, candidate_neighbors)

    def run_simulation(self):
        """
        运行【通用】仿真主循环。
        """
        max_iterations = self.sim_params['max_iterations']
        record_interval = self.sim_params['record_interval']
        
        # --- [新增] 初始化收敛判断参数 ---
        conv_config = self.sim_params.get('convergence_params', {})
        conv_enabled = conv_config.get('enabled', False)
        conv_threshold = conv_config.get('threshold', 1e-5)
        conv_patience = conv_config.get('patience', 50)
        patience_counter = 0

        # 耦合机制代码
        coupling_config = self.sim_params.get('coupling_params', {})
        coupling_enabled = coupling_config.get('enabled', False) and self.num_layers > 1
        if coupling_enabled:
            lambda_coupling = coupling_config.get('lambda', 0.0)
            c_kl_matrix = coupling_config.get('c_kl_matrix', None)
            if c_kl_matrix: c_kl = self.xp.array(c_kl_matrix)
            else:
                c_kl_global = coupling_config.get('c_kl_global', 0.0)
                c_kl = self.xp.full((self.num_layers, self.num_layers), c_kl_global); self.xp.fill_diagonal(c_kl, 0)
            print(f"圈层耦合机制已启用: lambda={lambda_coupling}")

        print("开始仿真...")
        if conv_enabled:
            print(f"收敛判断机制已启用: 阈值={conv_threshold}, 耐心值={conv_patience}")
        self._record_state()

        progress_bar = tqdm(range(max_iterations), desc="Simulating", total=max_iterations, mininterval=0.5)

        for t in progress_bar:
            self.current_iteration = t + 1
            opinions_at_start_of_step = self.opinions.copy()

            self._run_interaction_step(opinions_at_start_of_step)

            if coupling_enabled:
                op_k = opinions_at_start_of_step[:, :, None]; op_l = opinions_at_start_of_step[:, None, :]
                diffs = op_l - op_k; weighted_diffs = c_kl * diffs
                coupling_sum = self.xp.sum(weighted_diffs, axis=2)
                self.opinions += lambda_coupling * coupling_sum
                
            self.opinions = clip_opinions(self.opinions, tuple(self.sim_params['initial_state']['opinion_range']))

            # --- [新增] 收敛判断逻辑 ---
            if conv_enabled:
                opinion_change = self.xp.sum(self.xp.abs(self.opinions - opinions_at_start_of_step))
                
                progress_bar.set_postfix(opinion_change=f'{opinion_change:.2e}', patience=f'{patience_counter}/{conv_patience}')

                if opinion_change < conv_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0  # 若变化大于阈值，则重置计数器
                
                if patience_counter >= conv_patience:
                    print(f"\n系统在第 {self.current_iteration} 次迭代时达到收敛标准。提前结束仿真。")
                    break  # 退出循环

            if self.current_iteration % record_interval == 0:
                self._record_state()
        
        # --- [新增] 确保循环结束后，最终状态被记录 ---
        if self.history[-1]['iteration'] != self.current_iteration:
            self._record_state()

        print("仿真结束。")
        return self.history

    def save_results(self, output_dir="simulation_results"):
        """保存仿真结果。"""
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"basic_simulation_results_{timestamp}.npz"
        filepath = os.path.join(output_dir, filename)
        
        opinions_history = np.array([h['opinions_snapshot'] for h in self.history])
        np.savez(filepath, 
                 sim_params=self.sim_params, 
                 network_params=self.network_params, 
                 history=self.history, 
                 opinions_history=opinions_history)
        print(f"仿真结果已保存到: {filepath}")