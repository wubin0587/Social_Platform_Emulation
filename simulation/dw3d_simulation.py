# simulation/dw3d_simulation.py

import random
from tqdm import tqdm

from simulation.basic_simulation import BasicSimulation # --- 继承已重构的基类 ---
from utils.gpu_utils import xp
from simulation.utils.simulation_utils import (
    get_neighbors, 
    clip_opinions,
    update_three_body_opinion,
    dynamic_epsilon_extremity,
    dynamic_epsilon_time_evolution,
    is_within_trust_threshold
)

class DW3DSimulation(BasicSimulation):
    """
    三维 DW 观点动力学仿真模型（已重构）。

    - 继承自 BasicSimulation，复用其主循环、状态记录和耦合机制。
    - 覆写单步交互方法，以集成三主体交互、动态信任阈值和动态网络调整等特有机制。
    """
    def __init__(self, xp, network_params, sim_params):
        # 1. 调用父类的构造函数，完成所有基础初始化。
        super().__init__(xp, network_params, sim_params)
        
        # 2. 仅初始化本类特有的状态。
        self.initial_epsilons = self.trust_thresholds.copy()
        
        print("DW3D 扩展机制已初始化。")

    def _perform_interaction(self, active_node, layer_index, candidates):
        """
        【覆写父类方法】
        实现包含两人或三人交互的逻辑。这是 DW3D 的核心交互规则。
        """
        dw3d_ext_config = self.sim_params.get('dw3d_extensions', {})
        interaction_mode = dw3d_ext_config.get('interaction_mode', 'random')
        
        attempt_three_body = (interaction_mode in ['random', 'three_body_only']) and (len(candidates) >= 2)
        three_body_success = False

        if attempt_three_body:
            j, k = random.sample(candidates, 2)
            epsilon_j = self.trust_thresholds[j, layer_index]
            epsilon_k = self.trust_thresholds[k, layer_index]

            if (is_within_trust_threshold(self.opinions[j, layer_index], self.opinions[k, layer_index], epsilon_j) and
                is_within_trust_threshold(self.opinions[k, layer_index], self.opinions[active_node, layer_index], epsilon_k)):
                
                new_op_i, new_op_j, new_op_k = update_three_body_opinion(
                    self.opinions[active_node, layer_index], self.opinions[j, layer_index], self.opinions[k, layer_index]
                )
                self.opinions[active_node, layer_index] = new_op_i
                self.opinions[j, layer_index] = new_op_j
                self.opinions[k, layer_index] = new_op_k
                three_body_success = True

        if not three_body_success and interaction_mode != 'three_body_only':
            # 如果三主体交互未发生或不被允许，则退回至父类的标准二主体交互。
            # 这是非常好的继承实践！
            super()._perform_interaction(active_node, layer_index, candidates)

    def _dynamic_network_adjustment(self, active_node, layer_index):
        """【DW3D 扩展】动态调整网络结构。"""
        dw3d_ext_config = self.sim_params.get('dw3d_extensions', {})
        dyn_net_config = dw3d_ext_config.get('dynamic_network', {})
        if not dyn_net_config.get('enabled', False): return

        graph = self.graphs[layer_index]
        epsilon_active = self.trust_thresholds[active_node, layer_index]
        
        # ... (内部逻辑不变) ...
        disconnect_threshold = epsilon_active * dyn_net_config['disconnect_threshold_factor']
        for neighbor in list(graph.neighbors(active_node)):
            if self.xp.abs(self.opinions[active_node, layer_index] - self.opinions[neighbor, layer_index]) > disconnect_threshold:
                if graph.has_edge(active_node, neighbor): graph.remove_edge(active_node, neighbor)

        if random.random() < dyn_net_config['reconnect_probability']:
            non_neighbors = [n for n in range(self.num_nodes) if not graph.has_edge(active_node, n) and n != active_node]
            if non_neighbors:
                reconnect_opinion_threshold = dyn_net_config['reconnect_opinion_threshold']
                candidates = [n for n in non_neighbors if self.xp.abs(self.opinions[active_node, layer_index] - self.opinions[n, layer_index]) < reconnect_opinion_threshold]
                if candidates: graph.add_edge(active_node, random.choice(candidates))
    
    def _update_dynamic_epsilon(self, node_id, layer_index):
        """【DW3D 扩展】更新动态信任阈值。"""
        dw3d_ext_config = self.sim_params.get('dw3d_extensions', {})
        dyn_eps_config = dw3d_ext_config.get('dynamic_epsilon', {})
        if not dyn_eps_config.get('enabled', False): return
        
        # ... (内部逻辑不变) ...
        current_epsilon = self.initial_epsilons[node_id, layer_index]
        extremity_config = dyn_eps_config.get('extremity_inhibition', {})
        if extremity_config.get('enabled', False):
            current_epsilon = dynamic_epsilon_extremity(self.opinions[node_id, layer_index], current_epsilon, extremity_config['alpha'], extremity_config['opinion_center'])
        
        time_config = dyn_eps_config.get('time_evolution', {})
        if time_config.get('enabled', False):
            current_epsilon += dynamic_epsilon_time_evolution(self.initial_epsilons[node_id, layer_index], time_config['beta'], self.current_iteration)
        
        self.trust_thresholds[node_id, layer_index] = self.xp.clip(current_epsilon, 0.0, 1.0)

    def _run_interaction_step(self, opinions_at_start_of_step):
        """
        【覆写父类方法】
        实现包含 DW3D 特有机制的单步交互逻辑。
        """
        node_order = random.sample(range(self.num_nodes), self.num_nodes)
        for active_node in node_order:
            layer_index = random.randrange(self.num_layers)
            
            # 1. 【DW3D 扩展】在交互前更新动态信任阈值。
            self._update_dynamic_epsilon(active_node, layer_index)
            
            # 2. 标准的邻居筛选流程。
            neighbors = get_neighbors(self.graphs[layer_index], active_node)
            if not neighbors:
                # 即使没有邻居，网络也可能调整（例如，重新连接）。
                self._dynamic_network_adjustment(active_node, layer_index)
                continue

            epsilon_active = self.trust_thresholds[active_node, layer_index]
            candidate_neighbors = [n for n in neighbors if is_within_trust_threshold(
                opinions_at_start_of_step[active_node, layer_index],
                opinions_at_start_of_step[n, layer_index], 
                epsilon_active)]
            
            # 3. 如果有候选者，则执行 DW3D 特有的交互（2体或3体）。
            if candidate_neighbors:
                self._perform_interaction(active_node, layer_index, candidate_neighbors)

            # 4. 【DW3D 扩展】在交互后调整网络结构。
            self._dynamic_network_adjustment(active_node, layer_index)

    # 不再需要 run_simulation 方法！它将从 BasicSimulation 继承。