import random

from simulation.basic_simulation import BasicSimulation
from models.spatiotemporal_model import SpatioTemporalModel # 导入正确的模型
from simulation.utils.simulation_utils import (
    get_neighbors,
    is_within_trust_threshold,
    update_pairwise_opinion,
    initialize_poisson_events,
    calculate_event_influence,
    get_all_node_positions,
    build_kdtree,
    get_spatial_neighbors_kdtree
)

class SpatioTemporalSimulation(BasicSimulation):
    """
    时空耦合仿真过程（完全重构，支持多层网络）。

    - 继承自 BasicSimulation，复用其主循环、状态记录和耦合机制。
    - 覆写网络初始化方法，确保使用 SpatioTemporalModel 创建带有空间坐标的多层网络。
    - 覆写单步交互方法，将时空事件的影响（如动态学习率、信任阈值）注入到
      在多层网络中发生的每一次交互中。
    """
    def __init__(self, xp, network_params, sim_params):
        """
        初始化时空仿真器。
        """
        # 1. 调用父类构造函数，它会处理好网络和状态的【基础】初始化。
        #    父类的 __init__ 会自动调用我们下面覆写的 _initialize_network()，
        #    并在此之后初始化 self.num_layers。
        super().__init__(xp, network_params, sim_params)

        # 2. 仅初始化本类特有的时空事件相关属性。
        print("正在初始化 Spatio-Temporal 仿真机制...")
        st_config = self.sim_params.get('spatiotemporal_params', {})
        self.event_effects = {
            'interaction_prob': st_config.get('interaction_prob', {'base': 1.0, 'gain': 0.0}),
            'trust_scope': st_config.get('trust_scope', {'gain': 0.0}),
            'learning_rate': st_config.get('learning_rate', {'gain': 0.0})
        }
        self.event_decay = {
            'alpha': st_config.get('alpha', 0.5), # 空间衰减
            'beta': st_config.get('beta', 0.1)   # 时间衰减
        }
        # 【修改】将 self.num_layers 传递给事件生成器
        self.events = initialize_poisson_events(st_config, self.sim_params, self.num_layers, self.xp)
        self.spatial_neighbor_params = st_config.get('spatial_neighbors', {'enabled': False})
    
    def _initialize_network(self):
        """
        【覆写父类方法】
        使用 SpatioTemporalModel 初始化网络，确保节点具有统一的空间属性。
        这确保了 self.graphs 中的每个图都带有 'pos' 节点属性。
        """
        print("正在初始化【时空】多层网络...")
        st_model = SpatioTemporalModel(num_nodes=self.num_nodes)
        for layer_cfg in self.network_params['layers']:
            st_model.add_layer(layer_cfg['type'], **layer_cfg['params'])
        
        # 从 network_params 中获取位置配置
        position_config = self.network_params.get('position_distribution', {})
        self.graphs = st_model.build_spatiotemporal_multilayer_network(position_config)
        print("时空多层网络初始化完毕，节点已附加空间坐标。")

    def _run_interaction_step(self, opinions_at_start_of_step):
        """
        【覆写父类方法】
        实现包含时空事件影响的、在多层网络上运行的单步交互逻辑。
        【新增】支持混合邻居模型。
        """
        node_order = random.sample(range(self.num_nodes), self.num_nodes)
        
        # 【优化】在每一步开始时，预先计算一次所有节点的位置
        # 注意：这里假设所有层的节点位置是相同的，我们的模型设计保证了这一点
        all_positions = get_all_node_positions(self.graphs[0])
        kdtree, node_ids_for_kdtree = build_kdtree(all_positions)

        for active_node in node_order:
            # 1. 计算时空影响因子 (逻辑不变)
            influence_factor = calculate_event_influence(
                node_id=active_node,
                events=self.events,
                current_iteration=self.current_iteration,
                graphs=self.graphs,
                event_decay=self.event_decay,
                xp=self.xp
            )

            # 2. 动态调整交互概率 (逻辑不变)
            prob_config = self.event_effects['interaction_prob']
            final_interaction_prob = min(1.0, prob_config['base'] + prob_config['gain'] * influence_factor)
            if random.random() > final_interaction_prob:
                continue

            # 3. 随机选择交互层 (逻辑不变)
            layer_index = random.randrange(self.num_layers)
            current_graph = self.graphs[layer_index]
            
            # 4. 计算动态交互参数 (逻辑不变)
            trust_gain = self.event_effects['trust_scope']['gain']
            effective_epsilon = self.trust_thresholds[active_node, layer_index] + trust_gain * influence_factor
            
            # 5. 【核心修改】实现混合邻居发现
            # 5.1 获取固定的【拓扑邻居】
            structural_neighbors = get_neighbors(current_graph, active_node)
            
            spatial_neighbors = []
            if self.spatial_neighbor_params.get('enabled', False) and kdtree is not None:
                base_radius = self.spatial_neighbor_params.get('base_radius', 0.0)
                radius_gain = self.spatial_neighbor_params.get('radius_gain', 0.0)
                effective_radius = base_radius + radius_gain * influence_factor
                
                if effective_radius > 1e-6:
                    # 【替换】使用高效的 KD-Tree 查询
                    spatial_neighbors = get_spatial_neighbors_kdtree(
                        kdtree, node_ids_for_kdtree, active_node, effective_radius
                    )

            # 5.3 合并两类邻居并去重
            # 使用集合(set)可以高效地完成合并和去重
            potential_neighbors = list(set(structural_neighbors) | set(spatial_neighbors))
            if not potential_neighbors: 
                continue
            
            # 6. 从【潜在邻居池】中筛选可交互的邻居
            candidate_neighbors = [n for n in potential_neighbors if is_within_trust_threshold(
                opinions_at_start_of_step[active_node, layer_index],
                opinions_at_start_of_step[n, layer_index],
                effective_epsilon
            )]

            # 7. 如果找到邻居，则进行观点更新 (逻辑不变)
            if candidate_neighbors:
                base_mu = self.sim_params['dw_params']['mu']
                mu_gain = self.event_effects['learning_rate']['gain']
                effective_mu = min(0.5, base_mu + mu_gain * influence_factor)
                
                target_node = random.choice(candidate_neighbors)
                
                new_op_active, new_op_target = update_pairwise_opinion(
                    opinions_at_start_of_step[active_node, layer_index],
                    opinions_at_start_of_step[target_node, layer_index],
                    effective_mu
                )
                self.opinions[active_node, layer_index] = new_op_active
                self.opinions[target_node, layer_index] = new_op_target
                
    # 不再需要 run_simulation 方法，它会从 BasicSimulation 继承！
    # 也不再需要 _record_state, save_results 等方法，全部继承！