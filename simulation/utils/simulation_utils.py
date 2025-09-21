import networkx as nx
import random
import numpy as xp
from utils.gpu_utils import xp 

def initialize_opinions(num_nodes, opinion_range=(0.0, 1.0)):
    """
    初始化所有节点的观点。
    Args:
        num_nodes (int): 节点数量。
        opinion_range (tuple): 观点值的范围 (min, max)。
    Returns:
        xp.ndarray: 包含每个节点初始观点值的数组。
    """
    return xp.random.uniform(opinion_range[0], opinion_range[1], num_nodes)

def initialize_trust_thresholds(num_nodes, epsilon_base, epsilon_range=(0.0, 1.0)):
    """
    初始化所有节点的信任阈值。
    Args:
        num_nodes (int): 节点数量。
        epsilon_base (float or xp.ndarray): 基础信任阈值，可以是单个值或每个节点的数组。
        epsilon_range (tuple): 信任阈值的范围 (min, max)。
    Returns:
        xp.ndarray: 包含每个节点初始信任阈值的数组。
    """
    if isinstance(epsilon_base, (int, float)):
        thresholds = xp.full(num_nodes, epsilon_base)
    elif isinstance(epsilon_base, xp.ndarray) and epsilon_base.shape == (num_nodes,):
        thresholds = epsilon_base
    else:
        raise ValueError("epsilon_base 必须是浮点数或与节点数匹配的 numpy 数组。")
    
    # 确保信任阈值在有效范围内
    thresholds = xp.clip(thresholds, epsilon_range[0], epsilon_range[1])
    return thresholds

def calculate_opinion_difference(opinion1, opinion2):
    """
    计算两个观点之间的绝对差值。
    """
    return xp.abs(opinion1 - opinion2)

def update_pairwise_opinion(opinion_i, opinion_j, mu):
    """
    根据 Deffuant-Weisbuch 模型更新两个观点。
    """
    new_opinion_i = opinion_i + mu * (opinion_j - opinion_i)
    new_opinion_j = opinion_j + mu * (opinion_i - opinion_j)
    return new_opinion_i, new_opinion_j

def update_three_body_opinion(opinion_i, opinion_j, opinion_k):
    """
    更新三主体交互的观点（取平均值）。
    """
    avg_opinion = (opinion_i + opinion_j + opinion_k) / 3.0
    return avg_opinion, avg_opinion, avg_opinion

def get_neighbors(graph, node_id):
    """
    获取指定节点的邻居。
    Args:
        graph (networkx.Graph): 网络图。
        node_id: 节点ID。
    Returns:
        list: 邻居节点ID列表。
    """
    return list(graph.neighbors(node_id))

def dynamic_epsilon_extremity(opinion_i, epsilon_0, alpha, opinion_center=0.5):
    """
    根据观点极端性更新信任阈值。
    Args:
        opinion_i (float): 当前观点。
        epsilon_0 (float): 初始信任阈值。
        alpha (float): 调节强度。
        opinion_center (float): 观点中心点，默认为 0.5。
    Returns:
        float: 更新后的信任阈值。
    """
    return epsilon_0 * (1 - alpha * xp.abs(opinion_i - opinion_center))

def dynamic_epsilon_time_evolution(epsilon_i_0, beta, t):
    """
    根据时间演化更新信任阈值。
    Args:
        epsilon_i_0 (float): 初始信任阈值。
        beta (float): 时间演化速度。
        t (int): 当前时间步。
    Returns:
        float: 更新后的信任阈值。
    """
    return epsilon_i_0 + beta * xp.log(1 + t)

def is_within_trust_threshold(opinion_i, opinion_j, epsilon_i):
    """
    检查观点差异是否在信任阈值内。
    """
    return calculate_opinion_difference(opinion_i, opinion_j) <= epsilon_i

def clip_opinions(opinions, opinion_range=(0.0, 1.0)):
    """
    将观点裁剪到指定范围。
    """
    return xp.clip(opinions, opinion_range[0], opinion_range[1])

def initialize_poisson_events(st_config, sim_params, num_layers, xp):
    """
    根据泊松分布生成一系列随机事件。
    【更新】：为每个事件随机分配一个影响层。
    """
    poisson_rate = st_config.get('poisson_rate', 0.1)
    max_t = sim_params.get('max_iterations', 100)
    
    num_events = xp.random.poisson(poisson_rate * max_t)
    num_events = int(num_events) # 将 ndarray 转换为 int
    print(f"泊松事件生成器: 在 {max_t} 迭代中，根据速率 {poisson_rate} 生成了 {num_events} 个独立事件。")

    spatial_range = st_config.get('spatial_range', [[0, 1], [0, 1]])
    
    events = []
    if num_layers == 0:
        print("警告: 网络层数为 0，无法生成任何事件。")
        return []

    for _ in range(num_events):
        event_t0 = random.randint(0, max_t - 1)
        event_center_x = random.uniform(spatial_range[0][0], spatial_range[0][1])
        event_center_y = random.uniform(spatial_range[1][0], spatial_range[1][1])
        # 【新增】为事件随机指定一个层
        affected_layer = random.randint(0, num_layers - 1)
        
        events.append({
            't0': event_t0,
            'center': xp.array([event_center_x, event_center_y]),
            'layer_index': affected_layer  # <-- 新增记录
        })
    
    events.sort(key=lambda e: e['t0'])
    
    if events:
        print("生成的部分事件示例:")
        for event in events[:min(5, len(events))]:
            center_str = f"({event['center'][0]:.2f}, {event['center'][1]:.2f})"
            # 【更新】打印输出中也包含层信息
            print(f"  - 事件爆发于 t={event['t0']}, 位置={center_str}, 影响层={event['layer_index']}")
            
    return events


def calculate_event_influence(node_id, events, current_iteration, graphs, event_decay, xp):
    """
    为单个节点计算当前时刻由【所有已爆发事件】叠加产生的总时空影响因子 (0到1之间)。
    """
    t = current_iteration
    total_influence = xp.array(0.0, dtype=xp.float64)

    node_pos = graphs[0].nodes[node_id].get('pos')
    if node_pos is None:
        # 注意：这里的警告机制简化了，不再使用 self._warned_no_pos 标志
        # 每次遇到没有 'pos' 的节点都会警告一次，但在一个大规模仿真中通常只会显示前几次
        print(f"警告: 节点 {node_id} 缺少 'pos' 属性，时空效应将对该节点无效。")
        return 0.0
    
    node_pos_arr = xp.array(node_pos)
    
    alpha = event_decay['alpha']
    beta = event_decay['beta']
    
    for event in events:
        if t < event['t0']:
            continue

        distance_r = xp.linalg.norm(node_pos_arr - event['center'])
        influence_from_one_event = xp.exp(-alpha * distance_r) * xp.exp(-beta * (t - event['t0']))
        total_influence += influence_from_one_event
        
    return float(xp.clip(total_influence, 0.0, 1.0))