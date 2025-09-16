import networkx as nx
import numpy as np
from scipy.stats import skew, kurtosis
from collections import Counter

def shortest_path_distance(graph: nx.Graph, node1: int, node2: int) -> float:
    """
    计算两个节点之间的最短路径距离（最小跳数）。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        node1 (int): 起始节点 ID。
        node2 (int): 目标节点 ID。
        
    Returns:
        float: 两个节点之间的最短路径长度。如果节点不连通，则返回无穷大 (np.inf)。
    """
    try:
        return nx.shortest_path_length(graph, source=node1, target=node2)
    except nx.NetworkXNoPath:
        return np.inf

def weight_aware_structural_similarity(graph: nx.Graph, node1: int, node2: int) -> float:
    """
    计算节点间的权重感知结构相似度 (s_ij = exp(-d(i,j)))。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        node1 (int): 起始节点 ID。
        node2 (int): 目标节点 ID。
        
    Returns:
        float: 结构相似度值。如果节点不连通，距离为无穷大，相似度为 0。
    """
    distance = shortest_path_distance(graph, node1, node2)
    return np.exp(-distance)

def homophilic_bimodality_coefficient(graph: nx.Graph, opinions: np.ndarray) -> float:
    """
    计算同质双峰系数 (BC_hom)，用于量化结构化的极化团体。
    当 BC_hom > 5/9 时，通常认为存在显著的双峰分布（极化）。

    Args:
        graph (nx.Graph): NetworkX 图对象。
        opinions (np.ndarray): 一维 NumPy 数组，索引对应节点 ID，值为节点的观点。
        
    Returns:
        float: 计算出的同质双峰系数值。如果无法计算（例如，网络中没有边），则返回 np.nan。
    """
    node_b_list = []
    neighbor_b_nn_list = []

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue  # 跳过孤立节点

        node_opinion = opinions[node]
        neighbor_opinions = opinions[neighbors]
        neighbor_mean_opinion = np.mean(neighbor_opinions)
        
        node_b_list.append(node_opinion)
        neighbor_b_nn_list.append(neighbor_mean_opinion)

    if not node_b_list:
        # 如果没有节点有邻居，则无法计算
        return np.nan

    # 将数据点旋转 45°，并投影到新变量 b_dagger
    # b_dagger = (b - b_NN) / sqrt(2)
    # 由于我们只关心分布的形状，可以忽略常数因子 1/sqrt(2)
    b_dagger = np.array(node_b_list) - np.array(neighbor_b_nn_list)
    
    if np.std(b_dagger) == 0:
        # 如果所有差值都相同，不存在峰，返回0
        return 0.0

    n = len(b_dagger)
    m3 = skew(b_dagger)  # 样本偏度
    m4 = kurtosis(b_dagger, fisher=True) # 样本超额峰度 (Fisher's definition)
    
    # Bimodality Coefficient (BC) 的公式
    # BC = (skewness^2 + 1) / (kurtosis + 3)
    # 注意: scipy.stats.kurtosis 默认计算的是 "超额峰度" (excess kurtosis)，
    # 而 BC 公式需要的是 "峰度" (kurtosis)，关系为: kurtosis = excess_kurtosis + 3
    bc = (m3**2 + 1) / (m4 + 3)
    
    return bc

def opinion_aggregation_stability(opinions_t: np.ndarray, opinions_t1: np.ndarray) -> float:
    """
    计算系统的聚合稳定度，通过平均观点变动率来衡量。
    
    Args:
        opinions_t (np.ndarray): t 时刻的观点数组。
        opinions_t1 (np.ndarray): t+1 时刻的观点数组。
        
    Returns:
        float: 平均观点变动率。值越接近 0，系统越稳定。
    """
    if opinions_t.shape != opinions_t1.shape:
        raise ValueError("观点数组的形状必须匹配。")
    
    return np.mean(np.abs(opinions_t1 - opinions_t))

def network_density(graph: nx.Graph) -> float:
    """
    计算网络的密度。
    密度 = 实际边数 / 可能的最大边数。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        
    Returns:
        float: 网络的密度值，范围在 [0, 1]。
    """
    return nx.density(graph)

def average_clustering_coefficient(graph: nx.Graph) -> float:
    """
    计算网络的平均聚类系数。
    衡量节点的邻居之间相互连接的程度。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        
    Returns:
        float: 平均聚类系数值。
    """
    return nx.average_clustering(graph)

def average_shortest_path_length(graph: nx.Graph) -> float:
    """
    计算网络的平均最短路径长度。
    要求图是连通的。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        
    Returns:
        float: 平均最短路径长度。如果图不连通，会引发异常。
    """
    if not nx.is_connected(graph):
        print("警告: 图不是连通的。仅计算最大连通子图的平均最短路径。")
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph)
    return nx.average_shortest_path_length(graph)

def get_giant_component_size(graph: nx.Graph) -> int:
    """
    获取网络中最大连通子图（巨型组件）的大小。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        
    Returns:
        int: 巨型组件中的节点数量。
    """
    if not graph.nodes():
        return 0
    largest_cc = max(nx.connected_components(graph), key=len)
    return len(largest_cc)

def get_degree_distribution(graph: nx.Graph) -> dict:
    """
    获取网络的度分布。
    
    Args:
        graph (nx.Graph): NetworkX 图对象。
        
    Returns:
        dict: 一个字典，键是度数，值是拥有该度数的节点数量。
    """
    degrees = [d for n, d in graph.degree()]
    return Counter(degrees)

def opinion_variance(opinions: np.ndarray) -> float:
    """
    计算观点分布的方差。
    方差越小，共识程度越高。
    
    Args:
        opinions (np.ndarray): 观点数组。
        
    Returns:
        float: 观点的方差。
    """
    return np.var(opinions)

def number_of_opinion_clusters(opinions: np.ndarray, threshold: float = 0.05) -> int:
    """
    估算观点集群的数量。
    一个简单的实现：对观点进行排序，当相邻观点差异大于阈值时，认为是一个新的集群。
    
    Args:
        opinions (np.ndarray): 一维观点数组。
        threshold (float): 用于区分集群的阈值。
        
    Returns:
        int: 估算出的集群数量。
    """
    if len(opinions) == 0:
        return 0
    
    sorted_opinions = np.sort(opinions)
    diffs = np.diff(sorted_opinions)
    num_clusters = 1 + np.sum(diffs > threshold)
    return int(num_clusters)