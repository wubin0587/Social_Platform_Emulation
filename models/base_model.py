import networkx as nx
import yaml

class BaseModel:
    """
    基类，提供随机网络、小世界网络和无标度网络的生成方法。
    """
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """加载配置。"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def generate_small_world_network(self, n, k, beta):
        """
        生成小世界网络 (Watts–Strogatz Model)。
        Args:
            n (int): 节点数。
            k (int): 每个节点初始连接的最近邻节点数 (必须是偶数)。
            beta (float): 重连概率。
        Returns:
            networkx.Graph: 生成的小世界网络。
        """
        if k % 2 != 0:
            raise ValueError("参数 k 必须是偶数。")
        if not (0 <= beta <= 1):
            raise ValueError("重连概率 beta 必须在 [0, 1] 之间。")
        return nx.watts_strogatz_graph(n, k, beta)

    def generate_scale_free_network(self, n, m):
        """
        生成无标度网络 (Barabási–Albert Model)。
        Args:
            n (int): 节点总数。
            m (int): 每个新加入节点连接到已有节点的边数。
        Returns:
            networkx.Graph: 生成的无标度网络。
        """
        if m >= n:
            raise ValueError("参数 m 必须小于节点总数 n。")
        return nx.barabasi_albert_graph(n, m)
    
    def generate_random_network(self, n, p):
        """
        生成随机网络 (Erdős–Rényi Model)。
        Args:
            n (int): 节点数。
            p (float): 连边概率。
        Returns:
            networkx.Graph: 生成的随机网络。
        """
        if not (0 <= p <= 1):
            raise ValueError("连边概率 p 必须在 [0, 1] 之间。")
        return nx.erdos_renyi_graph(n, p)
