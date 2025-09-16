import numpy as np
import random
import networkx as nx
from models.multilayer_model import MultilayerModel # 导入 MultilayerModel

# --- BaseModel 和 MultilayerModel 的代码保持不变 ---
# 这里假设 BaseModel 和 MultilayerModel 的代码如您提供的那样，分别在各自的文件中。


class SpatioTemporalModel(MultilayerModel):
    """
    多层时空网络模型生成器。

    该类继承自 MultilayerModel，专注于生成一个【多层的】网络，
    并且网络中的每一个节点都带有二维地理空间坐标 ('pos') 属性。

    它首先利用 MultilayerModel 的能力配置和生成多层的拓扑结构，
    然后为所有节点生成一套统一的空间坐标，并将这些坐标附加到每一层的
    对应节点上。
    """
    def __init__(self, num_nodes, config_path=None):
        """
        初始化多层时空网络模型生成器。
        
        Args:
            num_nodes (int): 网络中的节点总数。
            config_path (str, optional): 配置文件路径。
        """
        # 调用父类 (MultilayerModel) 的构造函数
        super().__init__(num_nodes, config_path)
        self.positions = {} # 存储所有节点共享的坐标
        print("SpatioTemporalModel (multi-layer) 生成器已初始化。")

    def _generate_positions(self, position_config=None):
        """
        为网络中的所有节点生成一次共享的地理空间坐标。
        
        该函数根据 'type' 参数选择不同的地理分布模式，并从配置字典中
        读取相应模式所需的参数。

        支持的模式包括:
        - uniform: 全局均匀分布。
        - clustered: 单个圆形热点聚集。
        - multi_clustered: 多个大小、比例不同的圆形热点。
        - gaussian: 以某点为中心的正态(高斯)分布。
        - grid: 带随机抖动的规则网格分布。
        - linear: 沿指定线段的带状分布。
        - concentric_circles: 沿多个同心圆环的分布。
        - wedge: 在一个扇形(楔形)区域内的分布。
        
        Args:
            position_config (dict, optional): 节点位置分布的配置字典。
        """
        if position_config is None:
            position_config = {}
        
        dist_type = position_config.get('type', 'uniform')
        print(f"正在为 {self.num_nodes} 个节点生成共享的地理位置 (模式: {dist_type})...")
        self.positions.clear()

        # --- 模式1: 全局均匀分布 ---
        if dist_type == 'uniform':
            for node_id in range(self.num_nodes):
                self.positions[node_id] = (random.random(), random.random())

        # --- 模式2: 单核心/单热点聚集 ---
        elif dist_type == 'clustered':
            #
            # YAML params: hotspot {center, radius, ratio}
            #
            hotspot = position_config.get('hotspot', {})
            center = np.array(hotspot.get('center', [0.5, 0.5]))
            radius = hotspot.get('radius', 0.2)
            ratio = hotspot.get('ratio', 0.8)

            for node_id in range(self.num_nodes):
                if random.random() < ratio:
                    # 在热点区域内生成
                    r = radius * np.sqrt(random.random()) # 保证区域内均匀
                    theta = random.random() * 2 * np.pi
                    x = center[0] + r * np.cos(theta)
                    y = center[1] + r * np.sin(theta)
                    self.positions[node_id] = (np.clip(x, 0, 1), np.clip(y, 0, 1))
                else:
                    # 全局均匀分布
                    self.positions[node_id] = (random.random(), random.random())
        
        # --- 模式3: 多核心/多热点聚集 ---
        elif dist_type == 'multi_clustered':
            #
            # YAML params: hotspots [ {center, radius, ratio}, ... ]
            #
            hotspots = position_config.get('hotspots', [])
            ratios = [h.get('ratio', 0) for h in hotspots]
            
            for node_id in range(self.num_nodes):
                p = random.random()
                cumulative_ratio = 0.0
                in_any_hotspot = False
                
                for i, hotspot in enumerate(hotspots):
                    cumulative_ratio += ratios[i]
                    if p < cumulative_ratio:
                        center = np.array(hotspot.get('center', [0.5, 0.5]))
                        radius = hotspot.get('radius', 0.1)
                        r = radius * np.sqrt(random.random())
                        theta = random.random() * 2 * np.pi
                        x = center[0] + r * np.cos(theta)
                        y = center[1] + r * np.sin(theta)
                        self.positions[node_id] = (np.clip(x, 0, 1), np.clip(y, 0, 1))
                        in_any_hotspot = True
                        break
                
                if not in_any_hotspot:
                    # 未落入任何热点，全局均匀分布
                    self.positions[node_id] = (random.random(), random.random())

        # --- 模式4: 高斯/正态分布 ---
        elif dist_type == 'gaussian':
            #
            # YAML params: gaussian_params {mean, cov}
            #
            gauss_params = position_config.get('gaussian_params', {})
            mean = gauss_params.get('mean', [0.5, 0.5])
            cov = gauss_params.get('cov', [[0.02, 0], [0, 0.02]])
            
            points = np.random.multivariate_normal(mean, cov, self.num_nodes)
            for node_id, pos in enumerate(points):
                self.positions[node_id] = (np.clip(pos[0], 0, 1), np.clip(pos[1], 0, 1))

        # --- 模式5: 网格分布 ---
        elif dist_type == 'grid':
            #
            # YAML params: grid_params {rows, cols, jitter}
            #
            grid_params = position_config.get('grid_params', {})
            rows = grid_params.get('rows', int(np.sqrt(self.num_nodes)))
            cols = grid_params.get('cols', int(np.sqrt(self.num_nodes)))
            jitter = grid_params.get('jitter', 0.05)
            
            if rows * cols < self.num_nodes:
                print(f"警告: 网格 {rows}x{cols} 小于节点数 {self.num_nodes}。部分节点将无法放置。")

            node_id = 0
            for r in range(rows):
                for c in range(cols):
                    if node_id >= self.num_nodes: break
                    x = (c / (cols - 1)) if cols > 1 else 0.5
                    y = (r / (rows - 1)) if rows > 1 else 0.5
                    x += (random.random() - 0.5) * jitter
                    y += (random.random() - 0.5) * jitter
                    self.positions[node_id] = (np.clip(x, 0, 1), np.clip(y, 0, 1))
                    node_id += 1
        
        # --- 模式6: 沿线分布 ---
        elif dist_type == 'linear':
            #
            # YAML params: linear_params {start, end, width}
            #
            lin_params = position_config.get('linear_params', {})
            p1 = np.array(lin_params.get('start', [0.1, 0.1]))
            p2 = np.array(lin_params.get('end', [0.9, 0.9]))
            width = lin_params.get('width', 0.1)
            
            v = p2 - p1
            v_perp = np.array([-v[1], v[0]])
            v_perp_norm = v_perp / (np.linalg.norm(v_perp) + 1e-9)

            for node_id in range(self.num_nodes):
                t = random.random()
                d = (random.random() - 0.5) * width
                pos = p1 + t * v + d * v_perp_norm
                self.positions[node_id] = (np.clip(pos[0], 0, 1), np.clip(pos[1], 0, 1))

        # --- 模式7: 同心圆分布 ---
        elif dist_type == 'concentric_circles':
            #
            # YAML params: circle_params {center, rings: [{radius, width, ratio}, ...]}
            #
            circ_params = position_config.get('circle_params', {})
            center = np.array(circ_params.get('center', [0.5, 0.5]))
            rings = circ_params.get('rings', [])
            
            for node_id in range(self.num_nodes):
                p = random.random()
                cumulative_ratio = 0.0
                in_any_ring = False
                
                for ring in rings:
                    ratio = ring.get('ratio', 0)
                    cumulative_ratio += ratio
                    if p < cumulative_ratio:
                        r_center = ring.get('radius', 0.2)
                        r_width = ring.get('width', 0.05)
                        r_inner = max(0, r_center - r_width / 2)
                        r_outer = r_center + r_width / 2
                        
                        # 按面积均匀采样半径
                        r = np.sqrt(random.uniform(r_inner**2, r_outer**2))
                        theta = random.random() * 2 * np.pi
                        
                        x = center[0] + r * np.cos(theta)
                        y = center[1] + r * np.sin(theta)
                        self.positions[node_id] = (np.clip(x, 0, 1), np.clip(y, 0, 1))
                        in_any_ring = True
                        break
                
                if not in_any_ring:
                    self.positions[node_id] = (random.random(), random.random())

        # --- 模式8: 扇形/楔形分布 ---
        elif dist_type == 'wedge':
            #
            # YAML params: wedge_params {center, radius, start_angle, end_angle}
            #
            wedge_params = position_config.get('wedge_params', {})
            center = np.array(wedge_params.get('center', [0.5, 0.5]))
            radius = wedge_params.get('radius', 0.4)
            start_angle = np.deg2rad(wedge_params.get('start_angle', 0))
            end_angle = np.deg2rad(wedge_params.get('end_angle', 90))

            for node_id in range(self.num_nodes):
                r = radius * np.sqrt(random.random())
                theta = random.uniform(start_angle, end_angle)
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                self.positions[node_id] = (np.clip(x, 0, 1), np.clip(y, 0, 1))

        # --- 默认/回退情况 ---
        else:
            print(f"警告: 未知的分布类型 '{dist_type}'，将使用 'uniform'。")
            for node_id in range(self.num_nodes):
                self.positions[node_id] = (random.random(), random.random())

    def _attach_positions_to_graph(self, graph: nx.Graph):
        """
        将预先生成的共享坐标附加到单个图层的所有节点上。
        """
        for node_id, pos in self.positions.items():
            if node_id in graph:
                graph.nodes[node_id]['pos'] = pos
        return graph

    def build_spatiotemporal_multilayer_network(self, position_config: dict = None) -> list:
        """
        构建一个完整的多层时空网络。

        这是一个便捷的封装方法，整合了多层网络拓扑生成和统一空间属性附加两个步骤。

        Args:
            position_config (dict, optional): 节点位置分布的配置。

        Returns:
            list: 一个包含多个 NetworkX 图对象的列表，每个图都带有 'pos' 节点属性。
        """
        # 步骤 1: 为所有节点生成一套共享的空间坐标
        self._generate_positions(position_config)
        
        # 步骤 2: 调用父类的方法，根据已添加的配置构建基础的多层网络拓扑
        print("正在构建多层网络拓扑结构...")
        base_layers = super().build_multilayer_network()
        
        # 步骤 3: 遍历每一个网络层，并为该层附加空间坐标
        print("正在为多层网络的每一层附加空间属性...")
        for i, layer_graph in enumerate(self.layers):
            self._attach_positions_to_graph(layer_graph)
            print(f"  - 第 {i+1} 层空间属性附加完毕。")
            
        return self.layers