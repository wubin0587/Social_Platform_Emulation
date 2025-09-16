from models.base_model import BaseModel

class MultilayerModel(BaseModel):
    """
    自定义多层社交网络框架，能够自主创建多层网络结构。
    """
    def __init__(self, num_nodes, config_path=None):
        super().__init__(config_path)
        self.num_nodes = num_nodes
        self.layers = [] # 存储每个网络层
        self.layer_configs = [] # 存储每个网络层的配置

    def add_layer(self, layer_type, **kwargs):
        """
        添加一个网络层。
        Args:
            layer_type (str): 网络类型 ('random', 'small_world', 'scale_free')。
            **kwargs: 对应网络类型的参数。
        """
        self.layer_configs.append({'type': layer_type, 'params': kwargs})

    def build_multilayer_network(self):
        """
        根据添加的层配置构建多层网络。
        Returns:
            list: 包含每个层 NetworkX 图对象的列表。
        """
        self.layers = []
        for config in self.layer_configs:
            layer_type = config['type']
            params = config['params']
            
            graph = None
            if layer_type == 'random':
                graph = self.generate_random_network(n=self.num_nodes, **params)
            elif layer_type == 'small_world':
                graph = self.generate_small_world_network(n=self.num_nodes, **params)
            elif layer_type == 'scale_free':
                graph = self.generate_scale_free_network(n=self.num_nodes, **params)
            else:
                raise ValueError(f"不支持的网络类型: {layer_type}")
            
            self.layers.append(graph)
        return self.layers

    def get_layer(self, layer_index):
        """获取指定索引的网络层。"""
        if 0 <= layer_index < len(self.layers):
            return self.layers[layer_index]
        else:
            raise IndexError("层索引超出范围。")

    def describe_multilayer_network(self):
        """
        描述当前多层网络的结构。
        """
        description = "多层网络结构:\n"
        for i, config in enumerate(self.layer_configs):
            description += f"  第 {i+1} 层: 类型='{config['type']}', 参数={config['params']}\n"
        return description