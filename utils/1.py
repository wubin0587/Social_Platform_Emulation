import os

# --- 1. 定义网络模型 ---
network_models = {
    "1ra": {
        "description": "[Aligned] 时空仿真: 1层 随机网络",
        "layers": "    - {type: random, params: {p: 0.002}}"
    },
    "1sw": {
        "description": "[Aligned] 时空仿真: 1层 小世界网络",
        "layers": "    - {type: small_world, params: {k: 10, beta: 0.1}}"
    },
    "1sf": {
        "description": "[Aligned] 时空仿真: 1层 无标度网络",
        "layers": "    - {type: scale_free, params: {m: 3}}"
    },
    "9sw_1sf": {
        "description": "[Aligned] 时空仿真: 10层 (9sw+1sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 9 + 
                             ["    - {type: scale_free, params: {m: 3}}"])
    },
    "1sw_9sf": {
        "description": "[Aligned] 时空仿真: 10层 (1sw+9sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] + 
                             ["    - {type: scale_free, params: {m: 3}}"] * 9)
    },
    "5sw_5sf": {
        "description": "[Aligned] 时空仿真: 10层 (5sw+5sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 5 + 
                             ["    - {type: scale_free, params: {m: 3}}"] * 5)
    },
    "10ra": {
        "description": "[Aligned] 时空仿真: 10层 随机网络",
        "layers": "\n".join(["    - {type: random, params: {p: 0.002}}"] * 10)
    },
    "10sw": {
        "description": "[Aligned] 时空仿真: 10层 小世界网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 10)
    },
    "10sf": {
        "description": "[Aligned] 时空仿真: 10层 无标度网络",
        "layers": "\n".join(["    - {type: scale_free, params: {m: 3}}"] * 10)
    }
}

# --- 2. 定义地理位置信息 ---
geo_locations = {
    "multi_clustered": {
        "description": "模式: multi_clustered (多核心)",
        "config": """  position_distribution:
    type: 'multi_clustered'
    hotspots: 
      - {center: [0.25, 0.25], radius: 0.2, ratio: 0.45}
      - {center: [0.75, 0.75], radius: 0.15, ratio: 0.35}"""
    },
    "uniform": {
        "description": "模式: uniform (均匀分布)",
        "config": """  position_distribution:
    type: 'uniform'"""
    },
    "clustered": {
        "description": "模式: clustered (单核心)",
        "config": """  position_distribution:
    type: 'clustered'
    hotspot: {center: [0.5, 0.5], radius: 0.3, ratio: 0.8}"""
    },
    "gaussian": {
        "description": "模式: gaussian (高斯分布)",
        "config": """  position_distribution:
    type: 'gaussian'
    gaussian_params: {mean: [0.5, 0.5], cov: [[0.03, 0], [0, 0.03]]}"""
    },
    "grid": {
        "description": "模式: grid (网格分布)",
        "config": """  position_distribution:
    type: 'grid'
    grid_params: {rows: 32, cols: 32, jitter: 0.02}"""
    },
    "linear": {
        "description": "模式: linear (沿线分布)",
        "config": """  position_distribution:
    type: 'linear'
    linear_params: {start: [0.1, 0.9], end: [0.9, 0.1], width: 0.1}"""
    },
    "concentric_circles": {
        "description": "模式: concentric_circles (同心圆分布)",
        "config": """  position_distribution:
    type: 'concentric_circles'
    circle_params:
      center: [0.5, 0.5]
      rings:
        - {radius: 0.2, width: 0.1, ratio: 0.4}
        - {radius: 0.4, width: 0.1, ratio: 0.5}"""
    },
    "wedge": {
        "description": "模式: wedge (扇形分布)",
        "config": """  position_distribution:
    type: 'wedge'
    wedge_params: {center: [0.5, 0.5], radius: 0.45, start_angle: 0, end_angle: 120}"""
    }
}

# --- 3. 定义文件模板 ---
# 【核心修改】
# 1. 将所有网络相关的配置（layers, position_distribution）都放在一个 network 键下。
# 2. 删除了文件末尾的 '---' 分隔符。
file_template = """# D:\\Social_Platform_Emulation\\config\\{filename}
# {description}

# --- 1. 网络与地理位置统一配置 ---
network:
  num_nodes: 1000
  layers:
{layers_config}

  # {geo_description}
{geo_config_block}

# --- 2. 仿真模型参数 ---
simulation_params:
  max_iterations: 400
  record_interval: 10

  convergence_params:
    enabled: true        # 开启功能
    threshold: 0.0001    # 总观点变化阈值
    patience: 100        # 连续100次低于阈值则停止

  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}
  
  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.01
    c_kl_global: 0.01

  spatiotemporal_params:
    poisson_rate: 0.1
    spatial_range: [[0.0, 1.0], [0.0, 1.0]]
    alpha: 0.8
    beta: 0.05
    interaction_prob: {{base: 0.7, gain: 0.3}}
    trust_scope: {{gain: 0.15}}
    learning_rate: {{gain: 0.2}}
"""

def generate_files():
    """主函数，用于生成所有配置文件"""
    output_dir = "generated_configs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始生成配置文件，将保存到 '{output_dir}' 文件夹中...")

    single_layer_nets = ["1ra", "1sw", "1sf"]

    for net_key, net_value in network_models.items():
        
        coupling_enabled_value = 'false' if net_key in single_layer_nets else 'true'
            
        for geo_key, geo_value in geo_locations.items():
            
            filename = f"st_{net_key}.{geo_key}.yaml"
            print(f"正在生成: {filename}")
            
            # 【修改】模板的填充方式也需要调整，因为不再有独立的 network 块
            final_content = file_template.format(
                filename=filename,
                description=net_value['description'],
                layers_config=net_value['layers'],
                coupling_enabled=coupling_enabled_value,
                geo_description=geo_value['description'],
                geo_config_block=geo_value['config']
            )
            
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)

    print(f"\n所有 {len(network_models) * len(geo_locations)} 个配置文件已成功生成！")

if __name__ == "__main__":
    generate_files()