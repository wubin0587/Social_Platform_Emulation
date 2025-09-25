import os

# --- 1. 定义网络模型 (通用，保持不变) ---
network_models = {
    "1ra": {
        "description": "1层 随机网络",
        "layers": "    - {type: random, params: {p: 0.01001}}"
    },
    "1sw": {
        "description": "1层 小世界网络",
        "layers": "    - {type: small_world, params: {k: 10, beta: 0.1}}"
    },
    "1sf": {
        "description": "1层 无标度网络",
        "layers": "    - {type: scale_free, params: {m: 5}}"
    },
    "3sw_1sf": {
        "description": "4层 (3sw+1sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 3 +
                             ["    - {type: scale_free, params: {m: 5}}"])
    },
    "1sw_3sf": {
        "description": "4层 (1sw+3sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] +
                             ["    - {type: scale_free, params: {m: 5}}"] * 3)
    },
    "2sw_2sf": {
        "description": "4层 (2sw+2sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 2 +
                             ["    - {type: scale_free, params: {m: 5}}"] * 2)
    },
    "4ra": {
        "description": "4层 随机网络",
        "layers": "\n".join(["    - {type: random, params: {p: 0.01001}}"] * 4)
    },
    "4sw": {
        "description": "4层 小世界网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 4)
    },
    "4sf": {
        "description": "4层 无标度网络",
        "layers": "\n".join(["    - {type: scale_free, params: {m: 5}}"] * 4)
    }
}

# --- 2. 定义地理位置信息 (仅用于时空模型) ---
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

# 模板1: 用于时空仿真 (Spatiotemporal)
spatiotemporal_template = """# {output_dir}/{filename}
# [Aligned] 时空仿真: {net_description}
# {geo_description}

# --- 1. 网络与地理位置统一配置 ---
network:
  num_nodes: 1000
  layers:
{layers_config}

  # 地理位置分布
{geo_config_block}

# --- 2. 仿真模型参数 ---
simulation_params:
  max_iterations: 800
  record_interval: 10

  convergence_params:
    enabled: true
    threshold: 0.001
    patience: 20

  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}

  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.002
    c_kl_global: 0.002

  spatiotemporal_params:
    poisson_rate: 0.025
    spatial_range: [[0.0, 1.0], [0.0, 1.0]]
    alpha: 2
    beta: 0.04
    interaction_prob: {{base: 0.4, gain: 0.3}}
    trust_scope: {{gain: 0.3}}
    learning_rate: {{gain: 0.2}}

    spatial_neighbors:
      enabled: true          # 是否启用空间邻居发现
      base_radius: 0.02      # 节点的日常空间感知半径 (无事件影响时)
      radius_gain: 0.05       # 事件对空间感知半径的增益系数
"""

# 模板2: 用于基础 DW 模型 (Basic)
basic_template = """# {output_dir}/{filename}
# [Aligned] 基础DW模型: {net_description}

# --- 1. 网络配置 ---
network:
  num_nodes: 1000
  layers:
{layers_config}

# --- 2. 仿真模型参数 ---
simulation_params:
  max_iterations: 800
  record_interval: 10

  convergence_params:
    enabled: true
    threshold: 0.001
    patience: 20

  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}

  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.002
    c_kl_global: 0.002
"""

# --- 新增 ---
# 模板3: DW3D (Extremity Inhibition) 模型
dw3d_exh_template = """# {output_dir}/{filename}
# [Aligned] DW3D (Extremity Inhibition): {net_description}

# --- 1. 网络配置 ---
network:
  num_nodes: 1000
  layers:
{layers_config}

# --- 2. 仿真模型参数 ---
simulation_params:
  max_iterations: 800
  record_interval: 10

  convergence_params:
    enabled: true
    threshold: 0.001
    patience: 20

  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}

  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.002
    c_kl_global: 0.002

# --- 3. DW3D 扩展参数 ---
  dw3d_extensions:
    interaction_mode: 'random'

    dynamic_epsilon:
      enabled: true
      extremity_inhibition: {{enabled: true, alpha: 0.5, opinion_center: 0.5}}
      time_evolution: {{enabled: false, beta: 0.00001}}

    dynamic_network:
      enabled: true
      disconnect_threshold_factor: 1.8
      reconnect_probability: 0.01
      reconnect_opinion_threshold: 0.15
"""

# --- 新增 ---
# 模板4: DW3D (Time Evolution) 模型
dw3d_time_template = """# {output_dir}/{filename}
# [Aligned] DW3D (Time Evolution): {net_description}

# --- 1. 网络配置 ---
network:
  num_nodes: 1000
  layers:
{layers_config}

# --- 2. 仿真模型参数 ---
simulation_params:
  max_iterations: 800
  record_interval: 10

  convergence_params:
    enabled: true
    threshold: 0.001
    patience: 20

  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}

  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.002
    c_kl_global: 0.002

# --- 3. DW3D 扩展参数 ---
  dw3d_extensions:
    interaction_mode: 'random'

    dynamic_epsilon:
      enabled: true
      extremity_inhibition: {{enabled: false, alpha: 0.5, opinion_center: 0.5}}
      time_evolution: {{enabled: true, beta: 0.00001}}
      
    dynamic_network:
      enabled: true
      disconnect_threshold_factor: 1.8
      reconnect_probability: 0.01
      reconnect_opinion_threshold: 0.15
"""


# --- 4. 定义所有需要生成的配置场景 ---
config_scenarios = [
    {
        "name": "时空模型 (Spatiotemporal)",
        "prefix": "st",
        "output_dir": "spatiotemporal",
        "template": spatiotemporal_template,
        "use_geo": True
    },
    {
        "name": "基础模型 (Basic)",
        "prefix": "basic",
        "output_dir": "basic",
        "template": basic_template,
        "use_geo": False
    },
    {
        "name": "DW3D模型 (Extremity Inhibition)",
        "prefix": "dw3d_exh",
        "output_dir": "dw3d_exh",
        "template": dw3d_exh_template,
        "use_geo": False
    },
    {
        "name": "DW3D模型 (Time Evolution)",
        "prefix": "dw3d_time",
        "output_dir": "dw3d_time",
        "template": dw3d_time_template,
        "use_geo": False
    }
]

def generate_files():
    """主函数，根据 `config_scenarios` 中的定义生成所有配置文件"""
    total_files_generated = 0
    single_layer_nets = ["1ra", "1sw", "1sf"]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 遍历每一种配置场景
    for scenario in config_scenarios:
        output_dir = os.path.join(script_dir, scenario["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- 开始生成 [{scenario['name']}] 配置文件，将保存到 '{output_dir}' 文件夹中 ---")
        
        scenario_files_count = 0
        
        # 遍历每一种网络模型
        for net_key, net_value in network_models.items():
            coupling_enabled_value = 'false' if net_key in single_layer_nets else 'true'

            # 准备通用的模板参数
            params = {
                "output_dir": scenario["output_dir"],
                "net_description": net_value['description'],
                "layers_config": net_value['layers'],
                "coupling_enabled": coupling_enabled_value,
                "geo_description": "",
                "geo_config_block": ""
            }
            # 根据场景决定是否处理地理位置信息
            if scenario["use_geo"]:
                for geo_key, geo_value in geo_locations.items():
                    filename = f"{scenario['prefix']}_{net_key}.{geo_key}.yaml"
                    
                    # 更新地理位置相关的参数
                    params['filename'] = filename
                    params['geo_description'] = geo_value['description']
                    params['geo_config_block'] = geo_value['config']
                    
                    # 渲染模板并写入文件
                    final_content = scenario["template"].format(**params)
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(final_content)
                    scenario_files_count += 1
            else:
                filename = f"{scenario['prefix']}_{net_key}.yaml"
                params['filename'] = filename
                
                # 渲染模板并写入文件
                final_content = scenario["template"].format(**params)
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                scenario_files_count += 1
        
        print(f"[{scenario['name']}] 完成，共生成 {scenario_files_count} 个文件。\n")
        total_files_generated += scenario_files_count

    print(f"==================================================")
    print(f"所有配置文件已成功生成！总计: {total_files_generated} 个文件。")


if __name__ == "__main__":
    generate_files()