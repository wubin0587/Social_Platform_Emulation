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

# --- 3. Define Spatiotemporal Template (Copied from main script) ---
spatiotemporal_template = """# {output_dir}/{filename}
# [CONTROL GROUP] 时空仿真 (无事件): {net_description}
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
    poisson_rate: 0.0  # <<< THIS IS THE ONLY CHANGE FOR THE CONTROL GROUP
    spatial_range: [[0.0, 1.0], [0.0, 1.0]]
    alpha: 10
    beta: 0.04
    interaction_prob: {{base: 0.4, gain: 0.3}}
    trust_scope: {{gain: 0.2}}
    learning_rate: {{gain: 0.2}}
"""

def generate_control_group_files():
    """主函数，为时空模型生成无事件的对照组配置文件"""
    
    total_files_generated = 0
    single_layer_nets = ["1ra", "1sw", "1sf"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义专门的输出文件夹
    output_dir_name = "spatiotemporal_control_no_events"
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- 开始生成 [Spatiotemporal Control Group] 配置文件 ---")
    print(f"--- 所有文件将保存到 '{output_dir}' 文件夹中 ---")
    
    # 遍历每一种网络模型
    for net_key, net_value in network_models.items():
        coupling_enabled_value = 'false' if net_key in single_layer_nets else 'true'

        # 遍历每一种地理位置分布
        for geo_key, geo_value in geo_locations.items():
            
            # 准备通用的模板参数
            filename = f"st_control_{net_key}.{geo_key}.yaml"
            params = {
                "output_dir": output_dir_name,
                "filename": filename,
                "net_description": net_value['description'],
                "layers_config": net_value['layers'],
                "coupling_enabled": coupling_enabled_value,
                "geo_description": geo_value['description'],
                "geo_config_block": geo_value['config']
            }
            
            # 渲染模板并写入文件
            # 注意：模板中已将 poisson_rate 硬编码为 0.0
            final_content = spatiotemporal_template.format(**params)
            
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            total_files_generated += 1
            
    print(f"\n[Spatiotemporal Control Group] 生成完成！")
    print(f"==================================================")
    print(f"总计: {total_files_generated} 个对照组文件已成功生成。")


if __name__ == "__main__":
    generate_control_group_files()