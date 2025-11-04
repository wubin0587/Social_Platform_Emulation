import os
import re

# ==============================================================================
# 生成“敏感性分析” (Sensitivity Analysis) 的配置文件 (V2 - 完全独立版)
# 目的: 基于几个代表性原型，系统性地改变关键参数，批量生成配置文件。
# 版本说明: 此版本已内置所有必需的模板，无需外部文件，可直接运行。
# ==============================================================================

# --- 1. 内置所有必需的模板字符串 ---

basic_template_string = """# {output_dir}/{filename}
# [SENSITIVITY] 基础DW模型: {net_description}
# Parameter '{param_key}' set to '{param_value}'

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

spatiotemporal_template_string = """# {output_dir}/{filename}
# [SENSITIVITY] 时空仿真: {net_description}
# {geo_description}
# Parameter '{param_key}' set to '{param_value}'

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
    alpha: 10
    beta: 0.04
    interaction_prob: {{base: 0.4, gain: 0.3}}
    trust_scope: {{gain: 0.2}}
    learning_rate: {{gain: 0.2}}
"""


# --- 2. 内置所有必需的网络和地理位置定义 ---
network_models = {
    "1sw": {
        "description": "1层 小世界网络",
        "layers": "    - {type: small_world, params: {k: 10, beta: 0.1}}"
    },
    "2sw_2sf": {
        "description": "4层 (2sw+2sf) 混合网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 2 +
                             ["    - {type: scale_free, params: {m: 5}}"] * 2)
    },
    "4sw": {
        "description": "4层 小世界网络",
        "layers": "\n".join(["    - {type: small_world, params: {k: 10, beta: 0.1}}"] * 4)
    }
}
geo_locations = {
    "clustered": {
        "description": "模式: clustered (单核心)",
        "config": """  position_distribution:
    type: 'clustered'
    hotspot: {center: [0.5, 0.5], radius: 0.3, ratio: 0.8}"""
    }
}


# --- 3. 定义敏感性分析参数 ---
sensitivity_params = {
    'epsilon': {
        'target_path': 'simulation_params.initial_state.epsilon_base',
        'values': [0.05, 0.1, 0.15, 0.2, 0.25]
    },
    'mu': {
        'target_path': 'simulation_params.dw_params.mu',
        'values': [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'lambda': {
        'target_path': 'simulation_params.coupling_params.lambda',
        'values': [0.0, 0.001, 0.002, 0.005, 0.01],
        'requires_multilayer': True # 标记此参数仅对多层网络有效
    },
    'spatio_alpha': {
        'target_path': 'simulation_params.spatiotemporal_params.alpha',
        'values': [1, 2, 5, 10, 20],
        'requires_spatiotemporal': True # 标记此参数仅对时空模型有效
    }
}


# --- 4. 定义用于分析的代表性“原型” ---
archetypes = [
    {'prefix': 'basic', 'net_key': '1sw', 'template': basic_template_string},
    {'prefix': 'basic', 'net_key': '2sw_2sf', 'template': basic_template_string},
    {'prefix': 'st', 'net_key': '4sw', 'geo_key': 'clustered', 'template': spatiotemporal_template_string}
]


def set_param_value(template_content, path, value):
    """通过正则表达式安全地替换YAML字符串中的参数值。"""
    keys = path.split('.')
    current_level_content = template_content
    
    # 构建一个灵活的正则表达式来定位和替换
    # 例如路径: 'simulation_params.initial_state.epsilon_base'
    # 匹配 'epsilon_base:' 后面的值
    target_key = keys[-1]
    pattern = re.compile(rf"(^\s*{target_key}:\s+)([\d\.\-]+)", re.MULTILINE)

    if re.search(pattern, template_content):
        return re.sub(pattern, rf"\g<1>{value}", template_content)
    else:
        print(f"  [警告] 在模板中未找到参数路径 '{path}' 对应的键 '{target_key}'。文件可能未被正确修改。")
        return template_content

def generate_sensitivity_files():
    """主函数，生成所有敏感性分析文件。"""
    total_files = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "sensitivity_analysis")
    print("--- 开始生成 [Sensitivity Analysis] 配置文件 (独立版) ---")
    
    for param_key, param_info in sensitivity_params.items():
        param_path = param_info['target_path']
        param_values = param_info['values']
        
        output_dir = os.path.join(base_output_dir, f"sensitivity_{param_key}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n>> 正在为参数 '{param_key}' 生成文件，保存至 '{output_dir}'...")

        for archetype in archetypes:
            # --- 检查适用性 ---
            is_multilayer = archetype['net_key'] not in ['1ra', '1sw', '1sf']
            is_spatiotemporal = archetype['prefix'] == 'st'
            
            if param_info.get('requires_multilayer') and not is_multilayer:
                continue
            if param_info.get('requires_spatiotemporal') and not is_spatiotemporal:
                continue

            for value in param_values:
                net_key = archetype['net_key']
                net_info = network_models[net_key]
                geo_key = archetype.get('geo_key', '')
                
                # --- 构建文件名 ---
                filename_parts = [archetype['prefix'], net_key, param_key, str(value)]
                if geo_key:
                    filename_parts.append(geo_key)
                filename = ".".join(filename_parts).replace("_.", ".") + ".yaml"

                # --- 填充和修改模板 ---
                base_content = archetype['template']
                
                # 1. 填充网络和地理信息
                # Note: Added param_key and param_value for better comments in the generated file
                formatted_content = base_content.format(
                    output_dir=output_dir,
                    filename=filename,
                    net_description=net_info['description'],
                    layers_config=net_info['layers'],
                    coupling_enabled=str(is_multilayer).lower(),
                    geo_description=geo_locations.get(geo_key, {}).get('description', ''),
                    geo_config_block=geo_locations.get(geo_key, {}).get('config', ''),
                    param_key=param_key,
                    param_value=value
                )
                
                # 2. 修改敏感性参数
                final_content = set_param_value(formatted_content, param_path, value)

                # --- 写入文件 ---
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                total_files += 1

        print(f"  参数 '{param_key}' 的文件已生成。")

    print("\n--- [Sensitivity Analysis] 生成完成！ ---")
    print(f"==================================================")
    print(f"总计: {total_files} 个敏感性分析文件已成功生成。")


if __name__ == "__main__":
    generate_sensitivity_files()