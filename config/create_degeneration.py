import os

# ==============================================================================
# 生成“消融/退化研究” (Ablation/Degeneration Study) 的配置文件
# 目的: 生成极少量、精确修改的配置文件，用于验证模型各模块的正确性。
# ==============================================================================

# --- 定义基础模板 (仅包含本脚本需要的) ---

basic_template = """# {output_dir}/{filename}
# [DEGENERATION TEST] 基础DW模型 (无圈层耦合)
# {net_description}
network:
  num_nodes: 1000
  layers:
{layers_config}
simulation_params:
  max_iterations: 800
  record_interval: 10
  convergence_params: {{enabled: true, threshold: 0.001, patience: 20}}
  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}
  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.0  # <-- MODIFICATION: Coupling is turned off
    c_kl_global: 0.002
"""

dw3d_exh_template = """# {output_dir}/{filename}
# [DEGENERATION TEST] DW3D模型 (扩展功能已禁用)
# {net_description}
network:
  num_nodes: 1000
  layers:
{layers_config}
simulation_params:
  max_iterations: 800
  record_interval: 10
  convergence_params: {{enabled: true, threshold: 0.001, patience: 20}}
  initial_state: {{opinion_range: [0.0, 1.0], epsilon_base: 0.15}}
  dw_params: {{mu: 0.2}}
  coupling_params:
    enabled: {coupling_enabled}
    lambda: 0.002
    c_kl_global: 0.002
  dw3d_extensions:
    interaction_mode: 'random'
    dynamic_epsilon:
      enabled: false  # <-- MODIFICATION: Extension disabled
      extremity_inhibition: {{enabled: true, alpha: 0.5, opinion_center: 0.5}}
      time_evolution: {{enabled: false, beta: 0.01}}
    dynamic_network:
      enabled: false  # <-- MODIFICATION: Extension disabled
      disconnect_threshold_factor: 1.8
      reconnect_probability: 0.01
      reconnect_opinion_threshold: 0.15
"""

# --- 定义代表性网络配置 ---
representative_networks = {
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


def generate_ablation_files():
    """主函数，生成所有退化测试文件。"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "ablation_study")
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- 开始生成 [Ablation/Degeneration Study] 配置文件 ---")
    print(f"--- 所有文件将保存到 '{output_dir}' 文件夹中 ---")
    
    # --- 实验 1: 测试圈层耦合逻辑 ---
    net_key = "2sw_2sf"
    net_info = representative_networks[net_key]
    filename = f"degen_basic_no_coupling_{net_key}.yaml"
    
    content_1 = basic_template.format(
        output_dir=output_dir,
        filename=filename,
        net_description=net_info["description"],
        layers_config=net_info["layers"],
        coupling_enabled='true' # 保持为true，但lambda为0
    )
    
    file_path_1 = os.path.join(output_dir, filename)
    with open(file_path_1, 'w', encoding='utf-8') as f:
        f.write(content_1)
    print(f"  [OK] 已生成: {filename} (用于测试圈层耦合)")

    # --- 实验 2: 测试DW3D扩展逻辑 ---
    net_key = "4sw"
    net_info = representative_networks[net_key]
    filename = f"degen_dw3d_no_extensions_{net_key}.yaml"
    
    content_2 = dw3d_exh_template.format(
        output_dir=output_dir,
        filename=filename,
        net_description=net_info["description"],
        layers_config=net_info["layers"],
        coupling_enabled='true'
    )
    
    file_path_2 = os.path.join(output_dir, filename)
    with open(file_path_2, 'w', encoding='utf-8') as f:
        f.write(content_2)
    print(f"  [OK] 已生成: {filename} (用于测试DW3D扩展)")

    print("\n[重要提示] 测试“时空事件逻辑”无需新文件。")
    print("请使用 'spatiotemporal_control_no_events' 文件夹中的任一配置进行验证。")
    
    print("\n--- [Ablation/Degeneration Study] 生成完成！ ---")


if __name__ == "__main__":
    generate_ablation_files()