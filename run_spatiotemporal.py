import os
import sys
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

# --- 1. GPU/CPU 后端选择 ---
from utils.gpu_utils import xp, is_gpu_available

try:
    import cupy
except ImportError:
    # 如果 cupy 未安装，创建一个假的 cupy 对象，这样 isinstance 检查不会失败
    class CupyModuleMock:
        class ndarray: pass
        class generic: pass
    cupy = CupyModuleMock()

if is_gpu_available():
    print("✅ CUDA GPU is available. Using CuPy for acceleration.")
else:
    try:
        import cupy
        print("⚠️ CuPy is installed, but no GPU found. Using NumPy.")
    except ImportError:
        print("ℹ️ CuPy is not installed. Using NumPy.")

# --- 2. 导入【时空专属】模型和仿真模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from simulation.spatiotemporal_simulation import SpatioTemporalSimulation
from utils import metrics
from utils.pic import generate_all_plots
from utils.analysis import create_analysis_report 

# --- 辅助函数 (与 run.py 完全相同) ---
def convert_numpy_to_json(obj):
    """
    一个可以同时处理 NumPy 和 CuPy 对象的 JSON 序列化转换器。
    """
    # 首先处理 NumPy 类型
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # 接着处理 CuPy 类型 (仅当 CuPy 可用时)
    if is_gpu_available():
        if isinstance(obj, cupy.ndarray):
            # 关键步骤：先 .get() 转为 numpy 数组，再 .tolist() 转为列表
            return obj.get().tolist()
        # CuPy 的标量类型通常会自动转为 NumPy 或 Python 类型，但以防万一
        if isinstance(obj, cupy.generic):
            return obj.item()

    # 如果以上都不是，则抛出原始错误
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    """
    主运行函数，专门用于启动【时空耦合】仿真实验，并提供与 run.py 一致的命令行接口。
    """
    # --- 3. 设置与 run.py 一致的命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="运行一个由配置文件驱动的【时空耦合】社交平台仿真实验。"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="指向时空实验配置 YAML 文件的路径。"
    )
    parser.add_argument(
        "--metrics", 
        nargs='+', 
        required=True,
        help="要计算的指标列表。"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="自定义输出 JSON 文件的名称 (不含扩展名)。"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="对于多层网络，指定计算指标的层索引 (默认为 0)。"
    )
    parser.add_argument(
        "--all_layers",
        action='store_true',
        help="如果设置此项，将计算并保存所有网络层的指标，--layer 参数将被忽略。"
    )
    args = parser.parse_args()

    # --- 4. 加载和解析配置文件 ---
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在。")
        return
        
    # --- 加载配置文件 ---
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    # 【修改】现在我们直接使用 network_params_config，而不是 network_graphs
    network_params_config = config_dict['network']
    sim_params = config_dict['simulation_params']

    # =========================================================================
    # --- 【核心修改区域】 ---
    # =========================================================================

    # --- 1. 【删除】旧的模型构建阶段 ---
    # 旧代码中手动创建 SpatioTemporalModel 并生成网络的整个部分
    # 已被完全移除。这个职责已经转移到 SpatioTemporalSimulation 内部。
    
    # --- 2. 【修改】仿真实例化阶段 ---
    print("\n--- 1. 开始实例化【多层时空】仿真器 ---")
    # 我们现在直接将从配置文件中读取的 network_params_config 字典传递给仿真器。
    # 仿真器的 __init__ 方法会负责调用其内部的 _initialize_network 方法，
    # 从而使用 SpatioTemporalModel 来构建正确的、带空间属性的多层网络。
    simulation = SpatioTemporalSimulation(
        xp=xp,
        network_params=network_params_config, # <-- 关键修改：传递网络配置字典
        sim_params=sim_params
    )
    # --- 7. 运行和结果处理阶段 (与 run.py 完全一致) ---
    print("\n--- 3. 开始运行仿真 ---")
    simulation_history = simulation.run_simulation()
    
    print("\n--- 仿真完成，开始计算指定指标 ---")
    
    # 指标计算和结果保存逻辑完全复用 run.py 的
    num_layers = simulation.num_layers
    layers_to_analyze = range(num_layers) if args.all_layers else [args.layer]

    results_data = {
        "simulation_type": "spatiotemporal",
        "config_path": args.config,
        "config_used": config_dict,
        "timestamp": datetime.now().isoformat(),
        "metrics_calculated": args.metrics,
        "layer_analyzed": "all" if args.all_layers else args.layer,
        "results": {
            # --- 新增: 记录生成的随机事件 ---
            # 从simulation对象中直接获取已生成的事件列表
            "generated_events": simulation.events,
            "metrics_over_time": {"iteration": [h['iteration'] for h in simulation_history]},
            "final_metrics": {},
            "node_positions": [data['pos'] for _, data in simulation.graphs[0].nodes(data=True)],
            "raw_history": [h for h in simulation_history]
        }
    }
    
    metric_functions = {name: getattr(metrics, name) for name in args.metrics}
    
    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"
        results_data["results"]["metrics_over_time"][layer_key] = {name: [] for name in metric_functions}
        results_data["results"]["final_metrics"][layer_key] = {}
        
    for i, history_step in enumerate(tqdm(simulation_history, desc="Calculating Metrics")):
        opinions_snapshot = history_step['opinions_snapshot']
        for layer_idx in layers_to_analyze:
            layer_key = f"layer_{layer_idx}"
            current_opinions = opinions_snapshot[:, layer_idx]
            current_graph = simulation.graphs[layer_idx]
            for name, func in metric_functions.items():
                value = np.nan
                try:
                    if 'graph' in func.__code__.co_varnames and 'opinions' in func.__code__.co_varnames: value = func(current_graph, current_opinions)
                    elif 'graph' in func.__code__.co_varnames: value = func(current_graph)
                    else: value = func(current_opinions)
                except Exception as e:
                    print(f"计算指标 '{name}' (层 {layer_idx}) 时出错: {e}")
                results_data["results"]["metrics_over_time"][layer_key][name].append(value)

    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"
        for name in args.metrics:
            results_data["results"]["final_metrics"][layer_key][name] = results_data["results"]["metrics_over_time"][layer_key][name][-1]

    # --- 8. 保存结果 (与 run.py 完全一致) ---
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if args.output:
        output_filename = f"{args.output}.json"
    else:
        # 否则，根据YAML配置文件名和当前时间戳生成
        # 1. 从配置文件路径中提取不带扩展名的基本文件名
        config_basename = os.path.splitext(os.path.basename(args.config))[0]
        # 2. 获取当前时间戳字符串
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 3. 组合成最终的基本文件名，格式为: yaml文件名_时间
        output_filename = f"{config_basename}_{timestamp}.json"
    output_path = os.path.join(results_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, default=convert_numpy_to_json)
    print(f"\n时空实验结果已成功保存到: {output_path}")

    if args.output: 
        output_basename = args.output
    else: 
        output_basename = f"{args.sim_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    generate_all_plots(results_data, output_basename)
    create_analysis_report(results_data, output_basename)

if __name__ == "__main__":
    main()