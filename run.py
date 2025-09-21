import os
import sys
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

# --- 1. 决定计算后端 (CPU/GPU) ---
from utils.gpu_utils import xp, is_gpu_available

if is_gpu_available():
    print("✅ CUDA GPU is available. Using CuPy for acceleration.")
else:
    # 这里可以进一步判断 cupy 是否安装来给出更详细的信息
    try:
        import cupy
        print("⚠️ CuPy is installed, but no available CUDA GPU was found. Using NumPy.")
    except ImportError:
        print("ℹ️ CuPy is not installed. Using NumPy.")

# --- 2. 导入模型和工具 ---
# 确保根目录在路径中，以便进行绝对导入
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from simulation.basic_simulation import BasicSimulation
from simulation.dw3d_simulation import DW3DSimulation
from utils import metrics
from utils.pic import generate_all_plots
from utils.analysis import create_analysis_report 

# --- 辅助函数 ---
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
    if CUPY_AVAILABLE:
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
    主运行函数，处理命令行参数，加载配置文件，并启动指定的仿真。
    """
    # --- 3. 保留完整的命令行参数解析 ---
    parser = argparse.ArgumentParser(description="运行社交平台仿真模型。")
    # ... (这部分与您之前的 run.py 完全相同) ...
    parser.add_argument("--sim_type", type=str, required=True, choices=["basic", "dw3d"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metrics", nargs='+', required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all_layers", action='store_true')

    args = parser.parse_args()

    print(f"--- 准备运行仿真: {args.sim_type} ---")
    print(f"配置文件: {args.config}")
    print(f"计算指标: {', '.join(args.metrics)}")

    # --- 4. 加载配置文件并准备参数 ---
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在。")
        return
        
    # --- 核心修改：明确指定 encoding='utf-8' ---
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 从加载的字典中提取网络和仿真参数
    # 这种结构要求YAML文件包含 'network' 和 'simulation_params' 两个顶级键
    if 'network' not in config_dict or 'simulation_params' not in config_dict:
        print(f"错误: 配置文件 '{args.config}' 缺少 'network' 或 'simulation_params' 顶级块。")
        return
        
    network_params = config_dict['network']
    sim_params = config_dict['simulation_params']

    # --- 5. 根据命令行参数选择仿真类 ---
    SimulationClass = None
    if args.sim_type == "basic":
        SimulationClass = BasicSimulation
    elif args.sim_type == "dw3d":
        SimulationClass = DW3DSimulation
    
    # --- 6. 正确地实例化仿真类 (核心修正) ---
    # 将解析后的字典注入，而不是文件路径
    simulation = SimulationClass(
        xp=xp,
        network_params=network_params,
        sim_params=sim_params
    )
    
    simulation_history = simulation.run_simulation()

    print("\n--- 仿真完成，开始计算指定指标 ---")

    # --- 7. 指标计算和结果保存 (逻辑与之前版本兼容) ---
    # ... (这部分代码是正确的，它使用 args.metrics 和 args.all_layers 来决定行为) ...
    num_layers = simulation.num_layers
    layers_to_analyze = range(num_layers) if args.all_layers else [args.layer]
    results_data = {
        "simulation_type": args.sim_type, "config_path": args.config, "timestamp": datetime.now().isoformat(),
        "config_used": config_dict, "metrics_calculated": args.metrics,
        "layer_analyzed": "all" if args.all_layers else args.layer,
        "results": {
            "metrics_over_time": {"iteration": [h['iteration'] for h in simulation_history]},
            "final_metrics": {}, "raw_history": [h for h in simulation_history]
        }
    }
    metric_functions = {name: getattr(metrics, name) for name in args.metrics}
    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"; results_data["results"]["metrics_over_time"][layer_key] = {name: [] for name in metric_functions}; results_data["results"]["final_metrics"][layer_key] = {}
    for i, history_step in enumerate(tqdm(simulation_history, desc="Calculating Metrics")):
        opinions_snapshot = history_step['opinions_snapshot']
        for layer_idx in layers_to_analyze:
            layer_key = f"layer_{layer_idx}"; current_opinions = opinions_snapshot[:, layer_idx]; current_graph = simulation.graphs[layer_idx]
            for name, func in metric_functions.items():
                value = np.nan
                try:
                    if 'graph' in func.__code__.co_varnames and 'opinions' in func.__code__.co_varnames: value = func(current_graph, current_opinions)
                    elif 'graph' in func.__code__.co_varnames: value = func(current_graph)
                    else: value = func(current_opinions)
                except Exception as e: print(f"计算指标 '{name}' (层 {layer_idx}) 时出错: {e}")
                results_data["results"]["metrics_over_time"][layer_key][name].append(value)
    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"
        for name in metric_functions: results_data["results"]["final_metrics"][layer_key][name] = results_data["results"]["metrics_over_time"][layer_key][name][-1]
    
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if args.output: output_filename = f"{args.output}.json"
    else:
        # 否则，根据YAML配置文件名和当前时间戳生成
        # 1. 从配置文件路径中提取不带扩展名的基本文件名
        config_basename = os.path.splitext(os.path.basename(args.config))[0]
        # 2. 获取当前时间戳字符串
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 3. 组合成最终的基本文件名，格式为: yaml文件名_时间
        output_filename = f"{config_basename}_{timestamp}.json"
    output_path = os.path.join(results_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(results_data, f, indent=4, default=convert_numpy_to_json)
    print(f"\n结果已成功保存到: {output_path}")
    
    if args.output: 
        output_basename = args.output
    else: 
        output_basename = f"{config_basename}_{timestamp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    generate_all_plots(results_data, output_basename)
    create_analysis_report(results_data, output_basename)

if __name__ == "__main__":
    main()