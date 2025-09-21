import sys
import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed # <--- 添加这一行

# --- 1. 定义所有仿真配置文件 ---
# 将所有config文件的相对路径存储在一个列表中
CONFIG_FILES = [
    # Basic
    r"config/basic_1ra.yaml",
    r"config/basic_1sf.yaml",
    r"config/basic_1sw.yaml",
    r"config/basic_1sw_9sf.yaml",
    r"config/basic_3sw.yaml",
    r"config/basic_5sw.yaml",
    r"config/basic_9sw_1sf.yaml",
    r"config/basic_10ra.yaml",
    r"config/basic_10sf.yaml",
    r"config/basic_10sw.yaml",
    # DW3D
    r"config/dw3d_1ra.yaml",
    r"config/dw3d_1sf.yaml",
    r"config/dw3d_1sw.yaml",
    r"config/dw3d_1sw_9sf.yaml",
    r"config/dw3d_5sw_5sf.yaml",
    r"config/dw3d_9sw_1sf.yaml",
    r"config/dw3d_10ra.yaml",
    r"config/dw3d_10sf.yaml",
    r"config/dw3d_10sw.yaml",
    # Spatiotemporal
    r"config/spatiotemporal/st_1ra.clustered.yaml",
    r"config/spatiotemporal/st_1ra.concentric_circles.yaml",
    r"config/spatiotemporal/st_1ra.gaussian.yaml",
    r"config/spatiotemporal/st_1ra.grid.yaml",
    r"config/spatiotemporal/st_1ra.linear.yaml",
    r"config/spatiotemporal/st_1ra.multi_clustered.yaml",
    r"config/spatiotemporal/st_1ra.uniform.yaml",
    r"config/spatiotemporal/st_1ra.wedge.yaml",
    r"config/spatiotemporal/st_1sf.clustered.yaml",
    r"config/spatiotemporal/st_1sf.concentric_circles.yaml",
    r"config/spatiotemporal/st_1sf.gaussian.yaml",
    r"config/spatiotemporal/st_1sf.grid.yaml",
    r"config/spatiotemporal/st_1sf.linear.yaml",
    r"config/spatiotemporal/st_1sf.multi_clustered.yaml",
    r"config/spatiotemporal/st_1sf.uniform.yaml",
    r"config/spatiotemporal/st_1sf.wedge.yaml",
    r"config/spatiotemporal/st_1sw.clustered.yaml",
    r"config/spatiotemporal/st_1sw.concentric_circles.yaml",
    r"config/spatiotemporal/st_1sw.gaussian.yaml",
    r"config/spatiotemporal/st_1sw.grid.yaml",
    r"config/spatiotemporal/st_1sw.linear.yaml",
    r"config/spatiotemporal/st_1sw.multi_clustered.yaml",
    r"config/spatiotemporal/st_1sw.uniform.yaml",
    r"config/spatiotemporal/st_1sw.wedge.yaml",
    r"config/spatiotemporal/st_1sw_9sf.clustered.yaml",
    r"config/spatiotemporal/st_1sw_9sf.concentric_circles.yaml",
    r"config/spatiotemporal/st_1sw_9sf.gaussian.yaml",
    r"config/spatiotemporal/st_1sw_9sf.grid.yaml",
    r"config/spatiotemporal/st_1sw_9sf.linear.yaml",
    r"config/spatiotemporal/st_1sw_9sf.multi_clustered.yaml",
    r"config/spatiotemporal/st_1sw_9sf.uniform.yaml",
    r"config/spatiotemporal/st_1sw_9sf.wedge.yaml",
    r"config/spatiotemporal/st_5sw_5sf.clustered.yaml",
    r"config/spatiotemporal/st_5sw_5sf.concentric_circles.yaml",
    r"config/spatiotemporal/st_5sw_5sf.gaussian.yaml",
    r"config/spatiotemporal/st_5sw_5sf.grid.yaml",
    r"config/spatiotemporal/st_5sw_5sf.linear.yaml",
    r"config/spatiotemporal/st_5sw_5sf.multi_clustered.yaml",
    r"config/spatiotemporal/st_5sw_5sf.uniform.yaml",
    r"config/spatiotemporal/st_5sw_5sf.wedge.yaml",
    r"config/spatiotemporal/st_9sw_1sf.clustered.yaml",
    r"config/spatiotemporal/st_9sw_1sf.concentric_circles.yaml",
    r"config/spatiotemporal/st_9sw_1sf.gaussian.yaml",
    r"config/spatiotemporal/st_9sw_1sf.grid.yaml",
    r"config/spatiotemporal/st_9sw_1sf.linear.yaml",
    r"config/spatiotemporal/st_9sw_1sf.multi_clustered.yaml",
    r"config/spatiotemporal/st_9sw_1sf.uniform.yaml",
    r"config/spatiotemporal/st_9sw_1sf.wedge.yaml",
    r"config/spatiotemporal/st_10ra.clustered.yaml",
    r"config/spatiotemporal/st_10ra.concentric_circles.yaml",
    r"config/spatiotemporal/st_10ra.gaussian.yaml",
    r"config/spatiotemporal/st_10ra.grid.yaml",
    r"config/spatiotemporal/st_10ra.linear.yaml",
    r"config/spatiotemporal/st_10ra.multi_clustered.yaml",
    r"config/spatiotemporal/st_10ra.uniform.yaml",
    r"config/spatiotemporal/st_10ra.wedge.yaml",
    r"config/spatiotemporal/st_10sf.clustered.yaml",
    r"config/spatiotemporal/st_10sf.concentric_circles.yaml",
    r"config/spatiotemporal/st_10sf.gaussian.yaml",
    r"config/spatiotemporal/st_10sf.grid.yaml",
    r"config/spatiotemporal/st_10sf.linear.yaml",
    r"config/spatiotemporal/st_10sf.multi_clustered.yaml",
    r"config/spatiotemporal/st_10sf.uniform.yaml",
    r"config/spatiotemporal/st_10sf.wedge.yaml",
    r"config/spatiotemporal/st_10sw.clustered.yaml",
    r"config/spatiotemporal/st_10sw.concentric_circles.yaml",
    r"config/spatiotemporal/st_10sw.gaussian.yaml",
    r"config/spatiotemporal/st_10sw.grid.yaml",
    r"config/spatiotemporal/st_10sw.linear.yaml",
    r"config/spatiotemporal/st_10sw.multi_clustered.yaml",
    r"config/spatiotemporal/st_10sw.uniform.yaml",
    r"config/spatiotemporal/st_10sw.wedge.yaml",
]

# --- 2. 定义所有需要计算的指标 ---
METRICS_TO_CALCULATE = [
    'homophilic_bimodality_coefficient',
    'network_density',
    'average_clustering_coefficient',
    'average_shortest_path_length',
    'get_giant_component_size',
    'get_degree_distribution',
    'opinion_variance',
    'number_of_opinion_clusters'
]

def check_and_install_dependencies(requirements_file='requirements.txt'):
    """
    检查 requirements.txt 文件是否存在，并使用 pip 安装其中列出的所有库。
    """
    if not os.path.exists(requirements_file):
        print(f"错误: 依赖文件 '{requirements_file}' 不存在！")
        print("请确保在项目根目录下创建该文件，并列出所有必要的库。")
        sys.exit(1)

    print(f"--- 正在检查并安装 '{requirements_file}' 中的依赖项 ---")
    try:
        # 使用 sys.executable 来确保用的是当前Python环境关联的pip
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
            check=True,
            text=True,
            encoding='utf-8',
            capture_output=True 
        )
        print("--- 所有依赖项均已安装或为最新版本 ---")
        return True
    except subprocess.CalledProcessError as e:
        print("\n错误: 安装依赖项失败。")
        print("Pip 输出:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"错误: 找不到 Python 解释器 '{sys.executable}' 或 'pip'。请检查您的环境。")
        return False

def build_command_for_config(config_file):
    """根据配置文件构建单个命令列表，但不执行。"""
    python_executable = sys.executable
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 检查文件是否存在
    full_path = os.path.join(project_root, config_file)
    if not os.path.exists(full_path):
        # 返回一个错误标记，而不是命令
        return (config_file, None, f"配置文件不存在: {config_file}")

    command = [python_executable]
    
    if 'spatiotemporal' in config_file:
        run_script = os.path.join(project_root, "run_spatiotemporal.py")
        command.extend([run_script, '--config', config_file])
    else:
        run_script = os.path.join(project_root, "run.py")
        sim_type = 'basic' if 'basic' in config_file else 'dw3d' if 'dw3d' in config_file else None
        if not sim_type:
            return (config_file, None, f"无法从文件名识别sim_type: {config_file}")
        command.extend([run_script, '--sim_type', sim_type, '--config', config_file])
            
    command.extend(['--metrics', *METRICS_TO_CALCULATE])
    command.append('--all_layers')
    
    return (config_file, command, None)

def run_single_simulation_task(config_file, command):
    """执行单个仿真任务并捕获其输出。"""
    if command is None:
        return config_file, False, "命令构建失败\n"

    try:
        # --- 新增代码：为子进程设置环境变量 ---
        # 1. 复制当前进程的环境变量，这样子进程才能找到python等程序
        child_env = os.environ.copy()
        # 2. 强制子进程的Python I/O使用UTF-8编码
        child_env['PYTHONIOENCODING'] = 'utf-8'
        
        # 使用 subprocess.run 来等待完成并捕获输出
        # --- 修改之处：在下方增加了 env=child_env ---
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8', # 这个'encoding'参数是告诉父进程如何“解码”子进程的输出
            check=False,
            env=child_env     # 这个'env'参数是为子进程设置运行环境
        )
        
        output = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n"
        
        if result.returncode == 0:
            return config_file, True, output
        else:
            return config_file, False, f"子进程返回错误码 {result.returncode}\n{output}"

    except Exception as e:
        return config_file, False, f"执行时发生异常: {e}\n"

def main():

    # 第一步：检查并安装所有依赖项
    if not check_and_install_dependencies('requirements.txt'):
        print("由于依赖项安装失败，主程序已终止。")
        sys.exit(1)
    print("--- 并行批量仿真任务已启动 ---")
    
    # --- 关键: 设置并行工作进程数 ---
    # 逻辑核心数的一半是一个不错的起点，需要实验找到最佳值
    # 如果GPU内存是瓶颈，这个值可能需要设为2或3
    MAX_WORKERS = 1 
    print(f"将使用最多 {MAX_WORKERS} 个并行进程。")

    # 1. 首先构建所有命令
    tasks = [build_command_for_config(cf) for cf in CONFIG_FILES]
    
    successful_runs = 0
    failed_runs = 0
    
    # 2. 使用ProcessPoolExecutor并行执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_config = {
            executor.submit(run_single_simulation_task, config, cmd): config
            for config, cmd, err in tasks if cmd # 只提交命令构建成功的任务
        }
        
        # 处理构建失败的任务
        for config, cmd, err in tasks:
            if err:
                print(f"❌ 任务预处理失败: {err}")
                failed_runs += 1

        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_config), total=len(future_to_config), desc="总仿真进度"):
            config_file, success, output = future.result()
            
            if success:
                successful_runs += 1
                # 为了保持终端清洁，可以只在失败时打印详细日志
                # print(f"✅ 任务成功: {config_file}")
            else:
                failed_runs += 1
                print("\n" + "="*80)
                print(f"❌ 任务失败: {config_file}")
                print(output) # 打印失败的详细日志
                print("="*80)

    print("\n" + "="*80)
    print("🎉 所有仿真任务已完成 🎉")
    print(f"总计: {len(CONFIG_FILES)} 个任务")
    print(f"成功: {successful_runs} 个")
    print(f"失败: {failed_runs} 个")
    print("="*80)


if __name__ == "__main__":
    main()