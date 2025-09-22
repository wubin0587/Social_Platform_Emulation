import sys
import os
import subprocess
from datetime import datetime

def run_spatiotemporal_test_via_injection():
    """
    通过命令行注入的方式，调用 run_spatiotemporal.py 来执行一个
    预设的时空耦合测试场景。
    """
    print("--- 准备通过命令行注入方式运行【时空】测试 ---")
    
    # --- 1. 定义测试参数 ---
    
    # 确定 Python 解释器和要运行的脚本
    python_executable = sys.executable
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_script_path = os.path.join(project_root, "run_spatiotemporal.py")
    
    # 指定要使用的配置文件 (相对于项目根目录)
    config_file = "config/test_spatiotemporal.yaml" 
    
    # 为输出文件生成一个唯一的名字
    output_filename = f"test_spatiotemporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # --- 2. 构建完整的命令行参数列表 ---
    # 这个列表将模拟您在终端中手动输入的内容
    command = [
        python_executable,
        run_script_path,
        '--config', config_file,
        '--metrics', 'opinion_variance', 'homophilic_bimodality_coefficient', 'number_of_opinion_clusters',
        '--all_layers', # 对于单层网络，这仍然是安全的，会分析第0层
        '--output', output_filename
    ]
    
    # --- 3. 打印将要执行的命令，方便调试 ---
    print("\n将要执行的命令:")
    # 使用 subprocess.list2cmdline 来美观地显示命令
    print(subprocess.list2cmdline(command))
    print("\n--- 开始执行子进程 ---")
    
    # --- 4. 执行命令 ---
    try:
        result = subprocess.run(command, check=True, text=True, encoding='utf-8')
        print("\n--- 子进程执行成功 ---")
        # print("STDOUT:\n", result.stdout) # 如果需要，可以取消注释来打印 run.py 的输出
    except FileNotFoundError:
        print(f"错误: 找不到脚本或解释器 '{python_executable}' 或 '{run_script_path}'。")
    except subprocess.CalledProcessError as e:
        print(f"\n--- 子进程执行失败 ---")
        print(f"run.py 脚本返回了错误，退出码: {e.returncode}")
        # print("STDERR:\n", e.stderr) # 打印错误输出
        # print("STDOUT:\n", e.stdout) # 打印标准输出

    # 检查配置文件是否存在
    full_config_path = os.path.join(project_root, config_file)
    if not os.path.exists(full_config_path):
        print(f"\n警告: 配置文件 '{full_config_path}' 不存在。")
        print("请确保您已经在 `config/` 目录下创建了 `spatiotemporal_test.yaml` 文件。")


if __name__ == "__main__":
    run_spatiotemporal_test_via_injection()