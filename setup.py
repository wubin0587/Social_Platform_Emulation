import sys
import os
import subprocess
import glob # <--- 添加 glob 模块用于文件扫描
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. 定义所有需要计算的指标 (保持不变) ---
METRICS_TO_CALCULATE = [
    'homophilic_bimodality_coefficient',
    'opinion_variance',
    'number_of_opinion_clusters'
]

def run_config_generation():
    """
    执行 config/create_all_config.py 脚本来生成所有配置文件。
    """
    script_path = os.path.join("config", "create_all_config.py")
    if not os.path.exists(script_path):
        print(f"❌ 错误: 配置文件生成脚本 '{script_path}' 不存在！")
        return False

    print(f"--- 1. 正在执行脚本以生成所有配置文件: {script_path} ---")
    try:
        child_env = os.environ.copy()
        # 2. 强制子进程的Python I/O使用UTF-8编码
        child_env['PYTHONIOENCODING'] = 'utf-8'
        # 使用 sys.executable 确保使用当前 Python 解释器
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=child_env  # <--- 将设置好的环境变量应用到子进程
        )
        print("✅ 配置文件生成成功。")
        # print(result.stdout) # 如果需要，可以取消注释以查看生成脚本的输出
        return True
    except subprocess.CalledProcessError as e:
        print("\n❌ 错误: 配置文件生成脚本执行失败。")
        print(f"返回码: {e.returncode}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 Python 解释器 '{sys.executable}'。请检查您的环境。")
        return False

def discover_config_files(base_dir="config"):
    """
    动态扫描指定目录下所有子文件夹中的 .yaml 文件。
    """
    print(f"--- 2. 正在从 '{base_dir}' 目录中扫描配置文件... ---")
    # 使用 glob 搜索 config 目录下所有子文件夹中的 .yaml 文件
    # recursive=True 允许使用 ** 来匹配任意层级的子目录
    search_pattern = os.path.join(base_dir, "**", "*.yaml")
    config_files = glob.glob(search_pattern, recursive=True)

    # 对找到的文件进行排序，确保执行顺序一致
    config_files.sort()

    if not config_files:
        print(f"⚠️ 警告: 在 '{base_dir}' 目录中没有找到任何 .yaml 配置文件。")
    else:
        print(f"✅ 扫描完成，共找到 {len(config_files)} 个配置文件。")

    return config_files


def check_and_install_dependencies(requirements_file='requirements.txt'):
    """
    检查 requirements.txt 文件是否存在，并使用 pip 安装其中列出的所有库。
    """
    if not os.path.exists(requirements_file):
        print(f"错误: 依赖文件 '{requirements_file}' 不存在！")
        print("请确保在项目根目录下创建该文件，并列出所有必要的库。")
        sys.exit(1)

    print(f"--- 3. 正在检查并安装 '{requirements_file}' 中的依赖项 ---")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
            check=True, text=True, encoding='utf-8', capture_output=True
        )
        print("✅ 所有依赖项均已安装或为最新版本。")
        return True
    except subprocess.CalledProcessError as e:
        print("\n错误: 安装依赖项失败。")
        print(f"Pip 输出:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"错误: 找不到 Python 解释器 '{sys.executable}' 或 'pip'。请检查您的环境。")
        return False

def build_command_for_config(config_file):
    """根据配置文件路径构建单个命令列表，但不执行。"""
    python_executable = sys.executable
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 检查文件是否存在
    if not os.path.exists(config_file):
        return (config_file, None, f"配置文件不存在: {config_file}")

    command = [python_executable]

    # 根据文件路径判断执行哪个脚本和 sim_type
    normalized_path = os.path.normpath(config_file)
    path_parts = normalized_path.split(os.sep)

    if 'spatiotemporal' in path_parts:
        run_script = os.path.join(project_root, "run_spatiotemporal.py")
        command.extend([run_script, '--config', config_file])

    elif len(path_parts) > 1 and path_parts[0] == 'config':
        folder_name = path_parts[1]  # e.g., 'basic', 'dw3d_exh', 'dw3d_time'

        # --- 这是关键的修改 ---
        # 如果文件夹名以 'dw3d' 开头，则 sim_type 固定为 'dw3d'
        if folder_name.startswith('dw3d'):
            sim_type = 'dw3d'
        else:
            # 否则，使用文件夹名作为 sim_type (例如 'basic')
            sim_type = folder_name
        # --- 修改结束 ---

        run_script = os.path.join(project_root, "run.py")
        command.extend([run_script, '--sim_type', sim_type, '--config', config_file])
    else:
        return (config_file, None, f"无法从路径识别sim_type: {config_file}")

    command.extend(['--metrics', *METRICS_TO_CALCULATE])
    command.append('--all_layers')

    return (config_file, command, None)

def run_single_simulation_task(config_file, command):
    """执行单个仿真任务并捕获其输出。"""
    if command is None:
        return config_file, False, "命令构建失败\n"

    try:
        child_env = os.environ.copy()
        child_env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            command,
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            check=False, 
            env=child_env
        )

        output = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n"

        if result.returncode == 0:
            return config_file, True, output
        else:
            return config_file, False, f"子进程返回错误码 {result.returncode}\n{output}"

    except Exception as e:
        return config_file, False, f"执行时发生异常: {e}\n"

def main():
    # 第 1 步: 执行脚本生成所有配置文件
    if not run_config_generation():
        print("由于配置文件生成失败，主程序已终止。")
        sys.exit(1)

    # 第 2 步: 动态扫描生成的配置文件
    config_files_to_run = discover_config_files()
    if not config_files_to_run:
        print("未找到任何要执行的仿真任务，程序退出。")
        sys.exit(0)

    # 第 3 步：检查并安装所有依赖项
    if not check_and_install_dependencies('requirements.txt'):
        print("由于依赖项安装失败，主程序已终止。")
        sys.exit(1)

    print("\n--- 4. 并行批量仿真任务已启动 ---")

    # 设置并行工作进程数
    MAX_WORKERS = 1 # 您可以根据您的GPU核心数调整此值
    print(f"将使用最多 {MAX_WORKERS} 个并行进程。")

    # 1. 构建所有命令
    tasks = [build_command_for_config(cf) for cf in config_files_to_run]

    successful_runs = 0
    failed_runs = 0

    # 2. 使用 ProcessPoolExecutor 并行执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_config = {
            executor.submit(run_single_simulation_task, config, cmd): config
            for config, cmd, err in tasks if cmd
        }

        # 处理构建失败的任务
        for config, cmd, err in tasks:
            if err:
                print(f"❌ 任务预处理失败: {err}")
                failed_runs += 1

        # 使用 tqdm 显示进度
        pbar = tqdm(as_completed(future_to_config), total=len(future_to_config), desc="总仿真进度")
        for future in pbar:
            config_file, success, output = future.result()
            relative_path = os.path.relpath(config_file) # 获取相对路径，使输出更简洁
            pbar.set_postfix(file=f"...{relative_path[-40:]}") # 在进度条后显示当前处理的文件

            if success:
                successful_runs += 1
                # 成功时默认不打印日志，保持终端清洁
            else:
                failed_runs += 1
                print("\n" + "="*80)
                print(f"❌ 任务失败: {relative_path}")
                print(output)
                print("="*80)

    print("\n" + "="*80)
    print("🎉 所有仿真任务已完成 🎉")
    print(f"总计: {len(config_files_to_run)} 个任务")
    print(f"✅ 成功: {successful_runs} 个")
    print(f"❌ 失败: {failed_runs} 个")
    print("="*80)


if __name__ == "__main__":
    main()