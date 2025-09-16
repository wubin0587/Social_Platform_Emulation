# 不同社交平台信息茧房效应的仿真分析

用于多层社交网络上舆论演化与回音室效应的模拟与批量复现。

---

## 一句话说明
基于配置文件驱动的仿真框架，支持 Basic / DW3D / SpatioTemporal 三类仿真模式；支持 GPU 加速（可选），并提供批量运行、指标计算与结果可视化管道。

---

## 特性（Engineering highlights）
- 配置驱动（YAML）——轻松复现实验与修改参数。
- 多种仿真模块：basic、dw3d、spatiotemporal。
- 批量实验脚本 `setup.py`，并行/队列执行配置列表。
- 支持 GPU（通过 CuPy）或回退到 NumPy。
- 自动计算一系列网络与舆论指标并生成可视化图与 JSON 输出。

---

## 项目结构（简要）
```

Social\_Platform\_Emulation/
├─ config/                  # 仿真配置（YAML）
│  ├─ basic\_*.yaml
│  ├─ dw3d\_*.yaml
│  └─ spatiotemporal/*.yaml
├─ simulation/              # 各类仿真实现
│  ├─ basic\_simulation.py
│  ├─ dw3d\_simulation.py
│  └─ spatiotemporal\_simulation.py
├─ utils/                   # 指标、绘图、GPU 判定等工具
├─ run.py                   # 运行 basic / dw3d 仿真（参数化）
├─ run\_spatiotemporal.py    # 运行时空耦合仿真
├─ setup.py                 # 批量运行 config 列表并并行执行
├─ test.py                  # 基础仿真单元测试 / 快速验证
├─ test\_spatiotemporal.py   # 时空仿真测试
├─ requirements.txt
└─ README.md

````

---

## 快速开始（Quick Start）

1. 克隆仓库并进入目录：
```bash
git clone <repo-url>
cd Social_Platform_Emulation
````

2. 创建 Python 环境并安装依赖：

```bash
pip install -r requirements.txt
```

3. （可选）若有 NVIDIA GPU 并希望加速，按 CUDA 版本安装 CuPy，例如：

```bash
# 根据你的 CUDA 版本选择合适的 cupy 包名，例如 cupy-cuda12x
pip install cupy-cuda12x
```

> requirements.txt 中已列出 `cupy`（可按需安装具体 cuda 版本的包）。

---

## 使用方法（Examples）

### 运行 Basic / DW3D 仿真

`run.py` 的主要参数：

* `--sim_type`：`basic` 或 `dw3d`（必选）
* `--config`：YAML 配置文件路径（必选）
* `--metrics`：要计算的指标名称列表（必选，空格分隔）
* `--output`：输出文件名前缀（可选）
* `--layer`：单层指标索引（默认 0）
* `--all_layers`：若指定则计算并保存所有层指标

示例：

```bash
python run.py --sim_type basic \
  --config config/basic_1sw.yaml \
  --metrics opinion_variance homophilic_bimodality_coefficient \
  --output result_basic_1sw \
  --all_layers
```

### 运行 Spatio-Temporal 仿真

`run_spatiotemporal.py` 参数和用法与 `run.py` 类似（`--config`、`--metrics`、`--output`、`--layer`、`--all_layers` 等），示例：

```bash
python run_spatiotemporal.py \
  --config config/spatiotemporal/st_1ra.grid.yaml \
  --metrics homophilic_bimodality_coefficient opinion_variance \
  --output result_st_1ra
```

### 快速测试

```bash
# 验证基础仿真能跑通
python test.py

# 验证时空仿真能跑通
python test_spatiotemporal.py
```

### 批量复现实验（setup.py）

`setup.py` 内部维护了一份 `CONFIG_FILES` 列表并列举了常用指标（`METRICS_TO_CALCULATE`）。运行：

```bash
python setup.py
```

脚本会并行（可配置进程数）执行 `CONFIG_FILES` 中的每个配置，并输出批量执行结果统计。

---

## 配置文件（YAML）说明（常用字段）

每个 YAML 文件位于 `config/` 或 `config/spatiotemporal/`，典型结构如下（示例片段来自 `config/basic_1sw.yaml`）：

```yaml
# basic_1sw.yaml: 基础仿真，包含一个5000节点的小世界网络层
network:
  num_nodes: 1000
  layers:
    - {type: small_world, params: {k: 10, beta: 0.1}}

simulation_params:
  max_iterations: 400
  record_interval: 10

  convergence_params:
    enabled: true        # 开启功能
    threshold: 0.0001    # 总观点变化阈值
    patience: 100        # 连续100次低于阈值则停止
  
  initial_state: 
    opinion_range: [0.0, 1.0]
    epsilon_base: 0.15
    
  dw_params: 
    mu: 0.2
    
  coupling_params: 
    enabled: false
    lambda: 0.01
    c_kl_global: 0.1
```

时空仿真额外常见字段（示例来自 `config/spatiotemporal/st_1ra.grid.yaml`）：

```yaml
position_distribution:
  type: 'grid'
  grid_params: {rows: 32, cols: 32, jitter: 0.02}

spatiotemporal_params:
  poisson_rate: 0.1
  spatial_range: [[0.0, 1.0], [0.0, 1.0]]
  alpha: 0.8
  beta: 0.05
  interaction_prob: {base: 0.7, gain: 0.3}
```

**如何自定义**：通常需要设置 `network`（节点数与层定义）、`simulation_params`（迭代次数、记录频率、初始意见分布）、以及模型特有参数（如 `dw_params`、`coupling_params`、`spatiotemporal_params`）。修改后使用 `run.py` / `run_spatiotemporal.py` 指向该配置即可。

---

## 输出（结果与位置）

每次仿真会在 `results/`（或脚本中指定的结果目录）生成：

* `*.json`：完整结构化结果（时间序列、最终指标、配置元信息等）
* 一组图片（png）/图表：指标随时间变化、分布图等
* 自动生成的分析小结（如果 `utils.analysis.create_analysis_report` 可用）

文件命名通常含有配置名称与时间戳，示例： `basic_1sw_20250916_153011.json`。

---

## 常见问题 & 故障排查

1. **无法使用 GPU / 未检测到 CuPy**

   * 检查是否安装了匹配你 CUDA 版本的 CuPy（例如 `cupy-cuda12x`）。如果未安装，脚本会回退到 NumPy。
2. **依赖缺失导致报错**

   * 确认 `pip install -r requirements.txt` 成功执行；必要时手动安装缺失包（如 `networkx`、`pyyaml`、`tqdm` 等）。
3. **配置文件语法错误**

   * YAML 对缩进敏感，务必使用空格（不要用 tab）。可以用 `python -c "import yaml,sys; yaml.safe_load(open('config/xxx.yaml'))"` 快速检查。
4. **批量运行中某些任务失败但其他任务成功**

   * 查看 `setup.py` 打印的失败日志（脚本会在失败时输出详细信息）。建议单独用 `run.py` 跑该配置以定位错误。

---

## 开发者说明（快速上手）

* 仿真核心位于 `simulation/`；若要添加指标，把指标函数放入 `utils/metrics.py` 并在脚本中引用。
* `utils/gpu_utils.py` 管理 `xp`（NumPy 或 CuPy）的自动切换，确保向量运算兼容。
* 若要调整并发数、输出目录或日志等级，可在 `setup.py` 中定制或添加 CLI 参数。

---

## 依赖

见 `requirements.txt`。主要依赖包括（示例）：

* numpy, scipy, networkx, matplotlib, seaborn, pyyaml, tqdm
* 可选：cupy（GPU 加速）

---

## 许可证 & 联系

* 许可证：请在项目根目录补充 LICENSE 文件（当前仓库未显式声明）。
* 联系：在仓库 Issue 或直接通过作者邮箱（若有）联系。

---

## 致谢

该仓库为基于配置文件的仿真工程化实现，适合复现大量参数化实验与批量分析。