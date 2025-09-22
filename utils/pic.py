import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import platform
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# --- 字体设置 ---
try:
    # 根据操作系统选择合适的字体
    if platform.system() == 'Windows':
        font_name = 'Microsoft YaHei'
    elif platform.system() == 'Darwin':
        font_name = 'PingFang SC'
    else:
        font_name = 'WenQuanYi Zen Hei'
    
    # 全局设置字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

    # 在设置完字体后应用 Seaborn 主题
    sns.set_theme(style="whitegrid", font=font_name) 
    #print(f"✅ 已尝试将字体设置为: {font_name}")

except Exception as e:
    print(f"⚠️ 字体设置失败: {e}")
    print("⚠️ 图像中的中文可能无法正常显示。")

# --- 【新增】全局颜色方案 ---
# 定义一个在 0.5 处有明显分界的颜色图，用于空间分布和热力图，确保视觉统一
nodes = [0.0, 0.5, 0.5, 1.0] 
colors = ["royalblue", "lightsteelblue", "lightcoral", "firebrick"]
OPINION_CMAP = LinearSegmentedColormap.from_list("sharp_diverging", list(zip(nodes, colors)))
#print("✅ 已定义全局观点颜色图 (OPINION_CMAP)。")


def plot_metrics_over_time(results_data: Dict[str, Any], output_dir: str):
    """
    (增强版) 遍历所有指标绘图，并为时空模型在图上标注外部事件的发生时间。
    【修改】：将事件标记从垂直线改为插值计算的红色叉叉，仅作用于受影响的层。
    """
    metrics_to_plot = results_data.get('metrics_calculated', [])
    metrics_data = results_data['results']['metrics_over_time']
    iterations = metrics_data.get('iteration', [])
    if not iterations: return

    layer_keys = [key for key in metrics_data if key.startswith('layer_')]
    
    sim_type = results_data.get("simulation_type")
    generated_events = results_data.get('results', {}).get('generated_events', [])

    for metric_name in tqdm(metrics_to_plot, desc="  -> Plotting Metrics"):
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        # 首先，正常绘制所有图层的数据线
        for layer_key in layer_keys:
            if metric_name in metrics_data[layer_key]:
                ax.plot(iterations, metrics_data[layer_key][metric_name], marker='o', linestyle='-', label=layer_key)
        
        # 如果存在事件，则在受影响的图层上绘制红色叉叉
        if sim_type == "spatiotemporal" and generated_events:
            event_label_added = False
            for event in generated_events:
                event_time = event.get('t0')
                affected_layer_idx = event.get('layer_index')

                if event_time is None or affected_layer_idx is None:
                    continue

                target_layer_key = f"layer_{affected_layer_idx}"
                if target_layer_key in metrics_data and metric_name in metrics_data[target_layer_key]:
                    y_values = metrics_data[target_layer_key][metric_name]
                    interpolated_y = np.interp(event_time, iterations, y_values)
                    
                    label = '外部事件' if not event_label_added else None
                    # 【修改】增加 markeredgewidth 使叉叉变粗
                    ax.plot(event_time, interpolated_y, 'rx', markersize=12, markeredgewidth=3, label=label)
                    event_label_added = True
        
        ax.set_title(f"指标 '{metric_name}' 随时间演化", fontsize=16)
        ax.set_xlabel("迭代次数 (Iteration)", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend(title="图例")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plot_filename = os.path.join(output_dir, f"metric_{metric_name}_evolution.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_spatial_final_state(results_data: Dict[str, Any], output_dir: str):
    """
    (更新版) 为 spatiotemporal 模型的【每一层】绘制最终状态的 2D 空间观点分布图。
    - 绿色五角星: 事件作用于当前绘制的层。
    - 黄色五角星: 事件作用于其他层。
    """
    if results_data.get("simulation_type") != "spatiotemporal":
        return

    print("  -> Plotting Spatial Distribution for all layers (专属图表)...")
    
    raw_history = results_data['results'].get('raw_history', [])
    node_positions = results_data['results'].get('node_positions')
    generated_events = results_data.get('results', {}).get('generated_events', [])

    if not raw_history or not node_positions:
        print("     Warning: 缺少观点历史或节点位置数据，跳过空间分布图。")
        return
        
    opinions_snapshot_gpu = raw_history[-1]['opinions_snapshot']
    final_opinions_snapshot = opinions_snapshot_gpu.get() if hasattr(opinions_snapshot_gpu, 'get') else np.array(opinions_snapshot_gpu)

    node_positions_gpu = node_positions
    positions = node_positions_gpu.get() if hasattr(node_positions_gpu, 'get') else np.array(node_positions_gpu)
    num_layers = final_opinions_snapshot.shape[1]

    for plotted_layer_idx in tqdm(range(num_layers), desc="  -> Plotting Spatial States"):
        
        final_opinions = final_opinions_snapshot[:, plotted_layer_idx]

        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # 【修改】使用全局的 OPINION_CMAP 颜色图
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=final_opinions, cmap=OPINION_CMAP, s=50, alpha=0.8, vmin=0, vmax=1)
        
        if generated_events:
            affecting_events_pos = []
            other_layer_events_pos = []
            
            for event in generated_events:
                if event.get('layer_index') == plotted_layer_idx:
                    affecting_events_pos.append(event['center'])
                else:
                    other_layer_events_pos.append(event['center'])
            
            if affecting_events_pos:
                affecting_events_pos = np.array(affecting_events_pos)
                ax.scatter(affecting_events_pos[:, 0], affecting_events_pos[:, 1], 
                           c='lime', marker='*', s=300, edgecolor='black', 
                           label=f'作用于本层 (Layer {plotted_layer_idx}) 的事件')

            if other_layer_events_pos:
                other_layer_events_pos = np.array(other_layer_events_pos)
                ax.scatter(other_layer_events_pos[:, 0], other_layer_events_pos[:, 1], 
                           c='yellow', marker='*', s=300, edgecolor='black', 
                           label='作用于其他层的事件')

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('最终观点值', rotation=270, labelpad=15)

        ax.set_title(f'第 {plotted_layer_idx} 层: 最终观点空间分布与事件中心', fontsize=16)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

        plot_filename = os.path.join(output_dir, f"spatial_final_distribution_layer_{plotted_layer_idx}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_opinion_distributions(results_data: Dict[str, Any], output_dir: str):
    """
    为每个网络层绘制初始和最终观点分布的核密度估计图(KDE)。
    """
    raw_history = results_data['results'].get('raw_history', [])
    if len(raw_history) < 2:
        print("     Warning: Insufficient history data. Skipping distribution plots.")
        return
        
    initial_opinions = np.array(raw_history[0]['opinions_snapshot'])
    final_opinions = np.array(raw_history[-1]['opinions_snapshot'])
    final_iteration = raw_history[-1]['iteration']
    
    num_layers = initial_opinions.shape[1]

    for layer_idx in tqdm(range(num_layers), desc="  -> Plotting Distributions"):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        sns.kdeplot(initial_opinions[:, layer_idx], ax=ax, fill=True, label=f'初始 (t=0)')
        sns.kdeplot(final_opinions[:, layer_idx], ax=ax, fill=True, label=f'最终 (t={final_iteration})')
        
        ax.set_title(f"第 {layer_idx} 层: 观点分布对比", fontsize=16)
        ax.set_xlabel("观点值", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.legend()
        
        plot_filename = os.path.join(output_dir, f"distribution_layer_{layer_idx}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_opinion_evolution_heatmap(results_data: Dict[str, Any], output_dir: str):
    """
    为每个网络层绘制观点随时间演化的热力图。
    """
    raw_history = results_data['results'].get('raw_history', [])
    if not raw_history:
        print("     Warning: No history data found. Skipping heatmaps.")
        return

    iterations = [h['iteration'] for h in raw_history]
    num_nodes = len(raw_history[0]['opinions_snapshot'])
    num_layers = len(raw_history[0]['opinions_snapshot'][0])
    num_timesteps = len(iterations)

    # 【移除】局部颜色图定义，将使用全局 OPINION_CMAP

    for layer_idx in tqdm(range(num_layers), desc="  -> Plotting Heatmaps"):
        opinion_matrix_time_rows = np.array([
            [snapshot[node_idx][layer_idx] for node_idx in range(num_nodes)]
            for snapshot in [h['opinions_snapshot'] for h in raw_history]
        ])
        opinion_matrix_node_rows = opinion_matrix_time_rows.T
        sorted_at_each_step_matrix = np.zeros((num_nodes, num_timesteps))
        for t in range(num_timesteps):
            sorted_at_each_step_matrix[:, t] = np.sort(opinion_matrix_node_rows[:, t])

        plt.figure(figsize=(15, 8))
        # 【修改】使用全局的 OPINION_CMAP 颜色图
        ax = sns.heatmap(
            sorted_at_each_step_matrix,
            cmap=OPINION_CMAP,
            center=0.5, 
            cbar_kws={'label': '观点值'}
        )
        
        ax.set_title(f"第 {layer_idx} 层: 观点分布演化 (每步独立排序)", fontsize=16)
        ax.set_xlabel("迭代次数 (Iteration)", fontsize=12)
        ax.set_ylabel(f"节点 (按当前观点排序)", fontsize=12)

        tick_positions = np.linspace(0, len(iterations) - 1, num=min(len(iterations), 11), dtype=int)
        ax.set_xticks(tick_positions + 0.5)
        ax.set_xticklabels([iterations[i] for i in tick_positions])
        ax.set_yticks([])
        ax.set_yticklabels([])

        plot_filename = os.path.join(output_dir, f"heatmap_layer_{layer_idx}_sorted_each_step.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

def generate_all_plots(results_data: Dict[str, Any], output_dir_base: str):
    """
    (更新版) 主函数，调用所有绘图函数。
    """
    output_dir = os.path.join("results", output_dir_base)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- 准备在 '{output_dir}' 文件夹中生成可视化图表 ---")
    
    try:
        plot_metrics_over_time(results_data, output_dir)
        plot_opinion_distributions(results_data, output_dir)
        plot_opinion_evolution_heatmap(results_data, output_dir)
        plot_spatial_final_state(results_data, output_dir)
        
        print("--- 图表生成完毕 ---")
    except Exception as e:
        print(f"!!! 生成图表时发生错误: {e}")