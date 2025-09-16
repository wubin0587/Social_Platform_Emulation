import os
from datetime import datetime
import numpy as np

# --- 1. 指标知识库：这是智能分析的核心 ---
# 我们在这里定义每个指标的“身份”和“解读方式”
METRIC_KNOWLEDGE_BASE = {
    # --- 观点分布指标 ---
    "opinion_variance": {
        "full_name": "观点方差",
        "type": "opinion",
        "interpretation": {
            "trend_decrease": "观点方差的持续下降是系统趋向共识的最直接证据。",
            "trend_increase": "观点方差的上升表明系统内部出现了意见分歧或极化。",
            "final_low": "极低的方差值意味着该层的观点已高度统一。",
            "final_high": "较高的方差值说明该层内部仍存在显著的意见差异。"
        }
    },
    "homophilic_bimodality_coefficient": {
        "full_name": "同质性双峰系数",
        "type": "opinion",
        "interpretation": {
            "trend_decrease": "该系数的下降通常表示网络从两极分化的结构向单一共识社群演变。",
            "trend_increase": "系数的上升，特别是当其值超过临界点(5/9)时，是观点极化形成两个对立阵营的强烈信号。",
            "final_low": "较低的系数值表明仿真结束时网络未形成显著的结构化对立团体。",
            "final_high": "较高的系数值 (大于 5/9) 意味着网络分裂为两个观点鲜明、内部同质的阵营。"
        }
    },
    "number_of_opinion_clusters": {
        "full_name": "观点集群数量",
        "type": "opinion",
        "interpretation": {
            "trend_decrease": "观点集群数量的减少反映了意见的合并与统一过程。",
            "trend_increase": "集群数量的增加则标志着观点的碎片化，形成了多个不同的小团体。",
            "final_low": "最终集群数为1或2，分别代表完全共识或简单的两极对立。",
            "final_high": "最终存在多个集群，说明网络形成了多中心、碎片化的舆论格局。"
        }
    },
    # --- 网络结构指标 ---
    "network_density": {
        "full_name": "网络密度",
        "type": "structure",
        "interpretation": {
            "trend_decrease": "网络密度的下降表明连接正在减少，网络变得稀疏，这可能是由于意见分歧导致的“断交”。",
            "trend_increase": "网络密度的上升意味着新连接的形成多于断开，网络变得更紧密。",
            "final_low": "最终的低密度网络表明个体间的联系相对松散。",
            "final_high": "最终的高密度网络意味着信息可以在其中快速流通。"
        }
    },
    "average_clustering_coefficient": {
        "full_name": "平均聚类系数",
        "type": "structure",
        "interpretation": {
            "trend_decrease": "平均聚类系数的下降可能意味着紧密的小团体正在瓦解。",
            "trend_increase": "系数的上升表明网络中正在形成更多“抱团”的紧密社群结构。",
            "final_low": "较低的聚类系数类似于随机网络，个体间的连接较为分散。",
            "final_high": "较高的聚类系数是“小世界”网络的重要特征，表明网络社群化程度高。"
        }
    },
    "average_shortest_path_length": {
        "full_name": "平均最短路径长度",
        "type": "structure",
        "interpretation": {
            "trend_decrease": "路径长度的缩短意味着节点间信息传递的效率在提高，网络变得更加“小世界化”。",
            "trend_increase": "路径长度的增加表明网络变得更加疏远和隔离，信息传播效率降低。",
            "final_low": "较短的平均路径是信息易于快速全局传播的标志。",
            "final_high": "较长的平均路径意味着信息可能更多地被局限在局部区域。"
        }
    },
    "get_giant_component_size": {
        "full_name": "最大连通子图（巨型组件）规模",
        "type": "structure",
        "interpretation": {
            "trend_decrease": "巨型组件规模的减小是网络发生分裂和碎片化的最危险信号，表明主流舆论场正在瓦解。",
            "trend_increase": "规模的增加说明孤立的节点或小团体正在重新融入主流网络。",
            "final_low": "远小于总节点数的规模，证实了网络在仿真结束时处于分裂状态。",
            "final_high": "接近总节点数的规模，表明网络保持了良好的整体连通性。"
        }
    },
     # --- 系统动态指标 (未来可添加) ---
    "opinion_aggregation_stability": {
        "full_name": "聚合观点稳定性",
        "type": "dynamic",
        "interpretation": {
            "trend_decrease": "该指标值趋向于零的过程，标志着系统从剧烈振荡的混沌期进入观点收敛的稳定期。",
            "trend_increase": "指标值的反弹或持续偏高，说明系统受到了扰动或未能达成稳定状态。",
            "final_low": "最终值接近于零，是仿真达到稳态或伪稳态的标志。",
            "final_high": "最终值依然较高，表明系统在仿真结束时仍处于活跃的演化中。"
        }
    }
}

def _format_params_for_latex(params) -> str:
    """
    (最终简化版) 递归地将 Python 字典或列表转换为格式化的 LaTeX 字符串。
    【新规则】: 只有字典会创建 itemize 环境，任何列表都会被直接转换为单行文本。
    """
    if isinstance(params, dict):
        if not params: return "{}"
        items_str = ""
        for key, value in params.items():
            key_escaped = str(key).replace('_', '\\_')
            # 递归调用以格式化值
            value_formatted = _format_params_for_latex(value)
            items_str += f"        \\item \\textbf{{{key_escaped}}}: {value_formatted}\n"
        # 字典是唯一会创建 itemize 的地方
        return f"\\begin{{itemize}}[noitemsep, topsep=0pt, leftmargin=*]\n{items_str}    \\end{{itemize}}"
    
    elif isinstance(params, list):
        # 【修改】对于任何列表，无论内容多复杂，都直接转换为字符串
        # 这样就从根本上杜绝了由列表引起的嵌套
        return str(params).replace('_', '\\_').replace('[', '$\\lbrack$').replace(']', '$\\rbrack$').replace("'", "")
    
    else:
        # 对于所有其他基本类型，直接转换为字符串
        return str(params).replace('_', '\\_')

def _generate_latex_content(data: dict) -> str:
    """
    根据仿真结果数据，生成一份统一的、支持多层的深度分析报告。
    对于时空仿真，会额外添加专属信息。
    """
    # ... (数据提取部分无变化) ...
    sim_type = data.get("simulation_type", "N/A")
    timestamp = data.get("timestamp", "N/A")
    config = data.get("config_used", {})
    network_config = config.get("network", {})
    sim_params = config.get("simulation_params", {})
    metrics_calculated = data.get("metrics_calculated", [])
    results = data.get("results", {})
    metrics_over_time = results.get("metrics_over_time", {})

    num_nodes = network_config.get('num_nodes', '未知')
    layers = network_config.get("layers", [])
    max_iterations = sim_params.get('max_iterations', '未知')
    
    escaped_metrics = ', '.join([m.replace('_', '\\_') for m in metrics_calculated])
    
    # --- 【修改】LaTeX 导言区：增加 enumitem 宏包 ---
    latex_preamble = f"""
\\documentclass[a4paper]{{ctexart}}
\\usepackage{{geometry}} \\usepackage{{amsmath}} \\usepackage{{amssymb}}
\\usepackage{{graphicx}} \\usepackage{{float}} \\usepackage{{caption}} \\usepackage{{booktabs}}
\\usepackage{{enumitem}}
\\geometry{{a4paper, left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm}}
\\title{{仿真实验深度分析报告}}
\\author{{自动生成}}
\\date{{{timestamp.split('T')[0] if timestamp != 'N/A' else datetime.now().strftime('%Y-%m-%d')}}}
"""
    latex_start = "\\begin{document}\n\\maketitle\n"

    # --- 2. 统一的摘要 (Abstract) ---
    abstract_intro = f"本文档深度分析了于 {timestamp} 执行的“{sim_type.replace('_', '\\_')}”类型仿真实验。"
    abstract_details = f"实验在一个包含 {num_nodes} 个节点、{len(layers)} 层的多层网络上进行，共迭代 {max_iterations} 次。"
    
    # 为时空仿真添加专属摘要信息
    if sim_type == "spatiotemporal":
        generated_events = results.get("generated_events", [])
        abstract_details += f" 本次仿真额外引入了空间维度和 {len(generated_events)} 个随机时空事件。"
        
    abstract_conclusion = f"""
报告不仅分析了最终状态，还深入探讨了仿真过程中的动态演化特性，并对各网络层的表现进行了横向比较。
主要分析指标包括：{escaped_metrics}。
"""
    latex_abstract = f"\\begin{{abstract}}\n{abstract_intro}\n{abstract_details}\n{abstract_conclusion}\n\\end{{abstract}}\n"

    config_section = "\\section{仿真配置}\n\\subsection{基础网络与仿真参数}\n\\begin{itemize}\n"
    config_section += f"    \\item \\textbf{{节点数}}: {num_nodes}\n"
    config_section += f"    \\item \\textbf{{网络层数}}: {len(layers)}\n"
    config_section += f"    \\item \\textbf{{最大迭代次数}}: {max_iterations}\n"
    if "dw_params" in sim_params:
        dw_mu = sim_params.get("dw_params", {}).get("mu", "未设置")
        config_section += f"    \\item \\textbf{{DW模型参数 ($\\mu$)}}: {str(dw_mu).replace('_', '\\_')}\n"
    coupling_params = sim_params.get("coupling_params", {})
    if coupling_params.get("enabled", False):
        config_section += f"""    \\item \\textbf{{层间耦合参数}}: 状态: 启用, $\\lambda$: {coupling_params.get('lambda', 'N/A')}, $c_{{kl, global}}$: {coupling_params.get('c_kl_global', 'N/A')}\n"""

    if sim_type == "spatiotemporal":
        sp_params = sim_params.get("spatiotemporal_params", {})
        interaction_prob_cfg = sp_params.get("interaction_prob", {})
        trust_scope_cfg = sp_params.get("trust_scope", {})
        learning_rate_cfg = sp_params.get("learning_rate", {})
        
        position_dist_config = network_config.get("position_distribution", {})
        pos_dist_type = position_dist_config.get('type', 'N/A')
        # 我们不再需要 .pop()，因为智能函数会处理整个字典
        params_for_display = {k: v for k, v in position_dist_config.items() if k != 'type'}

        config_section += "\\end{itemize}\n\\subsection{时空专属参数}\n\\begin{itemize}\n"
        
        config_section += f"    \\item \\textbf{{节点地理位置分布}}:\n"
        config_section += f"    \\begin{{itemize}}[noitemsep, topsep=0pt, leftmargin=*]\n"
        config_section += f"        \\item \\textbf{{类型}}: {pos_dist_type.replace('_', '\\_')}\n"
        if params_for_display:
            params_str = _format_params_for_latex(params_for_display)
            config_section += f"        \\item \\textbf{{具体参数}}:\n{params_str}\n"
        config_section += f"    \\end{{itemize}}\n"

        config_section += f"    \\item \\textbf{{泊松事件发生率}}: {sp_params.get('poisson_rate', 'N/A')}\n"
        config_section += f"    \\item \\textbf{{空间衰减因子 (alpha)}}: {sp_params.get('alpha', 'N/A')}\n"
        config_section += f"    \\item \\textbf{{时间衰减因子 (beta)}}: {sp_params.get('beta', 'N/A')}\n"
        config_section += f"""    \\item \\textbf{{动态交互概率}}: 基础值(base) = {interaction_prob_cfg.get('base', 'N/A')}, 增益(gain) = {interaction_prob_cfg.get('gain', 'N/A')}\n"""
        config_section += f"""    \\item \\textbf{{动态信任范围增益}}: {trust_scope_cfg.get('gain', 'N/A')}\n"""
        config_section += f"""    \\item \\textbf{{动态学习率增益}}: {learning_rate_cfg.get('gain', 'N/A')}\n"""
        config_section += "\\end{itemize}\n"
    else:
        config_section += "\\end{itemize}\n"
    
    config_section += "\\subsubsection*{各层网络类型}\n\\begin{itemize}\n"
    for i, layer in enumerate(layers):
        params = layer.get('params', {})
        param_items = [f"{str(k).replace('_', '\\_')}: {str(v).replace('_', '\\_')}" for k, v in params.items()]
        params_str = f"{{{', '.join(param_items)}}}"
        config_section += f"\\item \\textbf{{层 {i}}}: 网络类型为“{layer.get('type', 'N/A').replace('_', '\\_')}”，参数为 {params_str}。\n"
    config_section += "\\end{itemize}\n"

    # --- 4. 统一的结果分析章节 (Analysis Section) ---
    analysis_section = "\\section{结果分析}\n"

    # 为时空仿真添加专属的事件详情表格
    if sim_type == "spatiotemporal":
        generated_events = results.get("generated_events", [])
        if generated_events:
            analysis_section += "\\subsection*{时空事件详情}\n"
            analysis_section += "下表列出了仿真过程中所有生成的时空事件及其关键参数。\n"
            analysis_section += """
\\begin{table}[H]
    \\centering
    \\caption{时空事件记录表} \\label{tab:events}
    \\begin{tabular}{ccc}
        \\toprule
        \\textbf{事件序号} & \\textbf{发生时间 (迭代)} & \\textbf{中心位置 (x, y)} \\\\ \\midrule
"""
            for i, event in enumerate(generated_events):
                center_pos = event.get('center', ['N/A', 'N/A'])
                analysis_section += f"        事件 {i+1} & {event.get('t0', 'N/A')} & ({center_pos[0]:.4f}, {center_pos[1]:.4f}) \\\\\n"
            analysis_section += "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"

    # 统一的、支持多层的指标分析逻辑
    for metric in metrics_calculated:
        info = METRIC_KNOWLEDGE_BASE.get(metric, {})
        metric_cn = info.get("full_name", metric.replace("_", " "))
        analysis_section += f"\\subsection{{指标分析：{metric_cn}}}\n"

        all_layers_series = [np.array(metrics_over_time.get(f"layer_{i}", {}).get(metric, [])) for i in range(len(layers))]
        all_layers_series = [s for s in all_layers_series if s.size > 0]
        if not all_layers_series: continue
        
        iterations = np.array(metrics_over_time.get("iteration", []))
        if iterations.size < 2: continue
        avg_series = np.mean(all_layers_series, axis=0)
        
        initial_avg, final_avg = avg_series[0], avg_series[-1]
        trend = "下降" if final_avg < initial_avg else "上升"
        interp = info.get('interpretation', {}).get(f'trend_{"decrease" if trend == "下降" else "increase"}', '')
        peak_val, valley_val = np.max(avg_series), np.min(avg_series)
        peak_iter, valley_iter = iterations[np.argmax(avg_series)], iterations[np.argmin(avg_series)]

        analysis_section += f"""
\\subsubsection*{{总体演化趋势}}
该指标在所有层上的平均值呈现明显的{trend}趋势，从初始的 {initial_avg:.4f} 收敛至最终的 {final_avg:.4f}。{interp}

在演化过程中，所有层的平均指标在第 {peak_iter} 次迭代时达到峰值 {peak_val:.4f}，并在第 {valley_iter} 次迭代时达到谷值 {valley_val:.4f}。
绝大部分显著变化发生在仿真的早期阶段，后期系统逐渐进入稳定或缓慢收敛状态。
"""
        final_values = [s[-1] for s in all_layers_series]
        final_mean, final_std = np.mean(final_values), np.std(final_values)
        
        if final_mean > 1e-6 and (final_std / final_mean) > 0.5: diff_desc = "显著的分化"
        elif final_mean > 1e-6 and (final_std / final_mean) > 0.2: diff_desc = "中等程度的分化"
        else: diff_desc = "较小的一致性差异"

        max_idx, min_idx = np.argmax(final_values), np.argmin(final_values)
        max_val, min_val = final_values[max_idx], final_values[min_idx]
        interp_high = info.get('interpretation', {}).get('final_high', '')

        # --- 【您要求必须保留的详细分析内容】 ---
        analysis_section += f"""
\\subsubsection*{{最终状态的层间差异}}
在仿真结束时（第 {iterations[-1]} 次迭代），各层的“{metric_cn}”指标表现出一定的差异性。
所有层的最终指标均值为 {final_mean:.4f}，标准差为 {final_std:.4f}，表明各层之间存在{diff_desc}。
具体来看：
\\begin{{itemize}}
    \\item \\textbf{{表现最突出（最高值）的层是 {max_idx} 层}}，其最终值为 {max_val:.4f}。{interp_high if max_idx == np.argmax(final_values) else ''}
    \\item \\textbf{{表现最收敛（最低值）的层是 {min_idx} 层}}，其最终值为 {min_val:.4f}。
\\end{{itemize}}
这种层间的差异性可能与各层的网络结构（如小世界、无标度）及其在多层网络中的相互作用有关。
"""

    # --- 5. 结论 (Conclusion) ---
    conclusion_section = "\\section{结论与展望}\n"
    if sim_type == "spatiotemporal":
        conclusion_section += """
本次时空仿真实验揭示了在一个多层网络中，空间邻近性和外部随机事件对观点动力学的复杂影响。
分析结果表明，节点的空间分布和事件的发生位置对各网络层的局部收敛和全局格局均有重要影响。
未来的研究可以进一步探索不同网络拓扑（层）如何响应相同的时空扰动。
"""
    else: # 适用于 basic, dw3d 等
        conclusion_section += """
本次仿真揭示了在一个动态演化的多层网络中观点变化的复杂过程。
分析结果表明，即使网络结构和个体行为都具有动态性，系统整体上仍然表现出向共识收敛的宏观趋势。
同时，不同网络结构层在演化终态上表现出显著差异，这为理解特定网络拓扑在舆情演化中的作用提供了重要线索。
"""
    
    latex_end = r"\end{document}\n"
    return (
        latex_preamble +
        latex_start +
        latex_abstract +
        config_section +
        analysis_section +
        conclusion_section +
        latex_end
    )

def create_analysis_report(data: dict, output_basename: str, result_base_dir: str = "results"):
    """
    根据仿真数据生成并保存一份完整的LaTeX分析报告。
    """
    latex_content = _generate_latex_content(data)
    
    output_folder_path = os.path.join(result_base_dir, output_basename)
    output_tex_path = os.path.join(output_folder_path, f"{output_basename}.tex")

    print("\n--- 开始生成深度分析报告 ---")
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        with open(output_tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print(f"✅ 深度分析报告已成功保存至: {output_tex_path}")
    except IOError as e:
        print(f"❌ 错误：无法写入文件 {output_tex_path}。错误信息: {e}")
    except Exception as e:
        print(f"❌ 生成分析报告时发生未知错误: {e}")