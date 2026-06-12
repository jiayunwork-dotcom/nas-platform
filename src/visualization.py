"""
可视化模块
包含: 帕累托前沿可视化、DAG图渲染、超体积曲线、网络结构展示
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import List, Tuple, Dict, Optional
import io
from PIL import Image

from .cell import Architecture, OP_COLORS
from .dag_utils import to_networkx_graph
from .metrics import fast_non_dominated_sort, hypervolume, get_reference_point, crowding_distance


def get_pareto_front_indices(points: np.ndarray, maximize: List[bool]) -> List[int]:
    """获取帕累托前沿点的索引"""
    fronts = fast_non_dominated_sort(points, maximize)
    return fronts[0] if fronts else []


def plot_pareto_2d(points: np.ndarray, maximize: List[bool],
                   x_dim: int = 1, y_dim: int = 0,
                   x_label: str = '参数量', y_label: str = '精度',
                   title: str = '帕累托前沿',
                   pareto_indices: Optional[List[int]] = None,
                   color: str = '#FF4444',
                   showlegend: bool = True,
                   customdata: Optional[np.ndarray] = None) -> go.Figure:
    """
    绘制二维帕累托散点图，支持点击交互
    """
    if pareto_indices is None:
        pareto_indices = get_pareto_front_indices(points, maximize)

    pareto_mask = np.zeros(len(points), dtype=bool)
    pareto_mask[pareto_indices] = True

    all_indices = np.arange(len(points))

    fig = go.Figure()

    dominated_x = points[~pareto_mask, x_dim]
    dominated_y = points[~pareto_mask, y_dim]
    dominated_idx = all_indices[~pareto_mask]
    if len(dominated_x) > 0:
        hover_text = [f'架构 {idx}<br>精度: {points[idx, 0]:.4f}<br>参数量: {points[idx, 1]/1e6:.2f}M<br>延迟: {points[idx, 2]:.3f}ms'
                      for idx in dominated_idx]
        fig.add_trace(go.Scatter(
            x=dominated_x,
            y=dominated_y,
            mode='markers',
            marker=dict(color='#888888', size=8, opacity=0.6),
            name='被支配解',
            showlegend=showlegend,
            customdata=dominated_idx.reshape(-1, 1) if customdata is None else customdata[~pareto_mask],
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text
        ))

    pareto_x = points[pareto_mask, x_dim]
    pareto_y = points[pareto_mask, y_dim]
    pareto_idx = all_indices[pareto_mask]
    if len(pareto_x) > 0:
        sort_idx = np.argsort(pareto_x)
        hover_text = [f'架构 {pareto_idx[i]}<br>精度: {points[pareto_idx[i], 0]:.4f}<br>参数量: {points[pareto_idx[i], 1]/1e6:.2f}M<br>延迟: {points[pareto_idx[i], 2]:.3f}ms'
                      for i in range(len(pareto_idx))]
        fig.add_trace(go.Scatter(
            x=pareto_x[sort_idx],
            y=pareto_y[sort_idx],
            mode='lines+markers',
            marker=dict(color=color, size=10, line=dict(width=2, color='white')),
            line=dict(color=color, width=2),
            name='帕累托前沿',
            showlegend=showlegend,
            customdata=pareto_idx[sort_idx].reshape(-1, 1) if customdata is None else customdata[pareto_mask][sort_idx],
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=[hover_text[i] for i in sort_idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        width=600,
        height=500,
        legend=dict(x=0.01, y=0.99),
        clickmode='event+select'
    )

    return fig


def plot_pareto_3d(points: np.ndarray, maximize: List[bool],
                   labels: List[str] = ['精度', '参数量', '延迟'],
                   title: str = '三目标帕累托前沿',
                   pareto_indices: Optional[List[int]] = None,
                   color: str = '#FF4444') -> go.Figure:
    """
    绘制三维帕累托散点图
    """
    if pareto_indices is None:
        pareto_indices = get_pareto_front_indices(points, maximize)

    pareto_mask = np.zeros(len(points), dtype=bool)
    pareto_mask[pareto_indices] = True

    fig = go.Figure()

    dominated = points[~pareto_mask]
    if len(dominated) > 0:
        fig.add_trace(go.Scatter3d(
            x=dominated[:, 0],
            y=dominated[:, 1],
            z=dominated[:, 2],
            mode='markers',
            marker=dict(color='#888888', size=5, opacity=0.5),
            name='被支配解'
        ))

    pareto = points[pareto_mask]
    if len(pareto) > 0:
        fig.add_trace(go.Scatter3d(
            x=pareto[:, 0],
            y=pareto[:, 1],
            z=pareto[:, 2],
            mode='markers',
            marker=dict(color=color, size=7, line=dict(width=2, color='white')),
            name='帕累托非支配解'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2]
        ),
        template='plotly_white',
        width=700,
        height=600
    )

    return fig


def plot_multi_experiment_pareto(experiments_data: Dict[str, np.ndarray],
                                 maximize: List[bool],
                                 x_dim: int = 1, y_dim: int = 0,
                                 x_label: str = '参数量', y_label: str = '精度',
                                 title: str = '多实验帕累托对比') -> go.Figure:
    """
    绘制多个实验的帕累托前沿对比图
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (exp_name, points) in enumerate(experiments_data.items()):
        color = colors[i % len(colors)]
        pareto_indices = get_pareto_front_indices(points, maximize)
        pareto_mask = np.zeros(len(points), dtype=bool)
        pareto_mask[pareto_indices] = True

        pareto_x = points[pareto_mask, x_dim]
        pareto_y = points[pareto_mask, y_dim]
        if len(pareto_x) > 0:
            sort_idx = np.argsort(pareto_x)
            fig.add_trace(go.Scatter(
                x=pareto_x[sort_idx],
                y=pareto_y[sort_idx],
                mode='lines+markers',
                marker=dict(color=color, size=10, line=dict(width=2, color='white')),
                line=dict(color=color, width=2),
                name=f'{exp_name} - 帕累托前沿'
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        width=700,
        height=500
    )

    return fig


def plot_hypervolume_curve(hypervolumes: List[float],
                           title: str = '超体积随代数变化',
                           label: str = '超体积',
                           color: str = '#1f77b4') -> go.Figure:
    """
    绘制超体积变化曲线
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(hypervolumes))),
        y=hypervolumes,
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=3, color=color),
        name=label
    ))

    fig.update_layout(
        title=title,
        xaxis_title='代数',
        yaxis_title='超体积',
        template='plotly_white',
        width=600,
        height=400
    )

    return fig


def plot_multi_hypervolume_curves(experiments_data: Dict[str, List[float]],
                                  title: str = '多实验超体积对比') -> go.Figure:
    """
    绘制多个实验的超体积曲线对比
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (exp_name, hvs) in enumerate(experiments_data.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=list(range(len(hvs))),
            y=hvs,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=3, color=color),
            name=exp_name
        ))

    fig.update_layout(
        title=title,
        xaxis_title='代数',
        yaxis_title='超体积',
        template='plotly_white',
        width=700,
        height=450
    )

    return fig


def plot_dag_graph(adj: np.ndarray, op_list: List[int], enabled_ops: List[str],
                   title: str = 'Cell结构') -> go.Figure:
    """
    使用plotly绘制DAG图
    """
    G = to_networkx_graph(adj, op_list, enabled_ops)
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, seed=42)
        for node in pos:
            pos[node] = (pos[node][0] * 200 + 250, pos[node][1] * 200 + 200)

    fig = go.Figure()

    edge_x = []
    edge_y = []
    edge_colors = []
    edge_labels = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        op_name = edge[2].get('op', 'unknown')
        edge_colors.append(OP_COLORS.get(op_name, '#888888'))
        edge_labels.append(op_name)

    for i in range(0, len(edge_x), 3):
        if edge_x[i] is not None:
            color = edge_colors[i // 3]
            fig.add_trace(go.Scatter(
                x=[edge_x[i], edge_x[i + 1]],
                y=[edge_y[i], edge_y[i + 1]],
                mode='lines',
                line=dict(width=3, color=color),
                hoverinfo='text',
                text=edge_labels[i // 3],
                showlegend=False
            ))

    node_x = []
    node_y = []
    node_labels = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=35, color='#3498db', line=dict(width=3, color='white')),
        text=node_labels,
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        width=500,
        height=400,
        margin=dict(b=20, l=5, r=5, t=40)
    )

    return fig


def plot_network_architecture(arch: Architecture, num_cells: int = 20,
                              title: str = '网络架构') -> go.Figure:
    """
    绘制完整网络结构（Cell堆叠图）
    """
    fig = go.Figure()

    y_positions = list(range(num_cells, 0, -1))
    cell_types = []
    for cell_idx in range(num_cells):
        reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
        cell_types.append(('Reduction Cell' if reduction else 'Normal Cell', reduction))

    x_normal = []
    y_normal = []
    x_reduction = []
    y_reduction = []

    for i, (cell_type, is_reduction) in enumerate(cell_types):
        if is_reduction:
            x_reduction.append(1)
            y_reduction.append(y_positions[i])
        else:
            x_normal.append(1)
            y_normal.append(y_positions[i])

    fig.add_trace(go.Scatter(
        x=x_normal,
        y=y_normal,
        mode='markers',
        marker=dict(size=40, color='#27ae60', symbol='square',
                    line=dict(width=3, color='white')),
        name='Normal Cell',
        text=['Normal Cell'] * len(x_normal),
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=x_reduction,
        y=y_reduction,
        mode='markers',
        marker=dict(size=45, color='#e74c3c', symbol='diamond',
                    line=dict(width=3, color='white')),
        name='Reduction Cell',
        text=['Reduction Cell (降采样)'] * len(x_reduction),
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=[1],
        y=[num_cells + 1],
        mode='markers',
        marker=dict(size=35, color='#3498db', symbol='circle'),
        name='Stem',
        text=['Stem (3x3 Conv)'],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=[1],
        y=[0],
        mode='markers',
        marker=dict(size=35, color='#9b59b6', symbol='circle'),
        name='Classifier',
        text=['Global Pool + Linear'],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, num_cells + 2], showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        width=350,
        height=600
    )

    return fig


def create_pareto_animation(all_generations_points: List[np.ndarray],
                            maximize: List[bool],
                            x_dim: int = 1, y_dim: int = 0,
                            x_label: str = '参数量', y_label: str = '精度') -> go.Figure:
    """
    创建帕累托前沿演变动画
    """
    frames = []
    max_x = max([np.max(p[:, x_dim]) for p in all_generations_points]) * 1.1
    min_x = min([np.min(p[:, x_dim]) for p in all_generations_points]) * 0.9
    max_y = max([np.max(p[:, y_dim]) for p in all_generations_points]) * 1.1
    min_y = min([np.min(p[:, y_dim]) for p in all_generations_points]) * 0.9

    for gen, points in enumerate(all_generations_points):
        pareto_indices = get_pareto_front_indices(points, maximize)
        pareto_mask = np.zeros(len(points), dtype=bool)
        pareto_mask[pareto_indices] = True

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=points[~pareto_mask, x_dim],
                    y=points[~pareto_mask, y_dim],
                    mode='markers',
                    marker=dict(color='#888888', size=8, opacity=0.6),
                    name='被支配解'
                ),
                go.Scatter(
                    x=points[pareto_mask, x_dim],
                    y=points[pareto_mask, y_dim],
                    mode='markers',
                    marker=dict(color='#FF4444', size=10, line=dict(width=2, color='white')),
                    name='帕累托前沿'
                )
            ],
            name=f'gen_{gen}',
            layout=go.Layout(title_text=f'第 {gen} 代帕累托前沿')
        ))

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            xaxis=dict(title=x_label, range=[min_x, max_x]),
            yaxis=dict(title=y_label, range=[min_y, max_y]),
            title='帕累托前沿演变动画',
            template='plotly_white',
            width=650,
            height=550,
            updatemenus=[dict(
                type='buttons',
                buttons=[dict(
                    label='播放',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True
                    )]
                )]
            )]
        ),
        frames=frames
    )

    return fig


def plot_operations_legend() -> go.Figure:
    """
    绘制操作图例
    """
    fig = go.Figure()

    op_display_names = {
        'conv3x3': '3x3 卷积',
        'conv5x5': '5x5 卷积',
        'dil_conv3x3': '3x3 扩张卷积',
        'max_pool3x3': '3x3 最大池化',
        'avg_pool3x3': '3x3 平均池化',
        'skip_connect': '恒等连接',
        'zero': '无连接'
    }

    for i, (op_name, color) in enumerate(OP_COLORS.items()):
        fig.add_trace(go.Scatter(
            x=[0],
            y=[-i],
            mode='markers',
            marker=dict(size=20, color=color),
            name=op_display_names.get(op_name, op_name),
            showlegend=True
        ))

    fig.update_layout(
        title='操作类型图例',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-7, 1]),
        template='plotly_white',
        width=300,
        height=400
    )

    return fig


def plot_adjacency_heatmap(adj: np.ndarray, title: str = '邻接矩阵') -> go.Figure:
    """
    绘制邻接矩阵热力图
    """
    n = adj.shape[0]
    fig = go.Figure(data=go.Heatmap(
        z=adj.astype(int),
        x=list(range(n)),
        y=list(range(n)),
        colorscale=[[0, '#f0f0f0'], [1, '#3498db']],
        showscale=False,
        hoverongaps=False,
        text=[[f'{i}→{j}' if adj[i, j] else '' for j in range(n)] for i in range(n)],
        texttemplate='%{text}',
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='目标节点',
        yaxis_title='源节点',
        template='plotly_white',
        width=400,
        height=350
    )

    return fig


def plot_hypervolume_curve_with_convergence(
    hypervolumes: List[float],
    convergence_gen: Optional[int] = None,
    title: str = '超体积随代数变化',
    label: str = '超体积',
    color: str = '#1f77b4'
) -> go.Figure:
    """
    绘制带收敛判定线的超体积变化曲线
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(hypervolumes))),
        y=hypervolumes,
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=3, color=color),
        name=label
    ))

    if convergence_gen is not None and 0 <= convergence_gen < len(hypervolumes):
        fig.add_vline(
            x=convergence_gen,
            line_dash='dash',
            line_color='#27ae60',
            line_width=3,
            annotation_text='收敛点',
            annotation_position='top right',
            annotation_font_color='#27ae60'
        )
        fig.add_trace(go.Scatter(
            x=[convergence_gen],
            y=[hypervolumes[convergence_gen]],
            mode='markers',
            marker=dict(size=15, color='#27ae60', symbol='diamond',
                        line=dict(width=3, color='white')),
            name='收敛点',
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        xaxis_title='代数',
        yaxis_title='超体积',
        template='plotly_white',
        width=700,
        height=450,
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def detect_convergence(hypervolumes: List[float],
                       threshold: float = 0.01,
                       window_size: int = 5) -> Tuple[Optional[int], float, float]:
    """
    检测搜索收敛点

    Args:
        hypervolumes: 超体积历史列表
        threshold: 变化率阈值 (默认1%)
        window_size: 连续窗口大小 (默认5代)

    Returns:
        (convergence_gen, avg_recent_rate, convergence_rate): 
            convergence_gen: 收敛代数，未收敛则为None
            avg_recent_rate: 最近窗口的平均变化率
            convergence_rate: 收敛时窗口内的最大变化率，未收敛则为最近窗口的最大变化率
    """
    if len(hypervolumes) < window_size + 1:
        return None, 0.0, 0.0

    change_rates = []
    for i in range(1, len(hypervolumes)):
        if hypervolumes[i - 1] != 0:
            rate = abs(hypervolumes[i] - hypervolumes[i - 1]) / abs(hypervolumes[i - 1])
        else:
            rate = 1.0
        change_rates.append(rate)

    convergence_gen = None
    convergence_rate = 0.0
    for i in range(len(change_rates) - window_size + 1):
        window_rates = change_rates[i:i + window_size]
        if all(r < threshold for r in window_rates):
            convergence_gen = i + window_size
            convergence_rate = max(window_rates)
            break

    recent_window = change_rates[-window_size:] if len(change_rates) >= window_size else change_rates
    avg_recent_rate = np.mean(recent_window) if recent_window else 0.0
    max_recent_rate = np.max(recent_window) if recent_window else 0.0

    if convergence_gen is None:
        convergence_rate = max_recent_rate

    return convergence_gen, avg_recent_rate, convergence_rate


def get_arch_rank_and_crowding(points: np.ndarray,
                               arch_idx: int,
                               maximize: List[bool]) -> Tuple[int, float]:
    """
    获取架构的非支配层级和拥挤距离

    Args:
        points: 所有架构的目标值矩阵
        arch_idx: 目标架构索引
        maximize: 各目标是否最大化

    Returns:
        (rank, crowding_distance): 非支配层级和拥挤距离
    """
    fronts = fast_non_dominated_sort(points, maximize)

    rank = -1
    front_idx_in_front = -1
    for front_rank, front in enumerate(fronts):
        if arch_idx in front:
            rank = front_rank
            front_idx_in_front = front.index(arch_idx)
            break

    if rank == -1:
        return -1, 0.0

    distances = crowding_distance(points, fronts[rank])
    crowding = distances[front_idx_in_front]

    return rank, crowding


def plot_prediction_scatter(true_values: np.ndarray, pred_values: np.ndarray,
                            dim_name: str = '精度',
                            title: str = '预测值 vs 真实值') -> go.Figure:
    """
    绘制代理预测值vs真实值散点图，带y=x参考线

    Args:
        true_values: 真实值数组 [N]
        pred_values: 预测值数组 [N]
        dim_name: 维度名称
        title: 图表标题

    Returns:
        plotly Figure
    """
    fig = go.Figure()

    min_val = min(np.min(true_values), np.min(pred_values))
    max_val = max(np.max(true_values), np.max(pred_values))
    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='y=x (完美预测)'
    ))

    fig.add_trace(go.Scatter(
        x=true_values,
        y=pred_values,
        mode='markers',
        marker=dict(size=8, color='#1f77b4', opacity=0.7,
                    line=dict(width=1, color='white')),
        name='数据点',
        hovertemplate=f'真实值: %{{x:.4f}}<br>预测值: %{{y:.4f}}<br>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f'真实{dim_name}',
        yaxis_title=f'预测{dim_name}',
        template='plotly_white',
        width=550,
        height=500,
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def plot_pareto_with_uncertainty(points: np.ndarray, uncertainties: np.ndarray,
                                  maximize: List[bool],
                                  x_dim: int = 1, y_dim: int = 0,
                                  x_label: str = '参数量', y_label: str = '精度',
                                  title: str = '帕累托前沿 (点大小=不确定度)') -> go.Figure:
    """
    绘制帕累托图，用点大小编码不确定度

    Args:
        points: 目标值矩阵 [N, 3]
        uncertainties: 不确定度矩阵 [N, 3]，使用综合不确定度
        maximize: 各目标是否最大化
        x_dim: x轴维度索引
        y_dim: y轴维度索引
        x_label: x轴标签
        y_label: y轴标签
        title: 图表标题

    Returns:
        plotly Figure
    """
    pareto_indices = get_pareto_front_indices(points, maximize)
    pareto_mask = np.zeros(len(points), dtype=bool)
    pareto_mask[pareto_indices] = True

    avg_uncertainty = np.mean(uncertainties, axis=1) if uncertainties.ndim > 1 else uncertainties

    if len(avg_uncertainty) > 0 and np.max(avg_uncertainty) > 0:
        marker_sizes = 5 + 20 * (avg_uncertainty / np.max(avg_uncertainty))
    else:
        marker_sizes = np.full(len(points), 8.0)

    fig = go.Figure()

    dominated_x = points[~pareto_mask, x_dim]
    dominated_y = points[~pareto_mask, y_dim]
    dominated_sizes = marker_sizes[~pareto_mask]
    dominated_unc = avg_uncertainty[~pareto_mask]
    if len(dominated_x) > 0:
        fig.add_trace(go.Scatter(
            x=dominated_x,
            y=dominated_y,
            mode='markers',
            marker=dict(color='#888888', size=dominated_sizes, opacity=0.6,
                        line=dict(width=1, color='white')),
            name='被支配解',
            hovertemplate=f'{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<br>不确定度: %{{text:.4f}}',
            text=dominated_unc
        ))

    pareto_x = points[pareto_mask, x_dim]
    pareto_y = points[pareto_mask, y_dim]
    pareto_sizes = marker_sizes[pareto_mask]
    pareto_unc = avg_uncertainty[pareto_mask]
    if len(pareto_x) > 0:
        sort_idx = np.argsort(pareto_x)
        fig.add_trace(go.Scatter(
            x=pareto_x[sort_idx],
            y=pareto_y[sort_idx],
            mode='lines+markers',
            marker=dict(color='#FF4444', size=pareto_sizes[sort_idx],
                        line=dict(width=2, color='white')),
            line=dict(color='#FF4444', width=2),
            name='帕累托非支配解',
            hovertemplate=f'{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<br>不确定度: %{{text:.4f}}',
            text=pareto_unc[sort_idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        width=650,
        height=550,
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def plot_surrogate_learning_curve(train_sizes: List[int], r2_scores: np.ndarray,
                                   target_names: List[str] = ['精度', '参数量', '延迟'],
                                   title: str = '代理模型学习曲线') -> go.Figure:
    """
    绘制代理模型学习曲线

    Args:
        train_sizes: 训练样本数列表
        r2_scores: R²分数 [N, 3]
        target_names: 各目标维度名称
        title: 图表标题

    Returns:
        plotly Figure
    """
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for dim in range(min(3, r2_scores.shape[1])):
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=r2_scores[:, dim],
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=3, color=colors[dim % len(colors)]),
            name=f'{target_names[dim]} R²'
        ))

    fig.add_hline(
        y=0.0,
        line_dash='dash',
        line_color='gray',
        line_width=1
    )

    fig.update_layout(
        title=title,
        xaxis_title='训练样本数',
        yaxis_title='验证集 R² 决定系数',
        template='plotly_white',
        width=700,
        height=450,
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def plot_eval_efficiency_bar(generations: List[int], actually_evaluated: List[int],
                              surrogate_skipped: List[int],
                              title: str = '每代评估架构数 vs 代理跳过数') -> go.Figure:
    """
    绘制每代实际评估数vs代理跳过数的堆叠柱状图

    Args:
        generations: 代数列表
        actually_evaluated: 每代实际评估数
        surrogate_skipped: 每代代理跳过数
        title: 图表标题

    Returns:
        plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=generations,
        y=actually_evaluated,
        name='实际评估',
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        x=generations,
        y=surrogate_skipped,
        name='代理跳过',
        marker_color='#ff7f0e'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='代数',
        yaxis_title='架构数',
        barmode='stack',
        template='plotly_white',
        width=700,
        height=450,
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def plot_strategy_duration_pie(strategy_durations: Dict[str, float],
                                title: str = '各评估策略使用时长占比') -> go.Figure:
    """
    绘制评估策略使用时长饼图

    Args:
        strategy_durations: {策略名: 时长}字典
        title: 图表标题

    Returns:
        plotly Figure
    """
    strategy_display_names = {
        'fast': '快速代理',
        'synflow': 'SynFlow零代价',
        'naswot': 'NASWOT零代价',
        'weight_sharing': '权重共享',
        'full': '完整训练'
    }

    labels = [strategy_display_names.get(k, k) for k in strategy_durations.keys()]
    values = list(strategy_durations.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title=title,
        template='plotly_white',
        width=550,
        height=500
    )

    return fig


def plot_param_diversity_curve(generations: List[int], diversities: List[float],
                                mutation_rates: List[float], crossover_rates: List[float],
                                title: str = '种群多样性与进化参数变化') -> go.Figure:
    """
    绘制种群多样性、变异率、交叉率随代数变化的曲线

    Args:
        generations: 代数列表
        diversities: 每代多样性值
        mutation_rates: 每代变异率
        crossover_rates: 每代交叉率
        title: 图表标题

    Returns:
        plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=generations, y=diversities, mode='lines+markers',
                   name='多样性', line=dict(color='#1f77b4', width=3), marker=dict(size=8)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=generations, y=mutation_rates, mode='lines+markers',
                   name='变异率', line=dict(color='#ff7f0e', width=2, dash='dash'), marker=dict(size=6)),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=generations, y=crossover_rates, mode='lines+markers',
                   name='交叉率', line=dict(color='#2ca02c', width=2, dash='dot'), marker=dict(size=6)),
        secondary_y=True,
    )

    fig.add_hline(y=0.3, line_dash='dash', line_color='red', line_width=1,
                  annotation_text='低多样性阈值(0.3)', secondary_y=False)
    fig.add_hline(y=0.7, line_dash='dash', line_color='green', line_width=1,
                  annotation_text='高多样性阈值(0.7)', secondary_y=False)

    fig.update_layout(
        title=title,
        xaxis_title='代数',
        template='plotly_white',
        width=700,
        height=450,
        legend=dict(x=0.01, y=0.99)
    )

    fig.update_yaxes(title_text='多样性 (0~1)', secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text='进化参数概率', secondary_y=True, range=[0, 1])

    return fig
