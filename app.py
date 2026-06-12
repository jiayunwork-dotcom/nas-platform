"""
多目标进化神经架构搜索(NAS)实验管理平台
Streamlit主界面
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import tempfile
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cell import Architecture, OP_NAMES, OP_COLORS
from src.dag_utils import validate_architecture
from src.experiment import (
    ExperimentConfig, Experiment, ExperimentManager,
    GenerationSnapshot
)
from src.visualization import (
    plot_pareto_2d, plot_pareto_3d,
    plot_multi_experiment_pareto, plot_hypervolume_curve,
    plot_multi_hypervolume_curves, plot_dag_graph,
    plot_network_architecture, create_pareto_animation,
    plot_operations_legend, get_pareto_front_indices,
    plot_adjacency_heatmap, plot_hypervolume_curve_with_convergence,
    detect_convergence, get_arch_rank_and_crowding,
    plot_prediction_scatter, plot_pareto_with_uncertainty,
    plot_surrogate_learning_curve, plot_eval_efficiency_bar,
    plot_strategy_duration_pie, plot_param_diversity_curve,
    plot_dual_hypervolume_curves, plot_dual_strategy_timeline,
    plot_dual_pareto_scatter, plot_calibration_history,
    plot_tsne_scatter, compute_tsne_embedding,
    plot_param_diversity_curve_enhanced
)
from src.metrics import count_architecture_flops, hypervolume, get_reference_point

st.set_page_config(
    page_title="NAS实验管理平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

OP_DISPLAY_NAMES = {
    'conv3x3': '3x3 卷积',
    'conv5x5': '5x5 卷积',
    'dil_conv3x3': '3x3 扩张卷积',
    'max_pool3x3': '3x3 最大池化',
    'avg_pool3x3': '3x3 平均池化',
    'skip_connect': '恒等连接',
    'zero': '无连接'
}

EVAL_STRATEGIES = {
    'fast': '快速代理 (秒级)',
    'synflow': 'SynFlow零代价代理',
    'naswot': 'NASWOT零代价代理',
    'weight_sharing': '权重共享',
    'full': '完整训练 (最慢,最准确)'
}

MAXIMIZE = [True, False, False]


def render_architecture_detail(arch, config, points=None, arch_idx=None, surrogate=None):
    """
    渲染架构详情面板

    Args:
        arch: Architecture对象
        config: ExperimentConfig对象
        points: 种群目标值矩阵 (用于计算层级和拥挤距离)
        arch_idx: 架构在种群中的索引
        surrogate: 代理模型对象 (用于显示预测置信区间)
    """
    st.subheader("🏗️ 架构详情")

    if arch.is_evaluated:
        st.success("✅ 已评估 (真实评估)")
    else:
        st.info("🤖 代理预测")
        if surrogate is not None and surrogate.trained:
            try:
                pred = surrogate.predict([arch])
                mae, rmse = surrogate.get_prediction_error([arch])
                st.caption("代理模型预测值，置信区间基于训练误差估计")
                st.caption(
                    f"训练MAE: 精度 {mae[0]:.4f}, "
                    f"参数量 {mae[1]/1e3:.1f}K, "
                    f"延迟 {mae[2]:.3f}ms"
                )
            except:
                st.caption("该架构由代理模型预测，尚未经过真实评估")
        else:
            st.caption("该架构尚未经过真实评估")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("🎯 精度", f"{arch.accuracy:.4f}")
    with metric_cols[1]:
        st.metric("📦 参数量", f"{arch.params/1e6:.2f} M")
    with metric_cols[2]:
        st.metric("⏱️ 延迟", f"{arch.latency:.3f} ms")

    if points is not None and arch_idx is not None:
        rank, crowding = get_arch_rank_and_crowding(points, arch_idx, MAXIMIZE)
        info_cols = st.columns(2)
        with info_cols[0]:
            st.metric("📊 非支配层级", f"第 {rank} 层" if rank >= 0 else "未知")
        with info_cols[1]:
            crowding_str = f"{crowding:.4f}" if not np.isinf(crowding) else "∞ (边界点)"
            st.metric("📏 拥挤距离", crowding_str)

    st.markdown("---")
    st.subheader("🔬 Cell DAG 图")

    dag_col1, dag_col2 = st.columns(2)
    with dag_col1:
        fig_n = plot_dag_graph(
            arch.normal_adj,
            arch.normal_op_list,
            arch.enabled_ops,
            "Normal Cell"
        )
        st.plotly_chart(fig_n, use_container_width=True)

    with dag_col2:
        fig_r = plot_dag_graph(
            arch.reduce_adj,
            arch.reduce_op_list,
            arch.enabled_ops,
            "Reduction Cell"
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.subheader("🧬 架构编码向量")

    enc_col1, enc_col2 = st.columns(2)
    with enc_col1:
        fig_adj_n = plot_adjacency_heatmap(
            arch.normal_adj,
            "Normal Cell 邻接矩阵"
        )
        st.plotly_chart(fig_adj_n, use_container_width=True)
        st.caption("Normal Cell 操作列表:")
        st.code(str(arch.normal_op_list))

    with enc_col2:
        fig_adj_r = plot_adjacency_heatmap(
            arch.reduce_adj,
            "Reduction Cell 邻接矩阵"
        )
        st.plotly_chart(fig_adj_r, use_container_width=True)
        st.caption("Reduction Cell 操作列表:")
        st.code(str(arch.reduce_op_list))

    st.markdown("---")
    st.subheader("📐 网络信息")

    flops = count_architecture_flops(
        arch,
        num_cells=config.num_cells,
        init_channels=config.init_channels
    )
    info_cols2 = st.columns(3)
    with info_cols2[0]:
        st.metric("🔢 FLOPs", f"{flops/1e6:.2f} M")
    with info_cols2[1]:
        st.metric("🔗 Normal Cell边数", int(arch.normal_adj.sum()))
    with info_cols2[2]:
        st.metric("🔗 Reduction Cell边数", int(arch.reduce_adj.sum()))


def init_session_state():
    """初始化会话状态"""
    if 'exp_manager' not in st.session_state:
        st.session_state.exp_manager = ExperimentManager()
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'selected_arch_idx' not in st.session_state:
        st.session_state.selected_arch_idx = 0


def sidebar():
    """侧边栏导航"""
    with st.sidebar:
        st.title("🧬 NAS实验平台")
        st.markdown("---")

        page = st.radio(
            "导航",
            ["🏠 首页", "🔬 创建实验", "📊 查看实验", "📈 对比分析", "⚙️ 操作图例"],
            index=0
        )

        st.markdown("---")
        st.caption(f"已保存实验: {len(st.session_state.exp_manager.list_experiments())}")
        st.caption(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return page


def home_page():
    """首页"""
    st.title("🧬 多目标进化神经架构搜索实验管理平台")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🔬 搜索空间\n- Cell-based DAG架构\n- Normal/Reduction Cell\n- 灵活配置节点数和操作集")

    with col2:
        st.success("### 🧬 NSGA-II算法\n- 快速非支配排序\n- 拥挤距离计算\n- 锦标赛选择\n- 精英保留策略")

    with col3:
        st.warning("### 📊 多目标优化\n- 精度最大化\n- 参数量最小化\n- 推理延迟最小化")

    st.markdown("---")

    st.subheader("📋 功能概览")

    feature_cols = st.columns(4)

    with feature_cols[0]:
        st.metric("评估策略", "5种", delta="快速→准确")
        st.caption("完整训练、SynFlow、NASWOT、权重共享、快速代理")

    with feature_cols[1]:
        st.metric("代理模型", "MLP", delta="3层")
        st.caption("预筛候选架构，减少评估成本")

    with feature_cols[2]:
        st.metric("可视化", "多种", delta="2D/3D/动画")
        st.caption("帕累托图、DAG渲染、超体积曲线")

    with feature_cols[3]:
        st.metric("实验对比", "多实验", delta="基线对比")
        st.caption("随机搜索基线、外部结果导入")

    st.markdown("---")

    if st.button("🚀 开始创建新实验", type="primary", use_container_width=True):
        st.switch_page(st.__file__)

    exp_list = st.session_state.exp_manager.list_experiments()
    if exp_list:
        st.subheader("📂 已有实验")
        for exp_name in exp_list:
            with st.expander(f"📊 {exp_name}"):
                exp = st.session_state.exp_manager.get_experiment(exp_name)
                if exp:
                    config = exp.config
                    st.write(f"- **算法**: {config.algorithm}")
                    st.write(f"- **节点数**: {config.num_nodes}")
                    st.write(f"- **种群大小**: {config.pop_size}")
                    st.write(f"- **代数**: {config.num_generations}")
                    st.write(f"- **评估策略**: {EVAL_STRATEGIES.get(config.eval_strategy, config.eval_strategy)}")
                    st.write(f"- **创建时间**: {config.created_at}")
                    if exp.result.completed:
                        st.success(f"✅ 已完成 - 已评估 {len(exp.result.all_evaluated)} 个架构")
                        st.write(f"最终超体积: {exp.result.hypervolume_history[-1]:.4f}")
                    else:
                        st.info(f"⏳ 运行中 - 已完成 {len(exp.result.generations) - 1}/{config.num_generations} 代")


def create_experiment_page():
    """创建实验页面"""
    st.title("🔬 创建新实验")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 基本配置")

        exp_name = st.text_input("实验名称", value=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        algorithm = st.selectbox(
            "搜索算法",
            ["nsga2", "random"],
            format_func=lambda x: "NSGA-II 多目标进化" if x == "nsga2" else "随机搜索 (基线)"
        )

        num_nodes = st.slider("中间节点数", min_value=3, max_value=8, value=6,
                             help="Cell中中间节点数量（不含输入输出）")

        num_cells = st.slider("Cell堆叠层数", min_value=4, max_value=32, value=8,
                             help="完整网络中Cell堆叠数量")

        init_channels = st.slider("初始通道数", min_value=8, max_value=64, value=16)

        st.subheader("⚙️ 进化参数")

        pop_size = st.slider("种群大小", min_value=10, max_value=200, value=50)
        num_generations = st.slider("进化代数", min_value=5, max_value=100, value=20)
        mutation_rate = st.slider("变异概率", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        crossover_rate = st.slider("交叉概率", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    with col2:
        st.subheader("🔧 操作集配置")

        enabled_ops = []
        for op_name in OP_NAMES:
            default_val = True if op_name != 'zero' else False
            if st.checkbox(OP_DISPLAY_NAMES.get(op_name, op_name), value=default_val, key=f"op_{op_name}"):
                enabled_ops.append(op_name)

        if len(enabled_ops) == 0:
            st.error("⚠️ 至少选择一个操作类型")

        st.subheader("📊 评估策略")

        eval_strategy = st.selectbox(
            "评估方法",
            list(EVAL_STRATEGIES.keys()),
            format_func=lambda x: EVAL_STRATEGIES[x],
            index=0
        )

        if eval_strategy == 'full':
            eval_epochs = st.slider("训练Epoch数", min_value=5, max_value=100, value=20)
        else:
            eval_epochs = 20

        device = st.selectbox("计算设备", ["cpu", "cuda"], index=0)

        st.subheader("🤖 代理模型")

        use_surrogate = st.checkbox("启用代理模型预筛", value=True)
        if use_surrogate:
            surrogate_min_samples = st.slider("代理最小训练样本数", min_value=20, max_value=200, value=50)
            surrogate_percentile = st.slider("预筛保留百分比", min_value=10, max_value=100, value=30,
                                            help="保留代理预测排名前多少百分比的架构做真实评估")
        else:
            surrogate_min_samples = 50
            surrogate_percentile = 30.0

        st.subheader("🧠 自适应调度")

        use_adaptive = st.checkbox("启用自适应调度", value=True,
                                  help="根据搜索过程实时反馈自动切换评估策略和调整进化参数",
                                  key="cb_use_adaptive")

        adaptive_placeholder = st.container()
        with adaptive_placeholder:
            if use_adaptive:
                st.success("✅ 自适应调度已启用，将采用以下智能策略：")
                st.markdown("""
                - **阶段1（前5代）**: 使用SynFlow零代价代理快速筛选，积累初始样本
                - **阶段2（第6代+）**: 代理模型就绪后，切换为「代理预筛 + 快速真实评估」的混合模式
                - **双指标低增长触发**: 连续3代超体积增长率<5% 且 加权目标改善率<3%，自动升级评估策略
                - **多样性警报触发**: 帕累托前沿解减少>20%，强制对当前前沿解做完整评估校准
                - **进化参数自适应**: 种群多样性<0.3时变异率↑50%，>0.7时变异率↓30%，交叉率反向调整
                """)
                st.caption("💡 每次策略切换都会记录详细日志，可在实验详情页的「策略日志」tab查看。")

                st.markdown("---")
                st.subheader("⚖️ 目标偏好权重")
                st.caption("设置三个目标的相对权重（归一化到和为1），用于计算加权目标改善率")

                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    w_accuracy = st.slider("精度权重", min_value=0.0, max_value=1.0, value=1/3, step=0.05, key="w_acc")
                with col_w2:
                    w_params = st.slider("参数量权重", min_value=0.0, max_value=1.0, value=1/3, step=0.05, key="w_params")
                with col_w3:
                    w_latency = st.slider("延迟权重", min_value=0.0, max_value=1.0, value=1/3, step=0.05, key="w_latency")

                total_w = w_accuracy + w_params + w_latency
                if total_w > 0:
                    w_accuracy_norm = w_accuracy / total_w
                    w_params_norm = w_params / total_w
                    w_latency_norm = w_latency / total_w
                else:
                    w_accuracy_norm = w_params_norm = w_latency_norm = 1/3

                st.info(
                    f"📊 归一化后权重: 精度={w_accuracy_norm:.2f}, "
                    f"参数量={w_params_norm:.2f}, 延迟={w_latency_norm:.2f}"
                )
            else:
                st.info("ℹ️ 自适应调度已关闭，将使用固定的评估策略和进化参数（勾选上方复选框可启用智能调度）。")
                w_accuracy_norm = w_params_norm = w_latency_norm = 1/3

    st.markdown("---")

    col_start, col_preview, _ = st.columns([1, 1, 1])

    with col_start:
        if st.button("🚀 开始搜索", type="primary", disabled=len(enabled_ops) == 0, use_container_width=True):
            objective_weights = [w_accuracy_norm, w_params_norm, w_latency_norm] if use_adaptive else [1/3, 1/3, 1/3]
            config = ExperimentConfig(
                name=exp_name,
                algorithm=algorithm,
                num_nodes=num_nodes + 2,
                enabled_ops=enabled_ops,
                num_cells=num_cells,
                init_channels=init_channels,
                pop_size=pop_size,
                num_generations=num_generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                eval_strategy=eval_strategy,
                eval_epochs=eval_epochs,
                use_surrogate=use_surrogate,
                surrogate_min_samples=surrogate_min_samples,
                surrogate_percentile=surrogate_percentile,
                use_adaptive_scheduling=use_adaptive,
                objective_weights=objective_weights,
                device=device
            )

            try:
                exp = st.session_state.exp_manager.create_experiment(config)
                st.session_state.current_experiment = exp

                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()

                def progress_callback(current, total, message):
                    progress = current / total if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.info(message)

                exp.run(progress_callback=progress_callback)
                exp.save()

                status_text.success("✅ 搜索完成！")
                st.balloons()

            except Exception as e:
                st.error(f"❌ 错误: {e}")

    with col_preview:
        if st.button("👁️ 预览搜索空间", use_container_width=True):
            st.session_state.current_experiment = None

            sample_arch = Architecture(
                num_nodes=num_nodes + 2,
                enabled_ops=enabled_ops if enabled_ops else OP_NAMES[:-1]
            )
            _, sample_arch = validate_architecture(sample_arch, fix=True)

            st.subheader("📐 示例架构预览")

            dag_col1, dag_col2 = st.columns(2)
            with dag_col1:
                fig1 = plot_dag_graph(
                    sample_arch.normal_adj,
                    sample_arch.normal_op_list,
                    sample_arch.enabled_ops,
                    "Normal Cell 示例"
                )
                st.plotly_chart(fig1, use_container_width=True)
            with dag_col2:
                fig2 = plot_dag_graph(
                    sample_arch.reduce_adj,
                    sample_arch.reduce_op_list,
                    sample_arch.enabled_ops,
                    "Reduction Cell 示例"
                )
                st.plotly_chart(fig2, use_container_width=True)


def view_experiment_page():
    """查看实验页面"""
    st.title("📊 查看实验结果")
    st.markdown("---")

    exp_list = st.session_state.exp_manager.list_experiments()

    if not exp_list:
        st.info("📭 还没有已保存的实验，请先创建实验。")
        return

    selected_exp = st.selectbox("选择实验", exp_list)

    if selected_exp:
        exp = st.session_state.exp_manager.get_experiment(selected_exp)
        if not exp:
            return

        st.session_state.current_experiment = exp
        config = exp.config
        result = exp.result

        st.subheader(f"📋 实验配置: {config.name}")

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("算法", "NSGA-II" if config.algorithm == "nsga2" else "随机搜索")
        with info_cols[1]:
            st.metric("种群大小", config.pop_size)
        with info_cols[2]:
            st.metric("总代数", config.num_generations)
        with info_cols[3]:
            st.metric("已评估架构", len(result.all_evaluated))

        st.markdown("---")

        if not result.generations:
            st.info("⏳ 实验尚未开始运行...")
            return

        max_gen = len(result.generations) - 1
        current_gen = st.slider("查看代数", min_value=0, max_value=max_gen, value=max_gen,
                               help="拖动查看不同代数的种群状态")

        snapshot = result.generations[current_gen]
        points = snapshot.get_fitness_matrix()

        st.subheader(f"📈 第 {current_gen} 代种群状态")

        metric_cols = st.columns(6)
        with metric_cols[0]:
            st.metric("超体积", f"{snapshot.hypervolume:.4f}")
        with metric_cols[1]:
            st.metric("平均精度", f"{snapshot.avg_accuracy:.4f}")
        with metric_cols[2]:
            st.metric("平均参数量", f"{snapshot.avg_params/1e6:.2f}M")
        with metric_cols[3]:
            st.metric("平均延迟", f"{snapshot.avg_latency:.3f}ms")
        with metric_cols[4]:
            st.metric("🧬 多样性", f"{snapshot.diversity:.3f}")
            if snapshot.diversity < 0.3:
                st.caption("⚠️ 低多样性")
            elif snapshot.diversity > 0.7:
                st.caption("✅ 高多样性")
        with metric_cols[5]:
            strategy_display = EVAL_STRATEGIES.get(snapshot.eval_strategy, snapshot.eval_strategy)
            st.metric("🔄 评估策略", strategy_display.split(' ')[0])
            st.caption(f"变异率: {snapshot.mutation_rate:.3f} | 交叉率: {snapshot.crossover_rate:.3f}")

        if snapshot.surrogate_used:
            st.success(f"🤖 本代使用了代理模型预筛 - 实际评估: {snapshot.actually_evaluated}, 代理跳过: {snapshot.surrogate_skipped}")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🎯 帕累托前沿 2D", "🌐 帕累托前沿 3D", "📐 架构详情",
            "📊 历史曲线", "🧠 预测分析", "📊 效率统计", "📜 策略日志", "⚔️ 实时对比"
        ])

        with tab1:
            st.info("💡 点击散点图上的任意点查看该架构的详细信息")

            pareto_col1, pareto_col2 = st.columns(2)
            with pareto_col1:
                fig1 = plot_pareto_2d(
                    points, MAXIMIZE,
                    x_dim=1, y_dim=0,
                    x_label='参数量', y_label='精度',
                    title=f'精度 vs 参数量 (第{current_gen}代)'
                )
                selection1 = st.plotly_chart(
                    fig1, use_container_width=True,
                    on_select="rerun",
                    key=f"pareto1_{current_gen}_{selected_exp}"
                )

            with pareto_col2:
                fig2 = plot_pareto_2d(
                    points, MAXIMIZE,
                    x_dim=2, y_dim=0,
                    x_label='延迟 (ms)', y_label='精度',
                    title=f'精度 vs 延迟 (第{current_gen}代)'
                )
                selection2 = st.plotly_chart(
                    fig2, use_container_width=True,
                    on_select="rerun",
                    key=f"pareto2_{current_gen}_{selected_exp}"
                )

            selected_idx = None
            selection_data = None

            if selection1 and selection1.get('selection') and selection1['selection'].get('points'):
                selection_data = selection1['selection']['points']
            elif selection2 and selection2.get('selection') and selection2['selection'].get('points'):
                selection_data = selection2['selection']['points']

            if selection_data and len(selection_data) > 0:
                point_data = selection_data[0]
                if point_data.get('customdata') is not None:
                    selected_idx = int(point_data['customdata'][0])

            if selected_idx is not None and 0 <= selected_idx < len(snapshot.population):
                selected_arch = snapshot.population[selected_idx]

                st.markdown("---")
                st.info(f"📍 已选中架构 #{selected_idx}")
                render_architecture_detail(
                    selected_arch, config,
                    points=points,
                    arch_idx=selected_idx,
                    surrogate=exp.surrogate if hasattr(exp, 'surrogate') else None
                )
            else:
                st.info("👆 点击上方散点图中的任意点查看架构详情")

        with tab2:
            fig3d = plot_pareto_3d(
                points, MAXIMIZE,
                labels=['精度', '参数量', '延迟 (ms)'],
                title=f'三目标帕累托前沿 (第{current_gen}代)'
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            pareto_indices = get_pareto_front_indices(points, MAXIMIZE)
            pareto_archs = [snapshot.population[i] for i in pareto_indices]

            if pareto_archs:
                selected_idx = st.selectbox(
                    "选择帕累托前沿架构",
                    range(len(pareto_archs)),
                    format_func=lambda i: f"架构 {i} - 精度: {pareto_archs[i].accuracy:.4f}, "
                                        f"参数量: {pareto_archs[i].params/1e6:.2f}M, "
                                        f"延迟: {pareto_archs[i].latency:.3f}ms"
                )

                selected_arch = pareto_archs[selected_idx]

                arch_col1, arch_col2 = st.columns(2)
                with arch_col1:
                    fig_n = plot_dag_graph(
                        selected_arch.normal_adj,
                        selected_arch.normal_op_list,
                        selected_arch.enabled_ops,
                        "Normal Cell"
                    )
                    st.plotly_chart(fig_n, use_container_width=True)

                with arch_col2:
                    fig_r = plot_dag_graph(
                        selected_arch.reduce_adj,
                        selected_arch.reduce_op_list,
                        selected_arch.enabled_ops,
                        "Reduction Cell"
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                net_col, detail_col = st.columns(2)
                with net_col:
                    fig_net = plot_network_architecture(
                        selected_arch,
                        num_cells=config.num_cells,
                        title="完整网络结构"
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                with detail_col:
                    st.subheader("📊 架构详情")
                    flops = count_architecture_flops(
                        selected_arch,
                        num_cells=config.num_cells,
                        init_channels=config.init_channels
                    )

                    st.write(f"**精度**: {selected_arch.accuracy:.4f}")
                    st.write(f"**参数量**: {selected_arch.params/1e6:.2f} M")
                    st.write(f"**推理延迟**: {selected_arch.latency:.3f} ms")
                    st.write(f"**FLOPs**: {flops/1e6:.2f} M")
                    st.write(f"**Normal Cell边数**: {int(selected_arch.normal_adj.sum())}")
                    st.write(f"**Reduction Cell边数**: {int(selected_arch.reduce_adj.sum())}")

                    adj_norm = selected_arch.normal_adj.astype(int)
                    adj_red = selected_arch.reduce_adj.astype(int)

                    st.write("**Normal Cell邻接矩阵**:")
                    st.dataframe(pd.DataFrame(adj_norm))
                    st.write("**Normal Cell操作列表**:", selected_arch.normal_op_list)

                    st.write("**Reduction Cell邻接矩阵**:")
                    st.dataframe(pd.DataFrame(adj_red))
                    st.write("**Reduction Cell操作列表**:", selected_arch.reduce_op_list)
            else:
                st.info("本代暂无帕累托前沿架构")

        with tab4:
            convergence_gen, avg_change_rate, convergence_rate = detect_convergence(
                result.hypervolume_history,
                threshold=0.01,
                window_size=5
            )

            hv_fig = plot_hypervolume_curve_with_convergence(
                result.hypervolume_history,
                convergence_gen=convergence_gen,
                title='超体积随代数变化',
                label=config.name
            )
            st.plotly_chart(hv_fig, use_container_width=True)

            if convergence_gen is not None:
                st.success(
                    f"✅ 搜索于第 {convergence_gen} 代收敛，"
                    f"超体积变化率已降至 {convergence_rate * 100:.2f}% 以下"
                )
            else:
                st.info(
                    f"⏳ 搜索尚未收敛，"
                    f"当前最近5代平均变化率为 {avg_change_rate * 100:.2f}%"
                )

            if len(result.generations) > 1:
                st.subheader("🎬 帕累托演变动画")
                all_points = [gen.get_fitness_matrix() for gen in result.generations]
                anim_fig = create_pareto_animation(
                    all_points, MAXIMIZE,
                    x_dim=1, y_dim=0,
                    x_label='参数量', y_label='精度'
                )
                st.plotly_chart(anim_fig, use_container_width=True)

            if len(result.generations) > 1:
                st.subheader("🧬 种群多样性与进化参数变化")
                st.caption("💡 点击曲线上的点可查看该代种群的 t-SNE 可视化")

                gens_list = list(range(len(result.generations)))
                div_list = [gen.diversity for gen in result.generations]
                mut_list = [gen.mutation_rate for gen in result.generations]
                cross_list = [gen.crossover_rate for gen in result.generations]

                strategy_switch_gens = []
                strategy_switch_labels = []
                if result.strategy_switch_logs:
                    for log in result.strategy_switch_logs:
                        strategy_switch_gens.append(log.generation)
                        new_display = EVAL_STRATEGIES.get(log.new_strategy, log.new_strategy)
                        strategy_switch_labels.append(f'→{new_display.split(" ")[0]}')

                param_fig = plot_param_diversity_curve_enhanced(
                    gens_list, div_list, mut_list, cross_list,
                    strategy_switch_gens=strategy_switch_gens if strategy_switch_gens else None,
                    strategy_switch_labels=strategy_switch_labels if strategy_switch_labels else None,
                    title='种群多样性与进化参数随代数变化'
                )
                param_selection = st.plotly_chart(
                    param_fig, use_container_width=True,
                    on_select="rerun",
                    key=f"param_div_{selected_exp}"
                )

                selected_gen_for_tsne = None
                if param_selection and param_selection.get('selection') and param_selection['selection'].get('points'):
                    sel_points = param_selection['selection']['points']
                    if len(sel_points) > 0:
                        x_val = sel_points[0].get('x')
                        if x_val is not None and 0 <= int(x_val) < len(result.generations):
                            selected_gen_for_tsne = int(x_val)

                if selected_gen_for_tsne is None:
                    selected_gen_for_tsne = current_gen

                st.markdown("---")
                st.subheader(f"🔬 第 {selected_gen_for_tsne} 代种群编码 t-SNE 可视化")

                tsne_col1, tsne_col2 = st.columns([2, 1])

                with tsne_col1:
                    snapshot_gen = result.generations[selected_gen_for_tsne]
                    pop = snapshot_gen.population
                    points_gen = snapshot_gen.get_fitness_matrix()

                    encodings = np.array([arch.encode().astype(float) for arch in pop])
                    perplexity = min(30, len(pop) - 1)

                    with st.spinner(f"计算 t-SNE 嵌入 (perplexity={perplexity})..."):
                        embedding = compute_tsne_embedding(encodings, perplexity=perplexity)

                    tsne_fig = plot_tsne_scatter(
                        embedding, points_gen,
                        maximize=MAXIMIZE,
                        title=f'第 {selected_gen_for_tsne} 代种群 t-SNE 可视化'
                    )
                    tsne_selection = st.plotly_chart(
                        tsne_fig, use_container_width=True,
                        on_select="rerun",
                        key=f"tsne_{selected_exp}_{selected_gen_for_tsne}"
                    )

                with tsne_col2:
                    st.markdown("### 📍 选中架构详情")

                    selected_tsne_idx = None
                    if tsne_selection and tsne_selection.get('selection') and tsne_selection['selection'].get('points'):
                        sel_pts = tsne_selection['selection']['points']
                        if len(sel_pts) > 0 and sel_pts[0].get('customdata') is not None:
                            selected_tsne_idx = int(sel_pts[0]['customdata'])

                    if selected_tsne_idx is not None and 0 <= selected_tsne_idx < len(pop):
                        selected_arch = pop[selected_tsne_idx]

                        is_pareto = False
                        pareto_indices = get_pareto_front_indices(points_gen, MAXIMIZE)
                        if selected_tsne_idx in pareto_indices:
                            is_pareto = True
                            st.success("🏆 帕累托前沿解")
                        else:
                            st.info("被支配解")

                        st.metric("🎯 精度", f"{selected_arch.accuracy:.4f}")
                        st.metric("📦 参数量", f"{selected_arch.params/1e6:.2f} M")
                        st.metric("⏱️ 延迟", f"{selected_arch.latency:.3f} ms")
                    else:
                        st.info("👆 点击左侧 t-SNE 散点图中的点查看架构详情")

        with tab5:
            st.subheader("🧠 代理模型性能预测置信度分析")

            if len(result.all_evaluated) == 0:
                st.warning("⚠️ 实验尚未完成或还未评估任何架构，暂无预测分析数据。")
                st.info("💡 提示：运行实验后，代理模型会根据评估的架构自动训练，训练完成后可在此查看预测分析。")
            elif exp.surrogate is None:
                st.warning("⚠️ 该实验未启用代理模型，无法进行预测分析。")
                st.info("💡 提示：创建实验时请勾选「启用代理模型预筛」选项。")
            elif not exp.surrogate.trained:
                n_eval = len(result.all_evaluated)
                min_samples = exp.config.surrogate_min_samples if hasattr(exp.config, 'surrogate_min_samples') else 50
                st.info(f"🤖 代理模型尚未训练。当前已评估 {n_eval} 个架构，需要至少 {min_samples} 个样本才能训练。")
                if n_eval < min_samples:
                    st.progress(n_eval / min_samples)
                    st.caption(f"代理模型训练进度: {n_eval}/{min_samples} ({n_eval/min_samples*100:.1f}%)")
                else:
                    st.caption("样本已满足要求，继续运行实验时代理模型将自动训练。")
            elif len(result.all_evaluated) < 10:
                st.warning(f"⚠️ 已评估架构数量过少（{len(result.all_evaluated)}个），无法进行有效的训练/验证集分割。")
                st.info("💡 提示：需要至少10个已评估架构才能计算预测性能指标（R²、MAPE等）。")
            else:
                with st.spinner("正在计算预测分析指标..."):
                    val_metrics = exp.surrogate.compute_validation_metrics(result.all_evaluated, train_ratio=0.8)

                    st.markdown("### 📊 验证集预测性能指标")
                    metric_cols = st.columns(3)
                    target_names = ['精度', '参数量', '延迟']
                    for i, name in enumerate(target_names):
                        with metric_cols[i]:
                            st.metric(
                                f"{name} R²",
                                f"{val_metrics['r2'][i]:.4f}",
                                delta=f"MAPE: {val_metrics['mape'][i]:.2f}%"
                            )

                    st.markdown("### 📈 预测值 vs 真实值散点图")
                    if len(val_metrics['val_true']) > 0:
                        scatter_cols = st.columns(3)
                        dim_info = [
                            (0, '精度', 'accuracy'),
                            (1, '参数量', 'params'),
                            (2, '延迟 (ms)', 'latency')
                        ]
                        for idx, (dim, name, _) in enumerate(dim_info):
                            with scatter_cols[idx]:
                                true_vals = val_metrics['val_true'][:, dim]
                                pred_vals = val_metrics['val_pred'][:, dim]
                                if dim == 1:
                                    true_vals = true_vals / 1e6
                                    pred_vals = pred_vals / 1e6
                                    name_display = f'{name} (M)'
                                else:
                                    name_display = name
                                fig_scatter = plot_prediction_scatter(
                                    true_vals, pred_vals,
                                    dim_name=name_display,
                                    title=f'{name_display} 预测 vs 真实'
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("验证集样本不足，无法绘制散点图。")

                    st.markdown("### 🎯 带不确定度的帕累托前沿")
                    st.caption("点大小代表预测不确定度（大点=不确定度高，小点=预测可靠）")

                    last_snapshot = result.generations[-1]
                    last_pop = last_snapshot.population
                    last_points = last_snapshot.get_fitness_matrix()

                    if exp.surrogate.trained:
                        _, uncertainties = exp.surrogate.predict_with_uncertainty(last_pop, n_bootstrap=10)

                        pareto_unc_cols = st.columns(2)
                        with pareto_unc_cols[0]:
                            fig_unc1 = plot_pareto_with_uncertainty(
                                last_points, uncertainties, MAXIMIZE,
                                x_dim=1, y_dim=0,
                                x_label='参数量', y_label='精度',
                                title='精度 vs 参数量 (大小=不确定度)'
                            )
                            st.plotly_chart(fig_unc1, use_container_width=True)
                        with pareto_unc_cols[1]:
                            fig_unc2 = plot_pareto_with_uncertainty(
                                last_points, uncertainties, MAXIMIZE,
                                x_dim=2, y_dim=0,
                                x_label='延迟 (ms)', y_label='精度',
                                title='精度 vs 延迟 (大小=不确定度)'
                            )
                            st.plotly_chart(fig_unc2, use_container_width=True)

                    st.markdown("### 📉 代理模型学习曲线")
                    with st.spinner("正在计算学习曲线（可能需要一些时间）..."):
                        learning_data = exp.surrogate.compute_learning_curve(
                            result.all_evaluated,
                            train_ratio=0.8, min_samples=10, step=10
                        )
                        if len(learning_data['train_sizes']) > 0:
                            fig_lc = plot_surrogate_learning_curve(
                                learning_data['train_sizes'],
                                learning_data['r2_scores'],
                                title='代理模型学习曲线 (验证集 R² 随训练样本数变化)'
                            )
                            st.plotly_chart(fig_lc, use_container_width=True)
                        else:
                            st.info("样本数量不足，无法绘制学习曲线。")

                    st.markdown("### 🔄 在线校准历史")
                    st.caption("记录每次增量训练后的验证集R²变化，与学习曲线形成对比")

                    calib_history = exp.surrogate.get_calibration_history()

                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("增量训练次数", calib_history['incremental_count'])
                    with metric_cols[1]:
                        st.metric("回滚次数", calib_history['rollback_count'])
                    with metric_cols[2]:
                        st.metric("全量重训次数", calib_history['full_retrain_count'])
                    with metric_cols[3]:
                        if len(calib_history['incremental_r2_history']) > 0:
                            current_r2 = calib_history['incremental_r2_history'][-1]
                            st.metric("当前验证R²", f"{current_r2:.4f}")
                        else:
                            st.metric("当前验证R²", "—")

                    if len(calib_history['incremental_r2_history']) > 0 or len(learning_data['train_sizes']) > 0:
                        fig_calib = plot_calibration_history(
                            calib_history['incremental_r2_history'],
                            learning_data['train_sizes'] if len(learning_data['train_sizes']) > 0 else None,
                            learning_data['r2_scores'] if len(learning_data['train_sizes']) > 0 else None,
                            title='在线校准历史 vs 学习曲线'
                        )
                        st.plotly_chart(fig_calib, use_container_width=True)
                    else:
                        st.info("暂无在线校准数据。运行实验后，每代新增的评估结果会触发增量校准。")

                    if calib_history['calibration_events']:
                        with st.expander("📋 校准事件详情"):
                            event_data = []
                            for i, event in enumerate(calib_history['calibration_events']):
                                event_type = '增量训练' if event['type'] == 'incremental' else '回滚并重训'
                                event_data.append({
                                    '序号': i + 1,
                                    '类型': event_type,
                                    '增量次数': event.get('incremental_count', '-'),
                                    '原R²': f"{event.get('prev_r2', 0):.4f}" if event.get('prev_r2') is not None else '-',
                                    '新R²': f"{event.get('new_r2', 0):.4f}" if event.get('new_r2') is not None else '-',
                                    '说明': event.get('reason', '')
                                })
                            st.dataframe(pd.DataFrame(event_data), use_container_width=True, hide_index=True)

        with tab6:
            st.subheader("📊 搜索效率统计")

            if len(result.generations) == 0:
                st.info("暂无效率统计数据。")
            else:
                gens_list = list(range(len(result.generations)))
                actually_eval_list = [gen.actually_evaluated for gen in result.generations]
                skipped_list = [gen.surrogate_skipped for gen in result.generations]

                total_eval_from_snap = sum(actually_eval_list)
                total_skip_from_snap = sum(skipped_list)
                total_eval_reliable = len(result.all_evaluated)
                pop_size_cfg = config.pop_size

                if total_eval_from_snap == 0 and total_eval_reliable > 0:
                    actually_eval_list = [pop_size_cfg] * len(gens_list)
                    skipped_list = [0] * len(gens_list)
                    total_eval_from_snap = total_eval_reliable
                    total_skip_from_snap = 0
                    st.caption("ℹ️ 该实验为历史数据，效率统计中每代评估数按种群大小估算。")

                total_eval = max(total_eval_from_snap, total_eval_reliable)
                total_skip = total_skip_from_snap
                total_candidates = total_eval + total_skip
                savings_percent = (total_skip / total_candidates * 100) if total_candidates > 0 else 0.0

                all_points = result.get_all_points()
                if len(all_points) > 0:
                    try:
                        from src.metrics import get_reference_point as grp, hypervolume as hv_func
                        ref_pt = grp(all_points, MAXIMIZE)
                        global_hv = hv_func(all_points, ref_pt, MAXIMIZE)
                    except:
                        global_hv = result.hypervolume_history[-1] if result.hypervolume_history else 0.0
                else:
                    global_hv = result.hypervolume_history[-1] if result.hypervolume_history else 0.0

                hv_per_eval = (global_hv / total_eval) if total_eval > 0 else 0.0

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("总候选架构数", total_candidates)
                with metric_cols[1]:
                    st.metric("实际评估数", total_eval, delta=f"跳过 {total_skip}")
                with metric_cols[2]:
                    st.metric("累计节省评估", f"{savings_percent:.1f}%",
                             delta=f"节省 {total_skip} 次评估")
                with metric_cols[3]:
                    if hv_per_eval == 0.0:
                        display_val = "—"
                    elif hv_per_eval < 1e-4:
                        display_val = f"{hv_per_eval:.2e}"
                    else:
                        display_val = f"{hv_per_eval:.6f}"
                    st.metric("搜索效率 (HV/评估)", display_val,
                             help="全局超体积 / 总评估次数 = 每次评估的平均HV贡献")
                    if hv_per_eval == 0.0 and global_hv > 0 and total_eval > 0:
                        st.caption(f"DEBUG: HV={global_hv:.6f}, Eval={total_eval}, 比值={global_hv/total_eval}")

                st.markdown("### 📊 每代评估架构数 vs 代理跳过数")
                fig_eff = plot_eval_efficiency_bar(
                    gens_list, actually_eval_list, skipped_list,
                    title='每代实际评估数 vs 代理预筛跳过数'
                )
                st.plotly_chart(fig_eff, use_container_width=True)

                st.markdown("### 🥧 各评估策略使用时长占比")
                strategy_durations = {}
                for gen in result.generations:
                    s = gen.eval_strategy
                    if s not in strategy_durations:
                        strategy_durations[s] = 0.0
                    strategy_durations[s] += gen.eval_duration

                if len(strategy_durations) > 0:
                    fig_pie = plot_strategy_duration_pie(
                        strategy_durations,
                        title='各评估策略累计使用时长占比'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab7:
            st.subheader("📜 策略切换日志")

            if not result.strategy_switch_logs:
                st.info("📭 本次搜索未记录策略切换（自适应调度可能未启用）。")
            else:
                st.caption(f"共发生 {len(result.strategy_switch_logs)} 次策略切换")

                switch_data = []
                for log in result.strategy_switch_logs:
                    old_display = EVAL_STRATEGIES.get(log.old_strategy, log.old_strategy)
                    new_display = EVAL_STRATEGIES.get(log.new_strategy, log.new_strategy)
                    switch_data.append({
                        '时间': log.timestamp,
                        '代数': f"第 {log.generation} 代",
                        '原策略': old_display,
                        '新策略': new_display,
                        '切换原因': log.reason
                    })
                switch_df = pd.DataFrame(switch_data)
                st.dataframe(switch_df, use_container_width=True, hide_index=True)

            if exp.surrogate is not None and exp.surrogate.trained:
                calib_history = exp.surrogate.get_calibration_history()
                rollback_events = [e for e in calib_history['calibration_events']
                                   if e['type'] == 'rollback_and_retrain']

                if rollback_events:
                    st.markdown("---")
                    st.subheader("⚠️ 代理模型回滚与重训事件")
                    st.warning(f"共发生 {len(rollback_events)} 次代理模型回滚 + 全量重训")

                    rollback_data = []
                    for i, event in enumerate(rollback_events):
                        rollback_data.append({
                            '序号': i + 1,
                            '增量训练次数': event.get('incremental_count', '-'),
                            '原R²': f"{event.get('prev_r2', 0):.4f}",
                            '新R²': f"{event.get('new_r2', 0):.4f}",
                            '下降幅度': f"{(event.get('prev_r2', 0) - event.get('new_r2', 0)) / abs(event.get('prev_r2', 1)) * 100:.1f}%"
                            if event.get('prev_r2', 0) > 0 else '-',
                            '原因': event.get('reason', '')
                        })
                    st.dataframe(pd.DataFrame(rollback_data), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("📋 每代参数快照")
            if len(result.generations) > 0:
                snapshot_data = []
                for gen in result.generations:
                    strat_display = EVAL_STRATEGIES.get(gen.eval_strategy, gen.eval_strategy)
                    snapshot_data.append({
                        '代数': gen.generation,
                        '评估策略': strat_display.split(' ')[0],
                        '变异率': f"{gen.mutation_rate:.3f}",
                        '交叉率': f"{gen.crossover_rate:.3f}",
                        '多样性': f"{gen.diversity:.3f}",
                        '实际评估': gen.actually_evaluated,
                        '代理跳过': gen.surrogate_skipped,
                        '超体积': f"{gen.hypervolume:.4f}"
                    })
                snapshot_df = pd.DataFrame(snapshot_data)
                st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

        with tab8:
            st.subheader("⚔️ 实时对比分析")
            st.caption("选择另一个已完成的实验进行逐代对比分析")

            exp_list = st.session_state.exp_manager.list_experiments()
            completed_exps = [name for name in exp_list
                              if name != selected_exp
                              and st.session_state.exp_manager.get_experiment(name)
                              and st.session_state.exp_manager.get_experiment(name).result.completed]

            if not completed_exps:
                st.info("📭 没有其他已完成的实验可供对比。请先运行更多实验。")
            else:
                compare_exp_name = st.selectbox(
                    "选择对比实验",
                    completed_exps,
                    key="compare_exp_select"
                )

                if compare_exp_name:
                    exp2 = st.session_state.exp_manager.get_experiment(compare_exp_name)
                    if exp2:
                        result2 = exp2.result
                        config2 = exp2.config

                        st.markdown("---")
                        st.subheader("📈 超体积曲线对比")

                        hv_fig = plot_dual_hypervolume_curves(
                            result.hypervolume_history,
                            result2.hypervolume_history,
                            exp1_name=config.name,
                            exp2_name=config2.name,
                            title='超体积曲线对比'
                        )
                        st.plotly_chart(hv_fig, use_container_width=True)

                        st.markdown("---")
                        st.subheader("📊 策略切换时间线对比")

                        exp1_gens = list(range(len(result.generations)))
                        exp1_strategies = [gen.eval_strategy for gen in result.generations]
                        exp2_gens = list(range(len(result2.generations)))
                        exp2_strategies = [gen.eval_strategy for gen in result2.generations]

                        timeline_fig = plot_dual_strategy_timeline(
                            exp1_gens, exp1_strategies,
                            exp2_gens, exp2_strategies,
                            exp1_name=config.name,
                            exp2_name=config2.name,
                            title='策略切换时间线对比'
                        )
                        st.plotly_chart(timeline_fig, use_container_width=True)

                        st.markdown("---")
                        st.subheader("🎯 最终帕累托前沿对比")

                        exp1_points = result.get_all_points()
                        exp2_points = result2.get_all_points()

                        pareto_fig = plot_dual_pareto_scatter(
                            exp1_points, exp2_points,
                            exp1_name=config.name,
                            exp2_name=config2.name,
                            maximize=MAXIMIZE,
                            x_dim=1, y_dim=0,
                            x_label='参数量', y_label='精度',
                            title='精度 vs 参数量 帕累托前沿对比'
                        )
                        st.plotly_chart(pareto_fig, use_container_width=True)

                        st.markdown("---")
                        st.subheader("📋 关键差异汇总")

                        total_eval1 = len(result.all_evaluated)
                        total_eval2 = len(result2.all_evaluated)
                        eval_diff = total_eval1 - total_eval2

                        all_points1 = result.get_all_points()
                        all_points2 = result2.get_all_points()
                        ref_point1 = get_reference_point(all_points1, MAXIMIZE)
                        ref_point2 = get_reference_point(all_points2, MAXIMIZE)
                        final_hv1 = hypervolume(all_points1, ref_point1, MAXIMIZE)
                        final_hv2 = hypervolume(all_points2, ref_point2, MAXIMIZE)
                        hv_diff = final_hv1 - final_hv2

                        savings1 = 0.0
                        savings2 = 0.0
                        if result.generations:
                            total_cand1 = sum(g.actually_evaluated + g.surrogate_skipped for g in result.generations)
                            total_skip1 = sum(g.surrogate_skipped for g in result.generations)
                            savings1 = (total_skip1 / total_cand1 * 100) if total_cand1 > 0 else 0.0
                        if result2.generations:
                            total_cand2 = sum(g.actually_evaluated + g.surrogate_skipped for g in result2.generations)
                            total_skip2 = sum(g.surrogate_skipped for g in result2.generations)
                            savings2 = (total_skip2 / total_cand2 * 100) if total_cand2 > 0 else 0.0
                        savings_diff = savings1 - savings2

                        conv_gen1, _, _ = detect_convergence(
                            result.hypervolume_history, threshold=0.01, window_size=5
                        )
                        conv_gen2, _, _ = detect_convergence(
                            result2.hypervolume_history, threshold=0.01, window_size=5
                        )
                        conv_diff = None
                        if conv_gen1 is not None and conv_gen2 is not None:
                            conv_diff = conv_gen1 - conv_gen2

                        diff_data = [
                            {
                                '指标': '总评估次数',
                                config.name: total_eval1,
                                config2.name: total_eval2,
                                '差异': f"{eval_diff:+d}"
                            },
                            {
                                '指标': '代理节省率',
                                config.name: f"{savings1:.1f}%",
                                config2.name: f"{savings2:.1f}%",
                                '差异': f"{savings_diff:+.1f}%"
                            },
                            {
                                '指标': '最终超体积(HV)',
                                config.name: f"{final_hv1:.4f}",
                                config2.name: f"{final_hv2:.4f}",
                                '差异': f"{hv_diff:+.4f}"
                            },
                            {
                                '指标': '收敛代数',
                                config.name: f"第{conv_gen1}代" if conv_gen1 is not None else "未收敛",
                                config2.name: f"第{conv_gen2}代" if conv_gen2 is not None else "未收敛",
                                '差异': f"{conv_diff:+d}代" if conv_diff is not None else "—"
                            },
                        ]
                        diff_df = pd.DataFrame(diff_data)
                        st.dataframe(diff_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        col_export, col_import, col_baseline = st.columns(3)

        with col_export:
            if st.button("📤 导出评估结果 (CSV)", use_container_width=True):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    exp.export_to_csv(f.name)
                    with open(f.name, 'rb') as file:
                        st.download_button(
                            label="⬇️ 下载CSV",
                            data=file,
                            file_name=f"{config.name}_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        with col_import:
            uploaded_file = st.file_uploader("📥 导入外部结果", type=['csv'])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write(uploaded_file.getvalue().decode())
                    f.flush()
                    exp.import_external_results(f.name)
                    st.success(f"✅ 已导入 {pd.read_csv(uploaded_file).shape[0]} 个架构")

        with col_baseline:
            if st.button("🎲 生成随机搜索基线", use_container_width=True):
                with st.spinner("运行随机搜索基线..."):
                    try:
                        baseline_exp = st.session_state.exp_manager.run_random_search_baseline(config)
                        st.success(f"✅ 随机搜索基线完成: {baseline_exp.config.name}")
                    except Exception as e:
                        st.error(f"❌ 错误: {e}")


def comparison_page():
    """对比分析页面"""
    st.title("📈 多实验对比分析")
    st.markdown("---")

    exp_list = st.session_state.exp_manager.list_experiments()

    if len(exp_list) < 2:
        st.info("⚠️ 需要至少2个实验才能进行对比。请先创建多个实验。")
        return

    selected_exps = st.multiselect("选择要对比的实验", exp_list, default=exp_list[:2])

    if len(selected_exps) < 2:
        st.warning("⚠️ 请至少选择2个实验进行对比")
        return

    pareto_data = st.session_state.exp_manager.get_multi_experiment_pareto(selected_exps)
    hv_data = st.session_state.exp_manager.get_multi_experiment_hypervolumes(selected_exps)

    tab1, tab2, tab3 = st.tabs(["🎯 帕累托前沿对比", "📊 超体积曲线对比", "📋 详细数据对比"])

    with tab1:
        pareto_col1, pareto_col2 = st.columns(2)
        with pareto_col1:
            fig1 = plot_multi_experiment_pareto(
                pareto_data, MAXIMIZE,
                x_dim=1, y_dim=0,
                x_label='参数量', y_label='精度',
                title='精度 vs 参数量'
            )
            st.plotly_chart(fig1, use_container_width=True)

        with pareto_col2:
            fig2 = plot_multi_experiment_pareto(
                pareto_data, MAXIMIZE,
                x_dim=2, y_dim=0,
                x_label='延迟 (ms)', y_label='精度',
                title='精度 vs 延迟'
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig_hv = plot_multi_hypervolume_curves(
            hv_data,
            title='超体积曲线对比'
        )
        st.plotly_chart(fig_hv, use_container_width=True)

    with tab3:
        comparison_data = []
        for exp_name in selected_exps:
            exp = st.session_state.exp_manager.get_experiment(exp_name)
            if exp and exp.result.generations:
                last_gen = exp.result.generations[-1]
                all_points = exp.result.get_all_points()
                ref_point = get_reference_point(all_points, MAXIMIZE)
                total_hv = hypervolume(all_points, ref_point, MAXIMIZE)
                pareto_pts = get_pareto_front_indices(all_points, MAXIMIZE)

                comparison_data.append({
                    '实验名称': exp_name,
                    '算法': 'NSGA-II' if exp.config.algorithm == 'nsga2' else '随机搜索',
                    '总评估数': len(exp.result.all_evaluated),
                    '帕累托解数量': len(pareto_pts),
                    '最终超体积': f"{exp.result.hypervolume_history[-1]:.4f}",
                    '总体超体积': f"{total_hv:.4f}",
                    '最高精度': f"{np.max(all_points[:, 0]):.4f}",
                    '最小参数量': f"{np.min(all_points[:, 1])/1e6:.2f}M",
                    '最小延迟': f"{np.min(all_points[:, 2]):.3f}ms"
                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

            st.subheader("🏆 超体积对比")
            hv_values = [float(d['总体超体积']) for d in comparison_data]
            hv_fig = go.Figure()
            hv_fig.add_trace(go.Bar(
                x=[d['实验名称'] for d in comparison_data],
                y=hv_values,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(comparison_data)]
            ))
            hv_fig.update_layout(
                title='总体超体积对比 (越高越好)',
                yaxis_title='超体积',
                template='plotly_white'
            )
            st.plotly_chart(hv_fig, use_container_width=True)


def legend_page():
    """操作图例页面"""
    st.title("⚙️ 操作类型说明")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_legend = plot_operations_legend()
        st.plotly_chart(fig_legend, use_container_width=True)

    with col2:
        st.subheader("📋 操作详细说明")

        op_info = [
            ('conv3x3', '3x3 标准卷积', '最常用的卷积操作，提取局部特征', 0.12),
            ('conv5x5', '5x5 标准卷积', '更大感受野，提取更全局特征', 0.25),
            ('dil_conv3x3', '3x3 扩张卷积', '扩张率=2，扩大感受野不增加参数量', 0.15),
            ('max_pool3x3', '3x3 最大池化', '降采样，保留最显著特征', 0.02),
            ('avg_pool3x3', '3x3 平均池化', '降采样，平滑特征', 0.02),
            ('skip_connect', '恒等连接', '残差连接，缓解梯度消失', 0.001),
            ('zero', '无连接', '丢弃该路径', 0.0),
        ]

        for op_name, display_name, desc, latency in op_info:
            color = OP_COLORS.get(op_name, '#888888')
            st.markdown(
                f"<div style='padding: 10px; border-left: 4px solid {color}; margin-bottom: 10px;'>"
                f"<b style='color: {color};'>{display_name}</b> ({op_name})<br>"
                f"<small>{desc}</small><br>"
                f"<small>延迟估算: {latency} ms</small>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.subheader("📐 架构编码说明")
        st.markdown("""
        **邻接矩阵**: N×N布尔矩阵，表示节点间的连接关系（上三角矩阵）
        - 节点0: 输入1（上一个Cell的输出）
        - 节点1: 输入2（上上个Cell的输出）
        - 节点2~N-2: 中间节点
        - 节点N-1: 输出节点

        **操作列表**: 每条存在的边对应一个操作索引，按行优先顺序排列

        **完整网络结构**:
        - Stem: 3×3卷积，通道数×3
        - Normal Cell: 保持特征图尺寸
        - Reduction Cell: 在1/3和2/3位置，降采样2×
        - Global Pool + Linear: 分类器
        """)


def main():
    """主函数"""
    init_session_state()

    page = sidebar()

    if "首页" in page:
        home_page()
    elif "创建实验" in page:
        create_experiment_page()
    elif "查看实验" in page:
        view_experiment_page()
    elif "对比分析" in page:
        comparison_page()
    elif "操作图例" in page:
        legend_page()


if __name__ == "__main__":
    main()
