"""
自适应调度模块功能测试脚本
验证新开发的自适应调度、预测分析、效率统计功能
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🧠 自适应调度模块 - 专项功能测试")
print("=" * 70)


def test_adaptive_core():
    """测试自适应调度核心模块"""
    print("\n1️⃣  测试自适应调度核心模块 (adaptive.py)...")
    from src.adaptive import (
        AdaptiveScheduler, compute_population_diversity,
        StrategySwitchLog, GenerationEfficiencyStats, EVAL_STRATEGY_ORDER
    )
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']

    scheduler = AdaptiveScheduler(base_mutation_rate=0.1, base_crossover_rate=0.9)
    print(f"  ✓ 创建AdaptiveScheduler成功")
    print(f"  ✓ 初始评估策略: {scheduler.state.current_eval_strategy}")
    print(f"  ✓ 初始变异率: {scheduler.state.current_mutation_rate}")
    print(f"  ✓ 初始交叉率: {scheduler.state.current_crossover_rate}")

    population = [Architecture(num_nodes=6, enabled_ops=enabled_ops) for _ in range(10)]
    for arch in population:
        arch.accuracy = np.random.uniform(0.5, 0.9)
        arch.params = np.random.uniform(1e5, 1e7)
        arch.latency = np.random.uniform(0.5, 5.0)

    diversity = compute_population_diversity(population)
    print(f"  ✓ 种群多样性计算: {diversity:.4f} (0~1)")
    assert 0.0 <= diversity <= 1.0, "多样性应在0~1之间"

    new_mut, new_cross = scheduler.update_evolution_params(population)
    print(f"  ✓ 进化参数调整: 变异率 {new_mut:.4f}, 交叉率 {new_cross:.4f}")
    print(f"  ✓ 更新后多样性值: {scheduler.state.current_diversity:.4f}")

    hv_history = [0.1, 0.2, 0.3, 0.31, 0.32, 0.325]
    scheduler.update_eval_strategy(3, population, hv_history, surrogate_ready=False)
    print(f"  ✓ 前5代评估策略应为synflow: {scheduler.state.current_eval_strategy}")
    assert scheduler.state.current_eval_strategy == 'synflow', "前5代应使用synflow"

    hv_history_long = hv_history + [0.326, 0.327, 0.328, 0.329]
    scheduler.update_eval_strategy(8, population, hv_history_long, surrogate_ready=True)
    print(f"  ✓ 第8代策略: {scheduler.state.current_eval_strategy}")

    gen_stats = GenerationEfficiencyStats(
        generation=0, total_candidates=10, actually_evaluated=5,
        surrogate_skipped=5, eval_strategy='fast', eval_duration=0.5
    )
    scheduler.record_efficiency_stats(gen_stats)
    summary = scheduler.get_efficiency_summary()
    print(f"  ✓ 效率统计汇总: 评估{summary['total_evaluated']}, 跳过{summary['total_skipped']}, 节省{summary['savings_percent']:.1f}%")
    assert summary['total_candidates'] == 10

    print("  ✅ 自适应核心模块测试通过")
    return True


def test_experiment_integration():
    """测试实验与自适应调度的集成"""
    print("\n2️⃣  测试实验与自适应调度集成 (experiment.py)...")
    from src.experiment import (
        ExperimentConfig, Experiment, ExperimentManager,
        GenerationSnapshot, ExperimentResult
    )
    from src.adaptive import StrategySwitchLog
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    config = ExperimentConfig(
        name='adaptive_test_exp',
        algorithm='nsga2',
        num_nodes=6,
        enabled_ops=['conv3x3', 'conv5x5', 'skip_connect'],
        num_cells=4,
        init_channels=8,
        pop_size=8,
        num_generations=3,
        mutation_rate=0.1,
        crossover_rate=0.9,
        eval_strategy='fast',
        use_surrogate=True,
        surrogate_min_samples=5,
        surrogate_percentile=50.0,
        use_adaptive_scheduling=True,
        device='cpu'
    )
    print(f"  ✓ 创建带自适应调度的实验配置")
    assert config.use_adaptive_scheduling == True

    manager = ExperimentManager(experiments_dir=temp_dir)
    exp = manager.create_experiment(config)
    print(f"  ✓ 创建实验，自适应调度器初始化: {exp.adaptive_scheduler is not None}")
    assert exp.adaptive_scheduler is not None

    snapshot = GenerationSnapshot(
        generation=0, population=[],
        hypervolume=0.5, avg_accuracy=0.8, avg_params=1e6, avg_latency=2.0,
        eval_strategy='synflow', mutation_rate=0.15, crossover_rate=0.75,
        diversity=0.35, actually_evaluated=8, surrogate_skipped=0,
        eval_duration=0.3
    )
    print(f"  ✓ 创建扩展GenerationSnapshot: 策略={snapshot.eval_strategy}, "
          f"变异率={snapshot.mutation_rate}, 多样性={snapshot.diversity}")

    result = ExperimentResult(config=config)
    log = StrategySwitchLog(generation=5, old_strategy='synflow', new_strategy='fast',
                            reason="测试切换", timestamp="2024-01-01 00:00:00")
    result.strategy_switch_logs.append(log)
    print(f"  ✓ 策略切换日志记录: 第{log.generation}代 {log.old_strategy}→{log.new_strategy}")
    assert len(result.strategy_switch_logs) == 1

    shutil.rmtree(temp_dir)
    print("  ✅ 实验集成测试通过")
    return True


def test_surrogate_extensions():
    """测试代理模型扩展功能"""
    print("\n3️⃣  测试代理模型扩展功能 (surrogate.py)...")
    from src.surrogate import SurrogateModel
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    input_dim = 2 * (6 * 6 + 6 * 5 // 2)

    surrogate = SurrogateModel(
        input_dim=input_dim, hidden_dim=64,
        min_train_samples=10, epochs=5
    )

    evaluated_archs = []
    for i in range(30):
        arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)
        arch.accuracy = 0.5 + 0.4 * (i / 30) + np.random.normal(0, 0.02)
        arch.params = 1e6 + i * 1e5 + np.random.normal(0, 1e4)
        arch.latency = 1.0 + i * 0.1 + np.random.normal(0, 0.05)
        evaluated_archs.append(arch)
    print(f"  ✓ 生成{len(evaluated_archs)}个时序评估架构数据")

    surrogate.train(evaluated_archs)
    assert surrogate.trained, "代理模型应训练成功"
    print(f"  ✓ 代理模型训练成功")

    train_archs, val_archs = surrogate.split_train_val_by_time(evaluated_archs, train_ratio=0.8)
    print(f"  ✓ 时间序列分割: 训练集{len(train_archs)}, 验证集{len(val_archs)}")
    assert len(train_archs) == 24 and len(val_archs) == 6

    val_metrics = surrogate.compute_validation_metrics(evaluated_archs, train_ratio=0.8)
    print(f"  ✓ 验证集R²分数: 精度{val_metrics['r2'][0]:.4f}, "
          f"参数量{val_metrics['r2'][1]:.4f}, 延迟{val_metrics['r2'][2]:.4f}")
    print(f"  ✓ 验证集MAPE: 精度{val_metrics['mape'][0]:.2f}%, "
          f"参数量{val_metrics['mape'][1]:.2f}%, 延迟{val_metrics['mape'][2]:.2f}%")
    assert val_metrics['val_pred'].shape[0] == len(val_archs)

    test_archs = evaluated_archs[:5]
    preds, uncs = surrogate.predict_with_uncertainty(test_archs, n_bootstrap=5)
    print(f"  ✓ Bootstrap不确定度估计: 预测形状{preds.shape}, 不确定度形状{uncs.shape}")
    print(f"  ✓ 平均不确定度: {np.mean(uncs):.6f}")
    assert preds.shape == (5, 3) and uncs.shape == (5, 3)

    learning_data = surrogate.compute_learning_curve(
        evaluated_archs, train_ratio=0.8, min_samples=10, step=5
    )
    print(f"  ✓ 学习曲线计算: {len(learning_data['train_sizes'])}个数据点, "
          f"训练样本数范围{min(learning_data['train_sizes'])}~{max(learning_data['train_sizes'])}")
    assert len(learning_data['train_sizes']) > 0

    print("  ✅ 代理模型扩展功能测试通过")
    return True


def test_visualization_extensions():
    """测试可视化扩展功能"""
    print("\n4️⃣  测试可视化扩展功能 (visualization.py)...")
    from src.visualization import (
        plot_prediction_scatter, plot_pareto_with_uncertainty,
        plot_surrogate_learning_curve, plot_eval_efficiency_bar,
        plot_strategy_duration_pie, plot_param_diversity_curve
    )

    true_vals = np.random.uniform(0.5, 0.9, 50)
    pred_vals = true_vals + np.random.normal(0, 0.03, 50)
    fig1 = plot_prediction_scatter(true_vals, pred_vals, dim_name='精度',
                                    title='测试预测散点图')
    print(f"  ✓ 预测vs真实散点图创建成功")

    points = np.array([
        [0.9, 1e6, 1.0], [0.8, 0.5e6, 0.5],
        [0.95, 2e6, 2.0], [0.85, 0.8e6, 0.8],
        [0.88, 1.2e6, 1.2]
    ])
    uncertainties = np.random.uniform(0.01, 0.1, (5, 3))
    maximize = [True, False, False]
    fig2 = plot_pareto_with_uncertainty(points, uncertainties, maximize,
                                         title='测试带不确定度帕累托图')
    print(f"  ✓ 带不确定度帕累托图创建成功")

    train_sizes = [10, 15, 20, 25, 30]
    r2_scores = np.array([[0.1, 0.2, 0.15], [0.3, 0.4, 0.35],
                           [0.5, 0.6, 0.55], [0.7, 0.75, 0.7],
                           [0.8, 0.85, 0.82]])
    fig3 = plot_surrogate_learning_curve(train_sizes, r2_scores,
                                          title='测试学习曲线')
    print(f"  ✓ 代理学习曲线图创建成功")

    gens = list(range(10))
    actually_eval = [20, 18, 15, 12, 10, 8, 10, 12, 15, 18]
    skipped = [0, 2, 5, 8, 10, 12, 10, 8, 5, 2]
    fig4 = plot_eval_efficiency_bar(gens, actually_eval, skipped,
                                     title='测试效率柱状图')
    print(f"  ✓ 评估效率堆叠柱状图创建成功")

    strategy_durations = {'fast': 5.0, 'synflow': 3.0, 'naswot': 2.0}
    fig5 = plot_strategy_duration_pie(strategy_durations,
                                       title='测试策略时长饼图')
    print(f"  ✓ 策略时长饼图创建成功")

    diversities = [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.32, 0.4, 0.5, 0.6]
    mutation_rates = [0.1, 0.1, 0.1, 0.12, 0.15, 0.15, 0.13, 0.1, 0.08, 0.07]
    crossover_rates = [0.9, 0.9, 0.9, 0.85, 0.75, 0.7, 0.78, 0.88, 0.95, 1.0]
    fig6 = plot_param_diversity_curve(gens, diversities, mutation_rates, crossover_rates,
                                       title='测试多样性参数曲线')
    print(f"  ✓ 多样性与进化参数双Y轴曲线创建成功")

    print("  ✅ 可视化扩展功能测试通过")
    return True


def test_full_integration():
    """测试完整流程集成（简化版，使用模拟数据）"""
    print("\n5️⃣  测试完整流程集成 (使用模拟数据)...")
    from src.adaptive import AdaptiveScheduler, GenerationEfficiencyStats
    from src.cell import Architecture
    from src.experiment import GenerationSnapshot
    from src.surrogate import SurrogateModel

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    scheduler = AdaptiveScheduler(base_mutation_rate=0.1, base_crossover_rate=0.9)

    hv_history = []
    for gen in range(10):
        population = [Architecture(num_nodes=6, enabled_ops=enabled_ops) for _ in range(20)]
        for arch in population:
            arch.accuracy = 0.5 + 0.03 * gen + np.random.normal(0, 0.02)
            arch.params = 1e6 - gen * 5e4 + np.random.normal(0, 1e4)
            arch.latency = 3.0 - gen * 0.2 + np.random.normal(0, 0.1)

        new_mut, new_cross = scheduler.update_evolution_params(population)

        surrogate_ready = gen >= 3
        hv_history.append(0.1 + gen * 0.05 if gen < 6 else 0.4 + (gen - 6) * 0.005)
        scheduler.update_eval_strategy(gen, population, hv_history, surrogate_ready)

        gen_stats = GenerationEfficiencyStats(
            generation=gen, total_candidates=20,
            actually_evaluated=max(5, 20 - gen * 2),
            surrogate_skipped=min(15, gen * 2),
            eval_strategy=scheduler.state.current_eval_strategy,
            eval_duration=0.1 + np.random.random() * 0.2
        )
        scheduler.record_efficiency_stats(gen_stats)

        print(f"  Gen {gen:2d}: 策略={scheduler.state.current_eval_strategy:8s}, "
              f"HV={hv_history[-1]:.4f}, 多样性={scheduler.state.current_diversity:.3f}, "
              f"变异率={new_mut:.3f}, 交叉率={new_cross:.3f}, "
              f"评估={gen_stats.actually_evaluated}, 跳过={gen_stats.surrogate_skipped}")

    summary = scheduler.get_efficiency_summary()
    print(f"\n  📊 最终效率汇总:")
    print(f"    - 总候选: {summary['total_candidates']}")
    print(f"    - 实际评估: {summary['total_evaluated']}")
    print(f"    - 代理跳过: {summary['total_skipped']}")
    print(f"    - 节省评估: {summary['savings_percent']:.1f}%")
    print(f"    - 策略时长占比: {summary['strategy_duration_percent']}")

    print(f"\n  📜 策略切换日志 ({len(scheduler.state.strategy_switch_logs)}次):")
    for log in scheduler.state.strategy_switch_logs:
        print(f"    - 第{log.generation}代: {log.old_strategy} → {log.new_strategy}")
        print(f"      原因: {log.reason}")

    print("  ✅ 完整流程集成测试通过")
    return True


def main():
    """运行所有自适应调度测试"""
    tests = [
        test_adaptive_core,
        test_experiment_integration,
        test_surrogate_extensions,
        test_visualization_extensions,
        test_full_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    if failed == 0:
        print("\n✅ 所有自适应调度模块功能测试通过！")
        print("\n🚀 可以使用以下命令启动Streamlit界面:")
        print("   streamlit run app.py")
    else:
        print(f"\n❌ 有 {failed} 个测试失败，请检查错误信息。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
