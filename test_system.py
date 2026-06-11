"""
系统测试脚本
验证各模块的基本功能
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🧬 NAS实验管理平台 - 系统测试")
print("=" * 60)


def test_cell_module():
    """测试搜索空间模块"""
    print("\n📦 测试搜索空间模块...")
    from src.cell import Architecture, OP_NAMES, AVAILABLE_OPS

    enabled_ops = ['conv3x3', 'conv5x5', 'max_pool3x3', 'avg_pool3x3', 'skip_connect']
    arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)

    print(f"  ✓ 创建架构成功")
    print(f"  ✓ Normal Cell邻接矩阵形状: {arch.normal_adj.shape}")
    print(f"  ✓ Normal Cell边数: {int(arch.normal_adj.sum())}")
    print(f"  ✓ Normal Cell操作数: {len(arch.normal_op_list)}")
    print(f"  ✓ Reduction Cell边数: {int(arch.reduce_adj.sum())}")
    print(f"  ✓ 编码维度: {arch.encode().shape}")

    arch_copy = arch.copy()
    print(f"  ✓ 架构复制成功")

    return True


def test_dag_utils():
    """测试DAG有效性检查模块"""
    print("\n🔍 测试DAG有效性检查模块...")
    from src.dag_utils import (
        has_cycle, is_output_reachable, has_isolated_nodes,
        validate_architecture, enforce_dag_constraints
    )
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)

    cycle = has_cycle(arch.normal_adj)
    print(f"  ✓ 环检测: {'无环' if not cycle else '有环'}")

    reachable = is_output_reachable(arch.normal_adj)
    print(f"  ✓ 输出可达: {'是' if reachable else '否'}")

    isolated, nodes = has_isolated_nodes(arch.normal_adj)
    print(f"  ✓ 孤立节点: {'无' if not isolated else f'有: {nodes}'}")

    valid, arch_fixed = validate_architecture(arch, fix=True)
    print(f"  ✓ 架构验证修复: {'有效' if valid else '无效'}")

    return True


def test_metrics_module():
    """测试性能指标计算模块"""
    print("\n📊 测试性能指标计算模块...")
    from src.metrics import (
        estimate_architecture_params, estimate_architecture_latency,
        count_architecture_flops, fast_non_dominated_sort,
        crowding_distance, hypervolume, get_reference_point, dominates
    )
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)

    params = estimate_architecture_params(arch, num_cells=8, init_channels=16)
    print(f"  ✓ 参数量估算: {params/1e6:.2f} M")

    latency = estimate_architecture_latency(arch, num_cells=8)
    print(f"  ✓ 延迟估算: {latency:.3f} ms")

    flops = count_architecture_flops(arch, num_cells=8, init_channels=16)
    print(f"  ✓ FLOPs计算: {flops/1e6:.2f} M")

    points = np.array([
        [0.9, 1e6, 1.0],
        [0.8, 0.5e6, 0.5],
        [0.95, 2e6, 2.0],
        [0.85, 0.8e6, 0.8]
    ])
    maximize = [True, False, False]

    p1_dominates_p2 = dominates(points[0], points[1], maximize)
    print(f"  ✓ 支配关系测试: p1 {'支配' if p1_dominates_p2 else '不支配'} p2")

    fronts = fast_non_dominated_sort(points, maximize)
    print(f"  ✓ 快速非支配排序: {len(fronts)} 个前沿")

    if fronts:
        distances = crowding_distance(points, fronts[0])
        print(f"  ✓ 拥挤距离计算: 前沿0有 {len(fronts[0])} 个点")

    ref_point = get_reference_point(points, maximize)
    print(f"  ✓ 参考点: {ref_point}")

    hv = hypervolume(points, ref_point, maximize)
    print(f"  ✓ 超体积计算: {hv:.4f}")

    return True


def test_nsga2_module():
    """测试NSGA-II算法模块"""
    print("\n🧬 测试NSGA-II算法模块...")
    from src.nsga2 import NSGAII
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    algorithm = NSGAII(
        num_nodes=6,
        enabled_ops=enabled_ops,
        pop_size=10,
        mutation_rate=0.1,
        crossover_rate=0.9
    )

    population = algorithm.initialize_population()
    print(f"  ✓ 种群初始化: {len(population)} 个个体")

    for i, arch in enumerate(population):
        arch.accuracy = np.random.uniform(0.5, 0.9)
        arch.params = np.random.uniform(1e5, 1e7)
        arch.latency = np.random.uniform(0.5, 5.0)

    offspring = algorithm.make_new_population(population)
    print(f"  ✓ 生成子代: {len(offspring)} 个个体")

    next_gen = algorithm.step(population, offspring)
    print(f"  ✓ 下一代种群: {len(next_gen)} 个个体")

    return True


def test_evaluation_module():
    """测试评估策略模块"""
    print("\n📈 测试评估策略模块...")
    from src.evaluation import get_evaluator
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)

    evaluator = get_evaluator(
        eval_strategy='fast',
        num_classes=10,
        num_cells=8,
        init_channels=16,
        device='cpu'
    )
    print(f"  ✓ 创建评估器: FastProxyEvaluator")

    evaluated = evaluator.evaluate(arch)
    print(f"  ✓ 评估完成 - 精度: {evaluated.accuracy:.4f}")
    print(f"  ✓ 参数量: {evaluated.params/1e6:.2f} M")
    print(f"  ✓ 延迟: {evaluated.latency:.3f} ms")

    return True


def test_surrogate_module():
    """测试代理模型模块"""
    print("\n🤖 测试代理模型模块...")
    from src.surrogate import SurrogateModel
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    input_dim = 2 * (6 * 6 + 6 * 5 // 2)

    surrogate = SurrogateModel(
        input_dim=input_dim,
        hidden_dim=64,
        min_train_samples=10,
        epochs=10
    )
    print(f"  ✓ 创建代理模型")

    evaluated_archs = []
    for i in range(20):
        arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)
        arch.accuracy = np.random.uniform(0.5, 0.9)
        arch.params = np.random.uniform(1e5, 1e7)
        arch.latency = np.random.uniform(0.5, 5.0)
        evaluated_archs.append(arch)

    print(f"  ✓ 生成 {len(evaluated_archs)} 个已评估架构")

    surrogate.train(evaluated_archs)
    print(f"  ✓ 代理模型训练完成: {'成功' if surrogate.trained else '失败'}")

    if surrogate.trained:
        predictions = surrogate.predict(evaluated_archs[:5])
        print(f"  ✓ 预测完成: {predictions.shape}")

        mae, rmse = surrogate.get_prediction_error(evaluated_archs)
        print(f"  ✓ 预测误差 - MAE: {mae}, RMSE: {rmse}")

        candidates = [Architecture(num_nodes=6, enabled_ops=enabled_ops) for _ in range(10)]
        screened = surrogate.pre_screen(candidates, percentile=50)
        print(f"  ✓ 预筛完成: {len(screened)}/{len(candidates)} 个候选保留")

    return True


def test_experiment_module():
    """测试实验管理模块"""
    print("\n📋 测试实验管理模块...")
    from src.experiment import ExperimentConfig, Experiment, ExperimentManager
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    print(f"  ✓ 临时目录: {temp_dir}")

    config = ExperimentConfig(
        name='test_exp',
        algorithm='nsga2',
        num_nodes=6,
        enabled_ops=['conv3x3', 'conv5x5', 'skip_connect'],
        num_cells=4,
        init_channels=8,
        pop_size=10,
        num_generations=2,
        eval_strategy='fast',
        use_surrogate=False,
        device='cpu'
    )
    print(f"  ✓ 创建实验配置")

    manager = ExperimentManager(experiments_dir=temp_dir)
    print(f"  ✓ 创建实验管理器")

    exp = manager.create_experiment(config)
    print(f"  ✓ 创建实验: {config.name}")

    def progress_cb(current, total, msg):
        pass

    exp.run(progress_callback=progress_cb)
    print(f"  ✓ 运行实验完成: {len(exp.result.all_evaluated)} 个架构已评估")

    exp.save()
    print(f"  ✓ 保存实验")

    loaded_exp = Experiment.load('test_exp', temp_dir)
    print(f"  ✓ 加载实验: {len(loaded_exp.result.all_evaluated)} 个架构")

    exp_list = manager.list_experiments()
    print(f"  ✓ 实验列表: {exp_list}")

    shutil.rmtree(temp_dir)
    print(f"  ✓ 清理临时目录")

    return True


def test_visualization_module():
    """测试可视化模块"""
    print("\n🎨 测试可视化模块...")
    from src.visualization import (
        plot_pareto_2d, plot_pareto_3d, plot_hypervolume_curve,
        plot_dag_graph, plot_multi_experiment_pareto
    )
    from src.cell import Architecture

    points = np.array([
        [0.9, 1e6, 1.0],
        [0.8, 0.5e6, 0.5],
        [0.95, 2e6, 2.0],
        [0.85, 0.8e6, 0.8],
        [0.88, 1.2e6, 1.2]
    ])
    maximize = [True, False, False]

    fig1 = plot_pareto_2d(points, maximize, title='测试2D帕累托图')
    print(f"  ✓ 2D帕累托图创建成功")

    fig2 = plot_pareto_3d(points, maximize, title='测试3D帕累托图')
    print(f"  ✓ 3D帕累托图创建成功")

    hvs = [0.1, 0.2, 0.3, 0.35, 0.4]
    fig3 = plot_hypervolume_curve(hvs, title='测试超体积曲线')
    print(f"  ✓ 超体积曲线创建成功")

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)
    fig4 = plot_dag_graph(arch.normal_adj, arch.normal_op_list, enabled_ops, title='测试DAG图')
    print(f"  ✓ DAG图创建成功")

    experiments = {
        'exp1': points,
        'exp2': points * 0.9 + np.array([0.05, 0, 0])
    }
    fig5 = plot_multi_experiment_pareto(experiments, maximize, title='测试多实验对比')
    print(f"  ✓ 多实验对比图创建成功")

    return True


def main():
    """运行所有测试"""
    tests = [
        test_cell_module,
        test_dag_utils,
        test_metrics_module,
        test_nsga2_module,
        test_evaluation_module,
        test_surrogate_module,
        test_experiment_module,
        test_visualization_module,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n✅ 所有测试通过！系统功能正常。")
        print("\n🚀 启动Streamlit界面:")
        print("   streamlit run app.py")
    else:
        print(f"\n❌ 有 {failed} 个测试失败，请检查错误信息。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
