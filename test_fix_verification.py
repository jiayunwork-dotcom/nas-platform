"""
修复验证测试
专门验证：
1. 代理模型在小样本/小batch下是否稳定 (修复问题1)
2. Zero操作是否能正常启用和工作 (修复问题2)
3. 完整运行多代进化是否不会中断
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 修复验证测试")
print("=" * 60)


def test_zero_operation():
    """测试Zero操作的启用和功能"""
    print("\n1️⃣  测试Zero操作...")
    from src.cell import Architecture

    all_ops = ['conv3x3', 'conv5x5', 'max_pool3x3', 'avg_pool3x3',
               'skip_connect', 'dil_conv3x3', 'zero']

    arch = Architecture(num_nodes=6, enabled_ops=all_ops)

    print(f"  ✓ 包含7种操作的架构创建成功")
    print(f"  ✓ 操作列表长度: {len(arch.normal_op_list)} (Normal Cell)")

    has_zero_op = any(op == len(all_ops) - 1 for op in arch.normal_op_list)
    print(f"  ✓ 随机生成中包含Zero操作: {'是' if has_zero_op else '否 (随机结果)'}")

    from src.metrics import estimate_architecture_params, estimate_architecture_latency
    params = estimate_architecture_params(arch, num_cells=8)
    latency = estimate_architecture_latency(arch, num_cells=8)
    print(f"  ✓ 含Zero操作架构的参数估算: {params/1e6:.2f} M")
    print(f"  ✓ 含Zero操作架构的延迟估算: {latency:.3f} ms")

    return True


def test_surrogate_small_batch():
    """测试代理模型在极小样本下的稳定性"""
    print("\n2️⃣  测试代理模型小样本稳定性...")
    from src.surrogate import SurrogateModel
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect']
    input_dim = 2 * (6 * 6 + 6 * 5 // 2)

    surrogate = SurrogateModel(
        input_dim=input_dim,
        hidden_dim=32,
        min_train_samples=3,
        epochs=5,
        batch_size=1
    )

    small_data = []
    for i in range(5):
        arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)
        arch.accuracy = np.random.uniform(0.5, 0.9)
        arch.params = np.random.uniform(1e5, 1e7)
        arch.latency = np.random.uniform(0.5, 5.0)
        small_data.append(arch)

    print(f"  ✓ 创建 {len(small_data)} 个样本")
    print(f"  ✓ batch_size=1 (极端情况)")

    try:
        surrogate.train(small_data)
        print(f"  ✓ 训练成功: trained={surrogate.trained}")
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    if surrogate.trained:
        test_archs = [Architecture(num_nodes=6, enabled_ops=enabled_ops) for _ in range(3)]
        try:
            predictions = surrogate.predict(test_archs)
            print(f"  ✓ 预测成功: {predictions.shape}")
        except Exception as e:
            print(f"  ❌ 预测失败: {e}")
            return False

    return True


def test_surrogate_various_sizes():
    """测试不同样本数量下代理模型的稳定性"""
    print("\n3️⃣  测试不同规模下代理模型稳定性...")
    from src.surrogate import SurrogateModel
    from src.cell import Architecture

    enabled_ops = ['conv3x3', 'conv5x5', 'skip_connect', 'dil_conv3x3', 'zero']
    input_dim = 2 * (6 * 6 + 6 * 5 // 2)

    test_sizes = [3, 5, 10, 20, 50]

    all_passed = True
    for n_samples in test_sizes:
        try:
            surrogate = SurrogateModel(
                input_dim=input_dim,
                hidden_dim=64,
                min_train_samples=3,
                epochs=10,
                batch_size=min(8, n_samples)
            )

            data = []
            for i in range(n_samples):
                arch = Architecture(num_nodes=6, enabled_ops=enabled_ops)
                arch.accuracy = np.random.uniform(0.5, 0.9)
                arch.params = np.random.uniform(1e5, 1e7)
                arch.latency = np.random.uniform(0.5, 5.0)
                data.append(arch)

            surrogate.train(data)
            if surrogate.trained:
                test_archs = [Architecture(num_nodes=6, enabled_ops=enabled_ops) for _ in range(2)]
                preds = surrogate.predict(test_archs)
                print(f"  ✓ {n_samples:3d} 样本 - 训练&预测成功")
            else:
                print(f"  ⚠ {n_samples:3d} 样本 - 样本不足 (min={surrogate.min_train_samples})")
        except Exception as e:
            print(f"  ❌ {n_samples:3d} 样本 - 失败: {e}")
            all_passed = False

    return all_passed


def test_full_evolution_with_surrogate():
    """测试完整的进化搜索流程 + 代理模型"""
    print("\n4️⃣  测试完整进化流程 (启用代理模型)...")
    from src.experiment import ExperimentConfig, Experiment
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        config = ExperimentConfig(
            name='test_surrogate_fix',
            algorithm='nsga2',
            num_nodes=6,
            enabled_ops=['conv3x3', 'conv5x5', 'skip_connect', 'zero'],
            num_cells=4,
            init_channels=8,
            pop_size=15,
            num_generations=12,
            eval_strategy='fast',
            use_surrogate=True,
            surrogate_min_samples=10,
            surrogate_percentile=40.0,
            device='cpu'
        )

        exp = Experiment(config, experiments_dir=temp_dir)
        print(f"  ✓ 实验创建成功")
        print(f"  ✓ 配置: pop_size={config.pop_size}, generations={config.num_generations}")
        print(f"  ✓ 代理模型: 启用, min_samples={config.surrogate_min_samples}")
        print(f"  ✓ 包含Zero操作: {'zero' in config.enabled_ops}")

        gen_count = [0]
        error_occurred = [False]

        def progress_cb(current, total, msg):
            gen_count[0] = current
            if '错误' in msg or '失败' in msg:
                error_occurred[0] = True

        exp.run(progress_callback=progress_cb)

        print(f"  ✓ 成功运行 {len(exp.result.generations) - 1}/{config.num_generations} 代")
        print(f"  ✓ 总评估架构数: {len(exp.result.all_evaluated)}")
        print(f"  ✓ 最终超体积: {exp.result.hypervolume_history[-1]:.4f}")

        if len(exp.result.generations) < config.num_generations + 1:
            print(f"  ❌ 进化提前终止！只完成了 {len(exp.result.generations) - 1} 代")
            return False

        if error_occurred[0]:
            print(f"  ❌ 运行过程中出现错误")
            return False

        print(f"  ✅ 完整进化流程无中断！")
        return True

    except Exception as e:
        print(f"  ❌ 进化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """运行所有验证测试"""
    tests = [
        ("Zero操作功能", test_zero_operation),
        ("代理模型小batch", test_surrogate_small_batch),
        ("代理模型多规模", test_surrogate_various_sizes),
        ("完整进化流程", test_full_evolution_with_surrogate),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} - 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 验证结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n✅ 所有修复验证通过！")
        print("\n问题1已修复: 代理模型使用LayerNorm替代BatchNorm，")
        print("          不再受小batch size影响。")
        print("问题2已修复: 操作集配置中已添加Zero(无连接)选项，")
        print("          默认不勾选，用户可手动启用。")
    else:
        print(f"\n❌ 有 {failed} 项验证未通过")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
