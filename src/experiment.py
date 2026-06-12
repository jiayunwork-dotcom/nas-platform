"""
实验管理模块
包含: 实验存储、搜索运行、对比分析、随机搜索基线
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime
import copy

from .cell import Architecture
from .nsga2 import NSGAII, RandomSearch
from .evaluation import Evaluator, get_evaluator
from .surrogate import SurrogateModel
from .metrics import (
    hypervolume, get_reference_point,
    fast_non_dominated_sort, get_pareto_front
)
from .adaptive import (
    AdaptiveScheduler, StrategySwitchLog, GenerationEfficiencyStats,
    AdaptiveState
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    algorithm: str = 'nsga2'
    num_nodes: int = 6
    enabled_ops: List[str] = field(default_factory=lambda: [
        'conv3x3', 'conv5x5', 'dil_conv3x3',
        'max_pool3x3', 'avg_pool3x3', 'skip_connect'
    ])
    num_cells: int = 8
    init_channels: int = 16
    pop_size: int = 50
    num_generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    eval_strategy: str = 'fast'
    eval_epochs: int = 20
    use_surrogate: bool = True
    surrogate_min_samples: int = 50
    surrogate_percentile: float = 30.0
    device: str = 'cpu'
    use_adaptive_scheduling: bool = True
    objective_weights: List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])
    created_at: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        return cls(**d)


@dataclass
class GenerationSnapshot:
    """每代快照"""
    generation: int
    population: List[Architecture]
    hypervolume: float
    avg_accuracy: float
    avg_params: float
    avg_latency: float
    surrogate_used: bool = False
    eval_strategy: str = 'fast'
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    diversity: float = 0.5
    actually_evaluated: int = 0
    surrogate_skipped: int = 0
    eval_duration: float = 0.0

    def get_fitness_matrix(self) -> np.ndarray:
        return np.array([[a.accuracy, a.params, a.latency] for a in self.population])


@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    generations: List[GenerationSnapshot] = field(default_factory=list)
    all_evaluated: List[Architecture] = field(default_factory=list)
    hypervolume_history: List[float] = field(default_factory=list)
    completed: bool = False
    strategy_switch_logs: List[StrategySwitchLog] = field(default_factory=list)
    efficiency_stats: List[GenerationEfficiencyStats] = field(default_factory=list)

    def get_all_points(self) -> np.ndarray:
        return np.array([[a.accuracy, a.params, a.latency] for a in self.all_evaluated])

    def get_pareto_front(self) -> np.ndarray:
        points = self.get_all_points()
        return get_pareto_front(points, [True, False, False])


class Experiment:
    """
    NAS实验类
    管理整个搜索过程
    """
    def __init__(self, config: ExperimentConfig, experiments_dir: str = './experiments'):
        self.config = config
        self.experiments_dir = experiments_dir
        self.result = ExperimentResult(config=config)
        self.algorithm = None
        self.evaluator = None
        self.evaluators: Dict[str, Evaluator] = {}
        self.surrogate = None
        self.random_search_baseline = None
        self.adaptive_scheduler = None
        self._initialized = False

        self._init_algorithm()
        self._init_evaluators()
        self._init_surrogate()
        self._init_adaptive_scheduler()

    def _init_algorithm(self):
        """初始化进化算法"""
        if self.config.algorithm == 'nsga2':
            self.algorithm = NSGAII(
                num_nodes=self.config.num_nodes,
                enabled_ops=self.config.enabled_ops.copy(),
                pop_size=self.config.pop_size,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate
            )
        elif self.config.algorithm == 'random':
            self.algorithm = RandomSearch(
                num_nodes=self.config.num_nodes,
                enabled_ops=self.config.enabled_ops.copy(),
                pop_size=self.config.pop_size
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def _init_evaluators(self):
        """初始化所有评估器"""
        all_strategies = ['fast', 'synflow', 'naswot', 'weight_sharing', 'full']
        for strategy in all_strategies:
            self.evaluators[strategy] = get_evaluator(
                eval_strategy=strategy,
                num_classes=10,
                num_cells=self.config.num_cells,
                init_channels=self.config.init_channels,
                device=self.config.device,
                epochs=self.config.eval_epochs
            )
        self.evaluator = self.evaluators[self.config.eval_strategy]

    def _init_adaptive_scheduler(self):
        """初始化自适应调度器"""
        if self.config.use_adaptive_scheduling:
            self.adaptive_scheduler = AdaptiveScheduler(
                base_mutation_rate=self.config.mutation_rate,
                base_crossover_rate=self.config.crossover_rate,
                objective_weights=self.config.objective_weights
            )

    def _init_surrogate(self):
        """初始化代理模型"""
        if self.config.use_surrogate:
            input_dim = self._get_surrogate_input_dim()
            self.surrogate = SurrogateModel(
                input_dim=input_dim,
                min_train_samples=self.config.surrogate_min_samples
            )

    def _get_surrogate_input_dim(self) -> int:
        """计算代理模型输入维度"""
        n = self.config.num_nodes
        adj_dim = n * n
        op_dim = n * (n - 1) // 2
        return 2 * (adj_dim + op_dim)

    def _compute_hypervolume(self, population: List[Architecture]) -> float:
        """计算种群的超体积"""
        points = np.array([[a.accuracy, a.params, a.latency] for a in population
                          if a.accuracy is not None])
        if len(points) == 0:
            return 0.0
        ref_point = get_reference_point(points, [True, False, False])
        return hypervolume(points, ref_point, [True, False, False])

    def _evaluate_population(self, population: List[Architecture],
                             use_surrogate: bool = False,
                             eval_strategy: Optional[str] = None) -> Tuple[List[Architecture], int, int, float]:
        """
        评估种群

        Args:
            population: 待评估的架构种群
            use_surrogate: 是否使用代理模型预筛
            eval_strategy: 指定使用的评估策略，None则使用默认

        Returns:
            (population, actually_evaluated, surrogate_skipped, eval_duration)
        """
        import time

        strategy = eval_strategy or self.config.eval_strategy
        evaluator = self.evaluators.get(strategy, self.evaluator)

        to_evaluate = population
        total_candidates = len(population)
        surrogate_skipped = 0

        if use_surrogate and self.surrogate and self.surrogate.is_ready():
            screened = self.surrogate.pre_screen(
                population,
                percentile=self.config.surrogate_percentile
            )
            screened_indices = set(id(a) for a in screened)
            to_evaluate = [a for a in population if id(a) in screened_indices]
            surrogate_skipped = total_candidates - len(to_evaluate)

        start_time = time.time()

        for arch in to_evaluate:
            if arch.accuracy is None:
                evaluator.evaluate(arch)
                arch.is_evaluated = True
                self.result.all_evaluated.append(arch)

        eval_duration = time.time() - start_time

        actually_evaluated = len(to_evaluate)

        for arch in population:
            if arch.accuracy is None:
                arch.accuracy = 0.1
                arch.params = 1e7
                arch.latency = 10.0

        return population, actually_evaluated, surrogate_skipped, eval_duration

    def _update_surrogate(self, new_archs: Optional[List[Architecture]] = None):
        """
        更新代理模型（支持增量训练）

        Args:
            new_archs: 本代新增的已评估架构，如果为None则进行全量训练
        """
        if self.surrogate and len(self.result.all_evaluated) >= self.config.surrogate_min_samples:
            if self.surrogate.trained and new_archs and len(new_archs) > 0:
                self.surrogate.incremental_train(new_archs, self.result.all_evaluated)
            else:
                self.surrogate.train(self.result.all_evaluated)

    def _create_snapshot(self, generation: int, population: List[Architecture],
                         surrogate_used: bool, eval_strategy: str = 'fast',
                         mutation_rate: float = 0.1, crossover_rate: float = 0.9,
                         diversity: float = 0.5, actually_evaluated: int = 0,
                         surrogate_skipped: int = 0, eval_duration: float = 0.0) -> GenerationSnapshot:
        """创建代数快照"""
        evaluated = [a for a in population if a.accuracy is not None]
        if len(evaluated) == 0:
            evaluated = population

        hv = self._compute_hypervolume(population)
        avg_acc = np.mean([a.accuracy for a in evaluated])
        avg_params = np.mean([a.params for a in evaluated])
        avg_latency = np.mean([a.latency for a in evaluated])

        return GenerationSnapshot(
            generation=generation,
            population=copy.deepcopy(population),
            hypervolume=hv,
            avg_accuracy=avg_acc,
            avg_params=avg_params,
            avg_latency=avg_latency,
            surrogate_used=surrogate_used,
            eval_strategy=eval_strategy,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            diversity=diversity,
            actually_evaluated=actually_evaluated,
            surrogate_skipped=surrogate_skipped,
            eval_duration=eval_duration
        )

    def run(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """运行完整搜索（集成自适应调度）"""
        if self._initialized:
            return

        population = self.algorithm.initialize_population()

        if progress_callback:
            progress_callback(0, self.config.num_generations, "初始化种群完成，开始评估...")

        use_adaptive = self.config.use_adaptive_scheduling and self.adaptive_scheduler is not None

        current_strategy = self.config.eval_strategy
        current_mut = self.config.mutation_rate
        current_cross = self.config.crossover_rate
        current_diversity = 0.5

        if use_adaptive:
            current_strategy = self.adaptive_scheduler.state.current_eval_strategy

        eval_count_before = len(self.result.all_evaluated)
        population, actually_eval, skipped, eval_dur = self._evaluate_population(
            population, use_surrogate=False, eval_strategy=current_strategy
        )
        new_archs_gen0 = self.result.all_evaluated[eval_count_before:]

        snapshot = self._create_snapshot(
            0, population, surrogate_used=False,
            eval_strategy=current_strategy,
            mutation_rate=current_mut, crossover_rate=current_cross,
            diversity=current_diversity,
            actually_evaluated=actually_eval, surrogate_skipped=skipped,
            eval_duration=eval_dur
        )
        self.result.generations.append(snapshot)
        self.result.hypervolume_history.append(snapshot.hypervolume)

        if use_adaptive:
            gen_stats = GenerationEfficiencyStats(
                generation=0, total_candidates=len(population),
                actually_evaluated=actually_eval, surrogate_skipped=skipped,
                eval_strategy=current_strategy, eval_duration=eval_dur
            )
            self.adaptive_scheduler.record_efficiency_stats(gen_stats)

        self._update_surrogate(new_archs_gen0)

        for gen in range(1, self.config.num_generations + 1):
            if progress_callback:
                progress_callback(gen, self.config.num_generations, f"运行第 {gen} 代...")

            if use_adaptive:
                current_diversity = self.adaptive_scheduler.state.current_diversity
                current_mut, current_cross = self.adaptive_scheduler.update_evolution_params(population)
                self.algorithm.mutation_rate = current_mut
                self.algorithm.crossover_rate = current_cross

            offspring = self.algorithm.make_new_population(population)

            if use_adaptive:
                surrogate_ready = (self.surrogate is not None and
                                   len(self.result.all_evaluated) >= self.config.surrogate_min_samples
                                   and self.surrogate.trained)
                self.adaptive_scheduler.update_eval_strategy(
                    gen, population, self.result.hypervolume_history, surrogate_ready
                )
                current_strategy = self.adaptive_scheduler.state.current_eval_strategy

                if self.adaptive_scheduler.check_diversity_alarm():
                    if progress_callback:
                        progress_callback(gen, self.config.num_generations,
                                         f"第 {gen} 代: 触发多样性警报，对帕累托前沿解做完整校准评估...")
                    from .metrics import fast_non_dominated_sort
                    points = np.array([[a.accuracy, a.params, a.latency] for a in population
                                       if a.accuracy is not None])
                    if len(points) > 0:
                        fronts = fast_non_dominated_sort(points, [True, False, False])
                        if fronts and len(fronts[0]) > 0:
                            pareto_archs = [population[i] for i in fronts[0]]
                            full_evaluator = self.evaluators.get('full', self.evaluator)
                            for arch in pareto_archs:
                                full_evaluator.evaluate(arch)
                                arch.is_evaluated = True
                                if arch not in self.result.all_evaluated:
                                    self.result.all_evaluated.append(arch)

            use_surrogate = (self.surrogate is not None and
                           len(self.result.all_evaluated) >= self.config.surrogate_min_samples)

            eval_count_before = len(self.result.all_evaluated)
            offspring, actually_eval, skipped, eval_dur = self._evaluate_population(
                offspring, use_surrogate=use_surrogate, eval_strategy=current_strategy
            )
            new_archs_this_gen = self.result.all_evaluated[eval_count_before:]

            population = self.algorithm.step(population, offspring)

            self._update_surrogate(new_archs_this_gen)

            snapshot = self._create_snapshot(
                gen, population, surrogate_used=use_surrogate,
                eval_strategy=current_strategy,
                mutation_rate=current_mut, crossover_rate=current_cross,
                diversity=current_diversity,
                actually_evaluated=actually_eval, surrogate_skipped=skipped,
                eval_duration=eval_dur
            )
            self.result.generations.append(snapshot)
            self.result.hypervolume_history.append(snapshot.hypervolume)

            if use_adaptive:
                gen_stats = GenerationEfficiencyStats(
                    generation=gen, total_candidates=len(offspring),
                    actually_evaluated=actually_eval, surrogate_skipped=skipped,
                    eval_strategy=current_strategy, eval_duration=eval_dur
                )
                self.adaptive_scheduler.record_efficiency_stats(gen_stats)

            if progress_callback:
                progress_callback(gen, self.config.num_generations,
                                f"第 {gen} 代完成 - HV: {snapshot.hypervolume:.4f}, "
                                f"策略: {current_strategy}, "
                                f"变异率: {current_mut:.3f}, 交叉率: {current_cross:.3f}, "
                                f"多样性: {current_diversity:.3f}")

        if use_adaptive:
            self.result.strategy_switch_logs = self.adaptive_scheduler.state.strategy_switch_logs
            self.result.efficiency_stats = self.adaptive_scheduler.state.efficiency_stats

        self.result.completed = True
        self._initialized = True

        if progress_callback:
            progress_callback(self.config.num_generations, self.config.num_generations, "搜索完成！")

    def save(self):
        """保存实验结果"""
        os.makedirs(self.experiments_dir, exist_ok=True)
        exp_path = os.path.join(self.experiments_dir, f"{self.config.name}.pkl")
        with open(exp_path, 'wb') as f:
            pickle.dump(self, f)

        config_path = os.path.join(self.experiments_dir, f"{self.config.name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, name: str, experiments_dir: str = './experiments') -> 'Experiment':
        """加载实验"""
        exp_path = os.path.join(experiments_dir, f"{name}.pkl")
        with open(exp_path, 'rb') as f:
            return pickle.load(f)

    def get_pareto_architectures(self) -> List[Architecture]:
        """获取帕累托前沿架构"""
        points = self.result.get_all_points()
        fronts = fast_non_dominated_sort(points, [True, False, False])
        if not fronts:
            return []
        return [self.result.all_evaluated[i] for i in fronts[0]]

    def export_to_csv(self, filepath: str):
        """导出评估结果到CSV"""
        data = []
        for i, arch in enumerate(self.result.all_evaluated):
            row = {
                'id': i,
                'accuracy': arch.accuracy,
                'params': arch.params,
                'latency': arch.latency,
                'normal_adj': json.dumps(arch.normal_adj.astype(int).tolist()),
                'normal_ops': json.dumps(arch.normal_op_list),
                'reduce_adj': json.dumps(arch.reduce_adj.astype(int).tolist()),
                'reduce_ops': json.dumps(arch.reduce_op_list),
            }
            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def import_external_results(self, filepath: str):
        """导入外部架构评估结果"""
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            arch = Architecture(
                num_nodes=self.config.num_nodes,
                enabled_ops=self.config.enabled_ops.copy(),
                normal_adj=np.array(json.loads(row['normal_adj']), dtype=bool),
                normal_op_list=json.loads(row['normal_ops']),
                reduce_adj=np.array(json.loads(row['reduce_adj']), dtype=bool),
                reduce_op_list=json.loads(row['reduce_ops'])
            )
            arch.accuracy = row['accuracy']
            arch.params = row['params']
            arch.latency = row['latency']
            self.result.all_evaluated.append(arch)


class ExperimentManager:
    """
    实验管理器
    管理多个实验，支持对比分析
    """
    def __init__(self, experiments_dir: str = './experiments'):
        self.experiments_dir = experiments_dir
        self.experiments: Dict[str, Experiment] = {}
        os.makedirs(self.experiments_dir, exist_ok=True)
        self._load_existing_experiments()

    def _load_existing_experiments(self):
        """加载已存在的实验"""
        if not os.path.exists(self.experiments_dir):
            return
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.pkl'):
                name = filename[:-4]
                try:
                    self.experiments[name] = Experiment.load(name, self.experiments_dir)
                except Exception as e:
                    print(f"Warning: Could not load experiment {name}: {e}")

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """创建新实验"""
        if config.name in self.experiments:
            raise ValueError(f"Experiment {config.name} already exists")
        exp = Experiment(config, self.experiments_dir)
        self.experiments[config.name] = exp
        return exp

    def delete_experiment(self, name: str):
        """删除实验"""
        if name in self.experiments:
            del self.experiments[name]
            for ext in ['.pkl', '_config.json']:
                filepath = os.path.join(self.experiments_dir, f"{name}{ext}")
                if os.path.exists(filepath):
                    os.remove(filepath)

    def list_experiments(self) -> List[str]:
        """列出所有实验"""
        return list(self.experiments.keys())

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """获取实验"""
        return self.experiments.get(name)

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, ExperimentResult]:
        """比较多个实验"""
        results = {}
        for name in experiment_names:
            if name in self.experiments:
                results[name] = self.experiments[name].result
        return results

    def get_multi_experiment_pareto(self, experiment_names: List[str]) -> Dict[str, np.ndarray]:
        """获取多个实验的所有评估点用于对比"""
        data = {}
        for name in experiment_names:
            if name in self.experiments:
                data[name] = self.experiments[name].result.get_all_points()
        return data

    def get_multi_experiment_hypervolumes(self, experiment_names: List[str]) -> Dict[str, List[float]]:
        """获取多个实验的超体积历史"""
        data = {}
        for name in experiment_names:
            if name in self.experiments:
                data[name] = self.experiments[name].result.hypervolume_history
        return data

    def run_random_search_baseline(self, base_config: ExperimentConfig,
                                   num_runs: int = 1) -> Experiment:
        """
        运行随机搜索基线实验
        """
        baseline_config = copy.deepcopy(base_config)
        baseline_config.name = f"{base_config.name}_random_baseline"
        baseline_config.algorithm = 'random'

        if baseline_config.name in self.experiments:
            return self.experiments[baseline_config.name]

        exp = self.create_experiment(baseline_config)
        exp.run()
        exp.save()
        return exp

    def save_all(self):
        """保存所有实验"""
        for exp in self.experiments.values():
            exp.save()
