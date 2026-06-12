"""
自适应调度模块
包含: 评估策略自适应调度、进化参数自适应调整、搜索效率统计
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import copy

from .cell import Architecture
from .metrics import fast_non_dominated_sort


EVAL_STRATEGY_ORDER = ['fast', 'synflow', 'naswot', 'weight_sharing', 'full']


@dataclass
class StrategySwitchLog:
    """策略切换日志"""
    generation: int
    old_strategy: str
    new_strategy: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


@dataclass
class GenerationEfficiencyStats:
    """每代效率统计"""
    generation: int
    total_candidates: int
    actually_evaluated: int
    surrogate_skipped: int
    eval_strategy: str
    eval_time_start: float = 0.0
    eval_time_end: float = 0.0
    eval_duration: float = 0.0


@dataclass
class AdaptiveState:
    """自适应调度状态"""
    current_eval_strategy: str = 'fast'
    current_mutation_rate: float = 0.1
    current_crossover_rate: float = 0.9
    current_diversity: float = 0.5
    hv_growth_history: List[float] = field(default_factory=list)
    pareto_size_history: List[int] = field(default_factory=list)
    strategy_switch_logs: List[StrategySwitchLog] = field(default_factory=list)
    efficiency_stats: List[GenerationEfficiencyStats] = field(default_factory=list)
    low_hv_growth_consecutive: int = 0
    diversity_alarm_triggered: bool = False
    base_mutation_rate: float = 0.1
    base_crossover_rate: float = 0.9


def compute_population_diversity(population: List[Architecture]) -> float:
    """
    计算种群多样性指标
    种群多样性指标 = 种群中所有个体编码向量的平均两两欧氏距离 / 编码空间对角线长度
    归一化到 0~1

    Args:
        population: 架构种群列表

    Returns:
        diversity: 0~1之间的多样性值
    """
    if len(population) < 2:
        return 0.0

    encodings = []
    for arch in population:
        encodings.append(arch.encode().astype(float))
    encodings = np.array(encodings)

    n = len(encodings)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(encodings[i] - encodings[j])
            distances.append(dist)

    avg_distance = np.mean(distances) if distances else 0.0

    encoding_dim = encodings.shape[1]
    max_op_value = len(population[0].enabled_ops) if population else 7
    diagonal_length = np.sqrt(encoding_dim) * max_op_value

    diversity = avg_distance / diagonal_length if diagonal_length > 0 else 0.0
    return min(1.0, max(0.0, diversity))


def get_pareto_front_size(population: List[Architecture], maximize: List[bool]) -> int:
    """获取帕累托前沿解的数量"""
    points = np.array([[a.accuracy, a.params, a.latency] for a in population
                       if a.accuracy is not None])
    if len(points) == 0:
        return 0
    fronts = fast_non_dominated_sort(points, maximize)
    return len(fronts[0]) if fronts else 0


def compute_hypervolume_growth_rate(hv_history: List[float], window: int = 1) -> float:
    """
    计算超体积增长率
    """
    if len(hv_history) < window + 1:
        return 1.0
    prev = hv_history[-(window + 1)]
    curr = hv_history[-1]
    if prev == 0:
        return 1.0 if curr > 0 else 0.0
    return (curr - prev) / abs(prev)


class AdaptiveScheduler:
    """
    自适应调度器
    根据搜索过程中的实时反馈自动调整评估策略和进化参数
    """

    def __init__(self, base_mutation_rate: float = 0.1, base_crossover_rate: float = 0.9):
        self.state = AdaptiveState(
            base_mutation_rate=base_mutation_rate,
            base_crossover_rate=base_crossover_rate,
            current_mutation_rate=base_mutation_rate,
            current_crossover_rate=base_crossover_rate
        )
        self.maximize = [True, False, False]

    def _switch_strategy(self, generation: int, new_strategy: str, reason: str):
        """切换评估策略并记录日志"""
        old_strategy = self.state.current_eval_strategy
        if old_strategy != new_strategy:
            log = StrategySwitchLog(
                generation=generation,
                old_strategy=old_strategy,
                new_strategy=new_strategy,
                reason=reason
            )
            self.state.strategy_switch_logs.append(log)
            self.state.current_eval_strategy = new_strategy

    def _upgrade_eval_strategy(self, generation: int, reason: str):
        """升级到更精确的评估策略"""
        current_idx = EVAL_STRATEGY_ORDER.index(self.state.current_eval_strategy)
        if current_idx < len(EVAL_STRATEGY_ORDER) - 1:
            new_strategy = EVAL_STRATEGY_ORDER[current_idx + 1]
            self._switch_strategy(generation, new_strategy, reason)

    def update_eval_strategy(self, generation: int, population: List[Architecture],
                              hv_history: List[float], surrogate_ready: bool):
        """
        根据搜索阶段和性能指标自适应调整评估策略

        规则:
        1. 前5代使用SynFlow快速筛选
        2. 第6代开始如果代理模型已训练好，切换为代理模型预筛+快速代理真实评估的混合模式
        3. 连续3代超体积增长率低于5%时，升级为更精确的评估策略
        4. 某代帕累托前沿解数量比上一代减少超过20%时，触发种群多样性警报，强制校准
        """
        hv_growth_rate = compute_hypervolume_growth_rate(hv_history)
        self.state.hv_growth_history.append(hv_growth_rate)

        pareto_size = get_pareto_front_size(population, self.maximize)
        self.state.pareto_size_history.append(pareto_size)

        if generation <= 5:
            if self.state.current_eval_strategy != 'synflow':
                self._switch_strategy(
                    generation, 'synflow',
                    f"第{generation}代(≤5)，使用SynFlow零代价代理快速筛选积累初始样本"
                )
            return

        if generation == 6 and surrogate_ready:
            self._switch_strategy(
                generation, 'fast',
                "第6代开始，代理模型已就绪，切换为代理预筛+快速代理评估的混合模式"
            )
            return

        if len(self.state.hv_growth_history) >= 3:
            recent_growths = self.state.hv_growth_history[-3:]
            if all(g < 0.05 for g in recent_growths):
                self.state.low_hv_growth_consecutive += 1
                if self.state.low_hv_growth_consecutive >= 1:
                    self._upgrade_eval_strategy(
                        generation,
                        f"连续3代超体积增长率低于5% "
                        f"({recent_growths[0]*100:.1f}%, {recent_growths[1]*100:.1f}%, "
                        f"{recent_growths[2]*100:.1f}%)，升级评估策略以获得更精确结果"
                    )
            else:
                self.state.low_hv_growth_consecutive = 0

        if len(self.state.pareto_size_history) >= 2:
            prev_size = self.state.pareto_size_history[-2]
            curr_size = self.state.pareto_size_history[-1]
            if prev_size > 0:
                reduction_rate = (prev_size - curr_size) / prev_size
                if reduction_rate > 0.2:
                    self.state.diversity_alarm_triggered = True
                    self._switch_strategy(
                        generation, 'full',
                        f"帕累托前沿解数量从{prev_size}减少到{curr_size}，"
                        f"减少{reduction_rate*100:.1f}%>20%，触发种群多样性警报，"
                        f"强制使用完整评估校准当前前沿解"
                    )

    def check_diversity_alarm(self) -> bool:
        """检查是否触发了多样性警报（需要强制校准）"""
        if self.state.diversity_alarm_triggered:
            self.state.diversity_alarm_triggered = False
            return True
        return False

    def update_evolution_params(self, population: List[Architecture]) -> Tuple[float, float]:
        """
        根据种群多样性动态调整变异概率和交叉概率

        规则:
        - 多样性 < 0.3: 变异概率提升50%，交叉概率降低（促进探索，减少趋同）
        - 多样性 > 0.7: 变异概率降低30%，交叉概率提升（加强开发，促进优秀基因组合）
        - 0.3 ≤ 多样性 ≤ 0.7: 使用基础参数

        Returns:
            (mutation_rate, crossover_rate): 调整后的参数
        """
        diversity = compute_population_diversity(population)
        self.state.current_diversity = diversity

        base_mut = self.state.base_mutation_rate
        base_cross = self.state.base_crossover_rate

        if diversity < 0.3:
            new_mut = base_mut * 1.5
            new_cross = base_cross * 0.7
        elif diversity > 0.7:
            new_mut = base_mut * 0.7
            new_cross = base_cross * 1.3
        else:
            new_mut = base_mut
            new_cross = base_cross

        new_mut = min(0.9, max(0.01, new_mut))
        new_cross = min(1.0, max(0.1, new_cross))

        self.state.current_mutation_rate = new_mut
        self.state.current_crossover_rate = new_cross

        return new_mut, new_cross

    def record_efficiency_stats(self, gen_stats: GenerationEfficiencyStats):
        """记录每代的效率统计"""
        self.state.efficiency_stats.append(gen_stats)

    def get_efficiency_summary(self) -> Dict:
        """
        获取效率统计汇总

        Returns:
            包含以下信息的字典:
            - total_evaluated: 总实际评估数
            - total_skipped: 总代理跳过数
            - total_candidates: 总候选架构数
            - savings_percent: 节省的评估百分比
            - strategy_durations: 每种策略的使用时长
            - strategy_duration_percent: 每种策略的使用时长占比
            - hv_per_eval: 每次评估带来的平均超体积贡献（需要外部传入最终HV）
        """
        total_evaluated = sum(s.actually_evaluated for s in self.state.efficiency_stats)
        total_skipped = sum(s.surrogate_skipped for s in self.state.efficiency_stats)
        total_candidates = total_evaluated + total_skipped
        savings_percent = (total_skipped / total_candidates * 100) if total_candidates > 0 else 0.0

        strategy_durations = {}
        for s in self.state.efficiency_stats:
            strategy = s.eval_strategy
            if strategy not in strategy_durations:
                strategy_durations[strategy] = 0.0
            strategy_durations[strategy] += s.eval_duration

        total_duration = sum(strategy_durations.values())
        strategy_duration_percent = {
            k: (v / total_duration * 100) if total_duration > 0 else 0.0
            for k, v in strategy_durations.items()
        }

        return {
            'total_evaluated': total_evaluated,
            'total_skipped': total_skipped,
            'total_candidates': total_candidates,
            'savings_percent': savings_percent,
            'strategy_durations': strategy_durations,
            'strategy_duration_percent': strategy_duration_percent,
        }
