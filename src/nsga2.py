"""
NSGA-II多目标进化算法实现
包含: 选择、交叉、变异操作
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import copy

from .cell import Architecture, set_op_in_list
from .dag_utils import validate_architecture, enforce_dag_constraints
from .metrics import fast_non_dominated_sort, crowding_distance, dominates


class NSGAII:
    """
    NSGA-II多目标进化算法
    """
    def __init__(self,
                 num_nodes: int = 6,
                 enabled_ops: Optional[List[str]] = None,
                 pop_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 tournament_size: int = 3):
        self.num_nodes = num_nodes
        self.enabled_ops = enabled_ops or ['conv3x3', 'conv5x5', 'dil_conv3x3',
                                           'max_pool3x3', 'avg_pool3x3', 'skip_connect']
        self.num_ops = len(self.enabled_ops)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.maximize = [True, False, False]

    def initialize_population(self) -> List[Architecture]:
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            arch = Architecture(self.num_nodes, self.enabled_ops.copy())
            _, arch = validate_architecture(arch, fix=True)
            population.append(arch)
        return population

    def tournament_selection(self, population: List[Architecture],
                             fitness: np.ndarray) -> Architecture:
        """
        锦标赛选择
        先比较支配层级，同层比较拥挤距离
        """
        fronts = fast_non_dominated_sort(fitness, self.maximize)

        rank = {}
        for front_idx, front in enumerate(fronts):
            for idx in front:
                rank[idx] = front_idx

        distances = np.zeros(len(population))
        for front in fronts:
            if len(front) > 1:
                front_distances = crowding_distance(fitness, front)
                for i, idx in enumerate(front):
                    distances[idx] = front_distances[i]

        candidates = np.random.choice(len(population), self.tournament_size, replace=False)

        best_idx = candidates[0]
        for idx in candidates[1:]:
            if rank[idx] < rank[best_idx]:
                best_idx = idx
            elif rank[idx] == rank[best_idx]:
                if distances[idx] > distances[best_idx]:
                    best_idx = idx

        return copy.deepcopy(population[best_idx])

    def crossover_adjacency(self, adj1: np.ndarray, adj2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        邻接矩阵单点行交叉
        随机选择一行，交换该行之后的所有行
        """
        n = adj1.shape[0]
        cross_point = np.random.randint(0, n)

        child1_adj = adj1.copy()
        child2_adj = adj2.copy()

        child1_adj[cross_point:, :] = adj2[cross_point:, :]
        child2_adj[cross_point:, :] = adj1[cross_point:, :]

        child1_adj = np.triu(child1_adj, k=1)
        child2_adj = np.triu(child2_adj, k=1)

        return child1_adj, child2_adj

    def crossover_op_list(self, op_list1: List[int], op_list2: List[int],
                          adj1: np.ndarray, adj2: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        操作列表均匀交叉
        对每条边独立决定继承哪个父代的操作
        """
        n = adj1.shape[0]

        child1_ops = []
        child2_ops = []

        idx1 = 0
        idx2 = 0

        for i in range(n):
            for j in range(i + 1, n):
                if adj1[i, j] or adj2[i, j]:
                    if np.random.random() > 0.5:
                        op1 = op_list1[idx1] if (idx1 < len(op_list1) and adj1[i, j]) else (
                            op_list2[idx2] if idx2 < len(op_list2) else 0)
                        op2 = op_list2[idx2] if (idx2 < len(op_list2) and adj2[i, j]) else (
                            op_list1[idx1] if idx1 < len(op_list1) else 0)
                    else:
                        op1 = op_list2[idx2] if (idx2 < len(op_list2) and adj2[i, j]) else (
                            op_list1[idx1] if idx1 < len(op_list1) else 0)
                        op2 = op_list1[idx1] if (idx1 < len(op_list1) and adj1[i, j]) else (
                            op_list2[idx2] if idx2 < len(op_list2) else 0)

                    if adj1[i, j]:
                        child1_ops.append(op1)
                        idx1 += 1
                    if adj2[i, j]:
                        child2_ops.append(op2)
                        idx2 += 1

        return child1_ops, child2_ops

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Tuple[Architecture, Architecture]:
        """
        交叉操作
        """
        if np.random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1_norm_adj, child2_norm_adj = self.crossover_adjacency(
            parent1.normal_adj, parent2.normal_adj
        )
        child1_red_adj, child2_red_adj = self.crossover_adjacency(
            parent1.reduce_adj, parent2.reduce_adj
        )

        child1_norm_ops, child2_norm_ops = self.crossover_op_list(
            parent1.normal_op_list, parent2.normal_op_list,
            child1_norm_adj, child2_norm_adj
        )
        child1_red_ops, child2_red_ops = self.crossover_op_list(
            parent1.reduce_op_list, parent2.reduce_op_list,
            child1_red_adj, child2_red_adj
        )

        child1 = Architecture(self.num_nodes, self.enabled_ops.copy(),
                              child1_norm_adj, child1_norm_ops,
                              child1_red_adj, child1_red_ops)
        child2 = Architecture(self.num_nodes, self.enabled_ops.copy(),
                              child2_norm_adj, child2_norm_ops,
                              child2_red_adj, child2_red_ops)

        _, child1 = validate_architecture(child1, fix=True)
        _, child2 = validate_architecture(child2, fix=True)

        return child1, child2

    def mutate_adjacency(self, adj: np.ndarray, op_list: List[int]) -> Tuple[np.ndarray, List[int]]:
        """
        变异邻接矩阵：随机添加或删除一条边
        """
        n = adj.shape[0]
        adj = adj.copy()
        op_list = op_list.copy()

        existing_edges = []
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    existing_edges.append((i, j))
                else:
                    possible_edges.append((i, j))

        if np.random.random() > 0.5 and len(existing_edges) > 1:
            idx = np.random.randint(len(existing_edges))
            i, j = existing_edges[idx]
            adj[i, j] = False

            new_op_list = []
            op_idx = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if adj[ii, jj]:
                        if op_idx < len(op_list):
                            new_op_list.append(op_list[op_idx])
                        op_idx += 1
                    elif (ii, jj) == (i, j):
                        op_idx += 1
            op_list = new_op_list
        elif len(possible_edges) > 0:
            idx = np.random.randint(len(possible_edges))
            i, j = possible_edges[idx]
            adj[i, j] = True

            new_op_list = []
            op_idx = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if (ii, jj) == (i, j):
                        new_op_list.append(np.random.randint(0, self.num_ops))
                    if adj[ii, jj] and (ii, jj) != (i, j):
                        if op_idx < len(op_list):
                            new_op_list.append(op_list[op_idx])
                        op_idx += 1
            op_list = new_op_list

        return adj, op_list

    def mutate_operations(self, adj: np.ndarray, op_list: List[int]) -> List[int]:
        """
        变异操作类型：随机修改某条边的操作
        """
        op_list = op_list.copy()
        if len(op_list) > 0:
            idx = np.random.randint(len(op_list))
            op_list[idx] = np.random.randint(0, self.num_ops)
        return op_list

    def mutate(self, arch: Architecture) -> Architecture:
        """
        变异操作
        """
        if np.random.random() > self.mutation_rate:
            return arch

        arch = copy.deepcopy(arch)

        if np.random.random() > 0.5:
            arch.normal_adj, arch.normal_op_list = self.mutate_adjacency(
                arch.normal_adj, arch.normal_op_list
            )
            arch.reduce_adj, arch.reduce_op_list = self.mutate_adjacency(
                arch.reduce_adj, arch.reduce_op_list
            )
        else:
            arch.normal_op_list = self.mutate_operations(
                arch.normal_adj, arch.normal_op_list
            )
            arch.reduce_op_list = self.mutate_operations(
                arch.reduce_adj, arch.reduce_op_list
            )

        _, arch = validate_architecture(arch, fix=True)

        return arch

    def _get_fitness_array(self, population: List[Architecture]) -> np.ndarray:
        """获取适应度数组"""
        fitness = []
        for arch in population:
            if arch.accuracy is None or arch.params is None or arch.latency is None:
                fitness.append([0.0, 1e10, 1e10])
            else:
                fitness.append([arch.accuracy, arch.params, arch.latency])
        return np.array(fitness)

    def make_new_population(self, population: List[Architecture]) -> List[Architecture]:
        """生成新一代种群（仅执行进化操作，不评估）"""
        fitness = self._get_fitness_array(population)
        new_pop = []

        while len(new_pop) < self.pop_size:
            parent1 = self.tournament_selection(population, fitness)
            parent2 = self.tournament_selection(population, fitness)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            _, child1 = validate_architecture(child1, fix=True)
            _, child2 = validate_architecture(child2, fix=True)
            new_pop.extend([child1, child2])

        return new_pop[:self.pop_size]

    def select_next_generation(self,
                               parent_pop: List[Architecture],
                               offspring_pop: List[Architecture]) -> List[Architecture]:
        """
        精英保留策略：从父代和子代合并种群中选择下一代
        """
        combined = parent_pop + offspring_pop
        fitness = self._get_fitness_array(combined)

        fronts = fast_non_dominated_sort(fitness, self.maximize)

        next_gen = []
        front_idx = 0

        while front_idx < len(fronts) and len(next_gen) + len(fronts[front_idx]) <= self.pop_size:
            for idx in fronts[front_idx]:
                next_gen.append(copy.deepcopy(combined[idx]))
            front_idx += 1

        if len(next_gen) < self.pop_size and front_idx < len(fronts):
            front = fronts[front_idx]
            if len(front) > 1:
                distances = crowding_distance(fitness, front)
                sorted_indices = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
                remaining = self.pop_size - len(next_gen)
                for i in range(remaining):
                    next_gen.append(copy.deepcopy(combined[front[sorted_indices[i]]]))
            else:
                for idx in front[:self.pop_size - len(next_gen)]:
                    next_gen.append(copy.deepcopy(combined[idx]))

        return next_gen

    def step(self, population: List[Architecture],
             offspring_pop: List[Architecture]) -> List[Architecture]:
        """执行一代进化"""
        return self.select_next_generation(population, offspring_pop)


class RandomSearch:
    """
    随机搜索基线
    """
    def __init__(self, num_nodes: int = 6, enabled_ops: Optional[List[str]] = None, pop_size: int = 100):
        self.num_nodes = num_nodes
        self.enabled_ops = enabled_ops or ['conv3x3', 'conv5x5', 'dil_conv3x3',
                                           'max_pool3x3', 'avg_pool3x3', 'skip_connect']
        self.pop_size = pop_size

    def initialize_population(self) -> List[Architecture]:
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            arch = Architecture(self.num_nodes, self.enabled_ops.copy())
            _, arch = validate_architecture(arch, fix=True)
            population.append(arch)
        return population

    def step(self, *args, **kwargs) -> List[Architecture]:
        """随机生成新一代"""
        return self.initialize_population()
