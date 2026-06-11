"""
DAG有效性检查和修复模块
确保Cell的有向无环图满足约束条件
"""

import numpy as np
import networkx as nx
from typing import Tuple, List
from collections import deque

from .cell import Architecture


def has_cycle(adj: np.ndarray) -> bool:
    """检查邻接矩阵表示的图是否有环"""
    n = adj.shape[0]
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(node):
        visited[node] = True
        rec_stack[node] = True
        for j in range(n):
            if adj[node, j]:
                if not visited[j]:
                    if dfs(j):
                        return True
                elif rec_stack[j]:
                    return True
        rec_stack[node] = False
        return False

    for i in range(n):
        if not visited[i]:
            if dfs(i):
                return True
    return False


def topological_sort(adj: np.ndarray) -> List[int]:
    """拓扑排序，返回节点顺序"""
    n = adj.shape[0]
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                in_degree[j] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for j in range(n):
            if adj[node, j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    return result


def is_output_reachable(adj: np.ndarray) -> bool:
    """检查输出节点是否可从所有中间节点到达"""
    n = adj.shape[0]
    output_node = n - 1
    intermediate_nodes = list(range(2, n - 1))

    def reachable_from(node):
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current == output_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for j in range(n):
                if adj[current, j] and j not in visited:
                    stack.append(j)
        return False

    for node in intermediate_nodes:
        if not reachable_from(node):
            return False
    return True


def has_isolated_nodes(adj: np.ndarray) -> Tuple[bool, List[int]]:
    """检查是否有孤立节点（没有入边或没有出边的中间节点）"""
    n = adj.shape[0]
    intermediate_nodes = list(range(2, n - 1))
    isolated = []

    for node in intermediate_nodes:
        in_edges = np.any(adj[:, node])
        out_edges = np.any(adj[node, :])
        if not in_edges or not out_edges:
            isolated.append(node)

    return len(isolated) > 0, isolated


def fix_isolated_nodes(adj: np.ndarray, op_list: List[int], num_ops: int) -> Tuple[np.ndarray, List[int]]:
    """修复孤立节点：随机连接到前驱或后继"""
    n = adj.shape[0]
    adj = adj.copy()
    op_list = op_list.copy()

    intermediate_nodes = list(range(2, n - 1))

    for node in intermediate_nodes:
        in_edges = np.any(adj[:, node])
        out_edges = np.any(adj[node, :])

        if not in_edges:
            possible_preds = list(range(node))
            if possible_preds:
                pred = np.random.choice(possible_preds)
                adj[pred, node] = True
                op_list.append(np.random.randint(0, num_ops))

        if not out_edges:
            possible_succs = list(range(node + 1, n))
            if possible_succs:
                succ = np.random.choice(possible_succs)
                adj[node, succ] = True
                op_list.append(np.random.randint(0, num_ops))

    return adj, op_list


def ensure_output_reachable(adj: np.ndarray, op_list: List[int], num_ops: int) -> Tuple[np.ndarray, List[int]]:
    """确保所有中间节点都能到达输出节点"""
    n = adj.shape[0]
    output_node = n - 1
    adj = adj.copy()
    op_list = op_list.copy()

    intermediate_nodes = list(range(2, n - 1))

    def can_reach_output(node):
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current == output_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for j in range(n):
                if adj[current, j] and j not in visited:
                    stack.append(j)
        return False

    for node in sorted(intermediate_nodes, reverse=True):
        if not can_reach_output(node):
            possible_succs = list(range(node + 1, n))
            for succ in sorted(possible_succs, reverse=True):
                if adj[node, succ]:
                    continue
                if succ == output_node or can_reach_output(succ):
                    adj[node, succ] = True
                    op_list.append(np.random.randint(0, num_ops))
                    break

    return adj, op_list


def remove_cycles(adj: np.ndarray, op_list: List[int]) -> Tuple[np.ndarray, List[int]]:
    """移除环（删除反向边）"""
    n = adj.shape[0]
    adj = adj.copy()
    op_list = op_list.copy()

    removed_edges = []

    while has_cycle(adj):
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                if adj[j, i]:
                    adj[j, i] = False
                    removed_edges.append((j, i))
                    found = True
                    break
            if found:
                break

    if removed_edges:
        new_op_list = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    if idx < len(op_list):
                        new_op_list.append(op_list[idx])
                    idx += 1
        op_list = new_op_list

    return adj, op_list


def enforce_dag_constraints(adj: np.ndarray, op_list: List[int], num_ops: int) -> Tuple[np.ndarray, List[int]]:
    """强制执行所有DAG约束"""
    adj = adj.copy()
    op_list = op_list.copy()

    adj, op_list = remove_cycles(adj, op_list)
    adj, op_list = fix_isolated_nodes(adj, op_list, num_ops)
    adj, op_list = ensure_output_reachable(adj, op_list, num_ops)

    return adj, op_list


def validate_architecture(arch: Architecture, fix: bool = True) -> Tuple[bool, Architecture]:
    """验证架构的有效性，可选自动修复"""
    valid = True

    for name, adj, op_list in [
        ('normal', arch.normal_adj, arch.normal_op_list),
        ('reduce', arch.reduce_adj, arch.reduce_op_list)
    ]:
        if has_cycle(adj):
            valid = False
        if not is_output_reachable(adj):
            valid = False
        isolated, _ = has_isolated_nodes(adj)
        if isolated:
            valid = False

    if not valid and fix:
        arch.normal_adj, arch.normal_op_list = enforce_dag_constraints(
            arch.normal_adj, arch.normal_op_list, arch.num_ops
        )
        arch.reduce_adj, arch.reduce_op_list = enforce_dag_constraints(
            arch.reduce_adj, arch.reduce_op_list, arch.num_ops
        )
        valid = True

    return valid, arch


def to_networkx_graph(adj: np.ndarray, op_list: List[int], enabled_ops: List[str]) -> nx.DiGraph:
    """将邻接矩阵和操作列表转换为networkx图"""
    n = adj.shape[0]
    G = nx.DiGraph()

    node_names = ['in1', 'in2']
    for i in range(2, n - 1):
        node_names.append(f'n{i-1}')
    node_names.append('out')

    for i, name in enumerate(node_names):
        G.add_node(name, label=name)

    op_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and op_idx < len(op_list):
                op_name = enabled_ops[op_list[op_idx]]
                G.add_edge(node_names[i], node_names[j], op=op_name)
                op_idx += 1

    return G
