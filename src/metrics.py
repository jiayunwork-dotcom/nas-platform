"""
性能指标计算模块
包含: 参数量估算、推理延迟估算、FLOPs计算、超体积指标
"""

import numpy as np
from typing import List, Tuple, Dict
from functools import reduce

from .cell import Architecture, OP_NAMES

DEFAULT_LATENCY_TABLE = {
    'conv3x3': 0.12,
    'conv5x5': 0.25,
    'dil_conv3x3': 0.15,
    'max_pool3x3': 0.02,
    'avg_pool3x3': 0.02,
    'skip_connect': 0.001,
    'zero': 0.0,
}


def count_conv_params(kernel_size: int, in_channels: int, out_channels: int, bias: bool = False) -> int:
    """计算卷积层参数量"""
    params = kernel_size * kernel_size * in_channels * out_channels
    if bias:
        params += out_channels
    return params


def count_bn_params(channels: int) -> int:
    """计算BN层参数量"""
    return 2 * channels


def count_op_params(op_name: str, in_channels: int, out_channels: int, stride: int) -> int:
    """计算单个操作的参数量"""
    if op_name == 'conv3x3':
        return count_conv_params(3, in_channels, out_channels) + count_bn_params(out_channels)
    elif op_name == 'conv5x5':
        return count_conv_params(5, in_channels, out_channels) + count_bn_params(out_channels)
    elif op_name == 'dil_conv3x3':
        return count_conv_params(3, in_channels, out_channels) + count_bn_params(out_channels)
    elif op_name == 'max_pool3x3':
        return count_bn_params(out_channels) if stride == 1 else 0
    elif op_name == 'avg_pool3x3':
        return count_bn_params(out_channels) if stride == 1 else 0
    elif op_name == 'skip_connect':
        if in_channels == out_channels and stride == 1:
            return 0
        return count_conv_params(1, in_channels, out_channels) + count_bn_params(out_channels)
    elif op_name == 'zero':
        return 0
    return 0


def count_cell_params(adj: np.ndarray, op_list: List[int], enabled_ops: List[str],
                      in_channels: int, out_channels: int, reduction: bool) -> int:
    """计算单个Cell的参数量"""
    total_params = 0
    n = adj.shape[0]

    preprocess_params = 2 * count_conv_params(1, in_channels, out_channels) + 2 * count_bn_params(out_channels)
    total_params += preprocess_params

    op_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and op_idx < len(op_list):
                op_name = enabled_ops[op_list[op_idx]]
                stride = 2 if (i < 2 and reduction) else 1
                total_params += count_op_params(op_name, out_channels, out_channels, stride)
                op_idx += 1

    return total_params


def estimate_architecture_params(arch: Architecture, num_cells: int = 20, init_channels: int = 36) -> int:
    """估算完整网络的参数量"""
    total_params = 0
    stem_params = count_conv_params(3, 3, init_channels * 3) + count_bn_params(init_channels * 3)
    total_params += stem_params

    C_prev_prev, C_prev, C_curr = init_channels * 3, init_channels * 3, init_channels

    for cell_idx in range(num_cells):
        reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
        if reduction:
            C_curr *= 2
        adj = arch.reduce_adj if reduction else arch.normal_adj
        op_list = arch.reduce_op_list if reduction else arch.normal_op_list
        total_params += count_cell_params(adj, op_list, arch.enabled_ops, C_prev_prev, C_curr, reduction)
        C_prev_prev, C_prev = C_prev, C_curr

    total_params += C_prev * 10

    arch.params = total_params
    return total_params


def estimate_op_latency(op_name: str, latency_table: Dict[str, float] = None) -> float:
    """估算单个操作的延迟"""
    latency_table = latency_table or DEFAULT_LATENCY_TABLE
    return latency_table.get(op_name, 0.0)


def estimate_cell_latency(adj: np.ndarray, op_list: List[int], enabled_ops: List[str],
                          latency_table: Dict[str, float] = None) -> float:
    """估算单个Cell的延迟"""
    total_latency = 0.0
    n = adj.shape[0]
    op_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and op_idx < len(op_list):
                op_name = enabled_ops[op_list[op_idx]]
                total_latency += estimate_op_latency(op_name, latency_table)
                op_idx += 1
    return total_latency


def estimate_architecture_latency(arch: Architecture, num_cells: int = 20,
                                  latency_table: Dict[str, float] = None) -> float:
    """估算完整网络的单张图片推理延迟"""
    total_latency = 0.0
    stem_latency = DEFAULT_LATENCY_TABLE['conv3x3'] + 0.01
    total_latency += stem_latency

    for cell_idx in range(num_cells):
        reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
        adj = arch.reduce_adj if reduction else arch.normal_adj
        op_list = arch.reduce_op_list if reduction else arch.normal_op_list
        total_latency += estimate_cell_latency(adj, op_list, arch.enabled_ops, latency_table)

    classifier_latency = 0.01
    total_latency += classifier_latency

    arch.latency = total_latency
    return total_latency


def count_conv_flops(kernel_size: int, in_channels: int, out_channels: int,
                     height: int, width: int, stride: int = 1) -> int:
    """计算卷积层FLOPs"""
    out_h = height // stride
    out_w = width // stride
    return kernel_size * kernel_size * in_channels * out_channels * out_h * out_w


def count_op_flops(op_name: str, in_channels: int, out_channels: int,
                   height: int, width: int, stride: int) -> int:
    """计算单个操作的FLOPs"""
    if op_name == 'conv3x3':
        return count_conv_flops(3, in_channels, out_channels, height, width, stride)
    elif op_name == 'conv5x5':
        return count_conv_flops(5, in_channels, out_channels, height, width, stride)
    elif op_name == 'dil_conv3x3':
        return count_conv_flops(3, in_channels, out_channels, height, width, stride)
    elif op_name in ['max_pool3x3', 'avg_pool3x3']:
        out_h = height // stride
        out_w = width // stride
        return in_channels * out_h * out_w
    elif op_name == 'skip_connect':
        if in_channels == out_channels and stride == 1:
            return 0
        return count_conv_flops(1, in_channels, out_channels, height, width, stride)
    elif op_name == 'zero':
        return 0
    return 0


def count_architecture_flops(arch: Architecture, num_cells: int = 20,
                             init_channels: int = 36, input_size: int = 32) -> int:
    """计算完整网络的FLOPs"""
    total_flops = 0
    h, w = input_size, input_size
    stem_flops = count_conv_flops(3, 3, init_channels * 3, h, w)
    total_flops += stem_flops

    C_prev_prev, C_prev, C_curr = init_channels * 3, init_channels * 3, init_channels

    for cell_idx in range(num_cells):
        reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
        if reduction:
            C_curr *= 2
            h, w = h // 2, w // 2
        adj = arch.reduce_adj if reduction else arch.normal_adj
        op_list = arch.reduce_op_list if reduction else arch.normal_op_list

        preprocess_flops = 2 * count_conv_flops(1, C_prev_prev, C_curr, h, w)
        total_flops += preprocess_flops

        n = adj.shape[0]
        op_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] and op_idx < len(op_list):
                    op_name = arch.enabled_ops[op_list[op_idx]]
                    stride = 2 if (i < 2 and reduction) else 1
                    total_flops += count_op_flops(op_name, C_curr, C_curr, h, w, stride)
                    op_idx += 1
        C_prev_prev, C_prev = C_prev, C_curr

    classifier_flops = C_prev * 10
    total_flops += classifier_flops

    return total_flops


def dominates(p1: np.ndarray, p2: np.ndarray, maximize: List[bool]) -> bool:
    """判断p1是否支配p2"""
    assert len(p1) == len(p2) == len(maximize)
    better = False
    for i in range(len(p1)):
        if maximize[i]:
            if p1[i] < p2[i]:
                return False
            if p1[i] > p2[i]:
                better = True
        else:
            if p1[i] > p2[i]:
                return False
            if p1[i] < p2[i]:
                better = True
    return better


def fast_non_dominated_sort(points: np.ndarray, maximize: List[bool]) -> List[List[int]]:
    """
    快速非支配排序算法
    时间复杂度: O(M*N^2), M为目标数, N为种群规模
    返回: 每个前沿的索引列表
    """
    N = len(points)
    S = [[] for _ in range(N)]
    n = [0] * N
    rank = [0] * N
    fronts = [[]]

    for p in range(N):
        S[p] = []
        n[p] = 0
        for q in range(N):
            if dominates(points[p], points[q], maximize):
                S[p].append(q)
            elif dominates(points[q], points[p], maximize):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def crowding_distance(points: np.ndarray, front: List[int]) -> np.ndarray:
    """
    计算拥挤距离
    各目标维度独立归一化后求和
    """
    M = points.shape[1]
    N = len(front)
    distances = np.zeros(N)

    for m in range(M):
        sorted_idx = sorted(range(N), key=lambda i: points[front[i], m])
        min_val = points[front[sorted_idx[0]], m]
        max_val = points[front[sorted_idx[-1]], m]

        if max_val == min_val:
            continue

        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        for i in range(1, N - 1):
            distances[sorted_idx[i]] += (
                (points[front[sorted_idx[i + 1]], m] - points[front[sorted_idx[i - 1]], m])
                / (max_val - min_val)
            )

    return distances


def get_pareto_front(points: np.ndarray, maximize: List[bool]) -> np.ndarray:
    """获取帕累托前沿点"""
    fronts = fast_non_dominated_sort(points, maximize)
    if fronts:
        return points[fronts[0]]
    return np.array([])


def hypervolume(points: np.ndarray, reference_point: np.ndarray, maximize: List[bool]) -> float:
    """
    计算超体积指标
    使用简单的递归实现（适用于小规模点集）
    """
    if len(points) == 0:
        return 0.0

    M = len(reference_point)

    transformed = points.copy()
    for m in range(M):
        if maximize[m]:
            transformed[:, m] = -transformed[:, m]

    ref_transformed = reference_point.copy()
    for m in range(M):
        if maximize[m]:
            ref_transformed[m] = -ref_transformed[m]

    dominated = np.all(transformed <= ref_transformed, axis=1)
    transformed = transformed[dominated]

    if len(transformed) == 0:
        return 0.0

    if M == 2:
        return _hypervolume_2d(transformed, ref_transformed)
    else:
        return _hypervolume_recursive(transformed, ref_transformed)


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """二维超体积计算"""
    sorted_idx = np.argsort(points[:, 0])
    points = points[sorted_idx]

    volume = 0.0
    prev_y = ref[1]

    for i in range(len(points)):
        width = ref[0] - points[i, 0]
        if width > 0 and prev_y > points[i, 1]:
            height = prev_y - points[i, 1]
            volume += width * height
            prev_y = points[i, 1]

    return volume


def _hypervolume_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    """递归超体积计算"""
    if len(points) == 0:
        return 0.0

    M = len(ref)

    non_dominated = []
    for i in range(len(points)):
        dominated = False
        for j in range(len(points)):
            if i != j and np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)

    points = points[non_dominated]

    if len(points) == 0:
        return 0.0

    if M == 1:
        return max(0, ref[0] - np.min(points[:, 0]))

    sorted_idx = np.argsort(points[:, -1])
    points = points[sorted_idx]

    volume = 0.0
    current_hyperplane = ref[-1]

    for i in range(len(points) - 1, -1, -1):
        if points[i, -1] < current_hyperplane:
            height = current_hyperplane - points[i, -1]
            sub_points = points[:i + 1, :-1]
            sub_ref = ref[:-1]
            sub_vol = _hypervolume_recursive(sub_points, sub_ref)
            volume += height * sub_vol
            current_hyperplane = points[i, -1]

    return volume


def get_reference_point(points: np.ndarray, maximize: List[bool], scale: float = 1.1) -> np.ndarray:
    """
    获取参考点（各目标最差值的scale倍）
    """
    M = len(maximize)
    ref = np.zeros(M)
    for m in range(M):
        if maximize[m]:
            ref[m] = np.min(points[:, m]) / scale if np.min(points[:, m]) > 0 else np.min(points[:, m]) * scale
        else:
            ref[m] = np.max(points[:, m]) * scale
    return ref
