"""
搜索空间定义模块
Cell-based搜索空间: Normal Cell和Reduction Cell
编码方式: 邻接矩阵 + 操作列表
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# 可用操作定义
AVAILABLE_OPS = {
    'conv3x3': lambda C_in, C_out, stride: nn.Sequential(
        nn.Conv2d(C_in, C_out, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True)
    ),
    'conv5x5': lambda C_in, C_out, stride: nn.Sequential(
        nn.Conv2d(C_in, C_out, 5, stride, padding=2, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True)
    ),
    'dil_conv3x3': lambda C_in, C_out, stride: nn.Sequential(
        nn.Conv2d(C_in, C_out, 3, stride, padding=2, dilation=2, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True)
    ),
    'max_pool3x3': lambda C_in, C_out, stride: nn.Sequential(
        nn.MaxPool2d(3, stride, padding=1),
        nn.BatchNorm2d(C_out) if stride == 1 else nn.Identity()
    ),
    'avg_pool3x3': lambda C_in, C_out, stride: nn.Sequential(
        nn.AvgPool2d(3, stride, padding=1, count_include_pad=False),
        nn.BatchNorm2d(C_out) if stride == 1 else nn.Identity()
    ),
    'skip_connect': lambda C_in, C_out, stride: nn.Identity() if (C_in == C_out and stride == 1) else nn.Sequential(
        nn.Conv2d(C_in, C_out, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(C_out)
    ),
    'zero': lambda C_in, C_out, stride: Zero(stride),
}

OP_NAMES = ['conv3x3', 'conv5x5', 'dil_conv3x3', 'max_pool3x3', 'avg_pool3x3', 'skip_connect', 'zero']
OP_COLORS = {
    'conv3x3': '#FF6B6B',
    'conv5x5': '#4ECDC4',
    'dil_conv3x3': '#45B7D1',
    'max_pool3x3': '#96CEB4',
    'avg_pool3x3': '#FFEAA7',
    'skip_connect': '#DDA0DD',
    'zero': '#95A5A6',
}


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        return torch.zeros(n, c, h // self.stride, w // self.stride, device=x.device)


class OpChoice(nn.Module):
    """可选择的操作集合，用于权重共享超网"""
    def __init__(self, C_in, C_out, stride, enabled_ops=None):
        super().__init__()
        self._ops = nn.ModuleDict()
        self.enabled_ops = enabled_ops if enabled_ops else OP_NAMES[:-1]  # 默认不包含zero
        for op_name in self.enabled_ops:
            op = AVAILABLE_OPS[op_name](C_in, C_out, stride)
            self._ops[op_name] = op

    def forward(self, x, op_idx):
        op_name = self.enabled_ops[op_idx]
        return self._ops[op_name](x)


class Cell(nn.Module):
    """
    Cell基类
    节点编号: 0=输入1, 1=输入2, 2..N-2=中间节点, N-1=输出
    """
    def __init__(self, num_nodes: int, C: int, reduction: bool, enabled_ops: List[str]):
        super().__init__()
        self.num_nodes = num_nodes
        self.C = C
        self.reduction = reduction
        self.enabled_ops = enabled_ops
        self._preprocess0 = None
        self._preprocess1 = None
        self._ops = nn.ModuleList()

    def build_edges(self, adjacency: np.ndarray, op_list: List[int], C_in_prev_prev: int, C_in_prev: int):
        """根据邻接矩阵和操作列表构建边"""
        stride = 2 if self.reduction else 1
        C_out = self.C

        self._preprocess0 = ConvBnRelu(C_in_prev_prev, C_out, 1, 1, 0)
        self._preprocess1 = ConvBnRelu(C_in_prev, C_out, 1, 1, 0)

        self._ops = nn.ModuleList()
        op_idx_ptr = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if adjacency[i, j]:
                    stride_edge = stride if (i < 2 and self.reduction) else 1
                    if op_idx_ptr < len(op_list):
                        op_name = self.enabled_ops[op_list[op_idx_ptr]]
                        op = AVAILABLE_OPS[op_name](C_out, C_out, stride_edge)
                        self._ops.append(op)
                    op_idx_ptr += 1

    def forward(self, s0, s1, adjacency: np.ndarray):
        s0 = self._preprocess0(s0)
        s1 = self._preprocess1(s1)

        states = [s0, s1]
        op_idx = 0
        for j in range(2, self.num_nodes - 1):
            node_inputs = []
            for i in range(j):
                if adjacency[i, j]:
                    if op_idx < len(self._ops):
                        h = self._ops[op_idx](states[i])
                        node_inputs.append(h)
                    op_idx += 1
            if len(node_inputs) == 0:
                node_inputs.append(torch.zeros_like(states[0]))
            s_j = sum(node_inputs)
            states.append(s_j)

        output_inputs = []
        j_out = self.num_nodes - 1
        for i in range(j_out):
            if adjacency[i, j_out]:
                if op_idx < len(self._ops):
                    h = self._ops[op_idx](states[i])
                    output_inputs.append(h)
                op_idx += 1
        if len(output_inputs) == 0:
            output_inputs.append(torch.zeros_like(states[0]))
        return sum(output_inputs)


class ConvBnRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class NASNetwork(nn.Module):
    """
    由Cell堆叠而成的完整网络
    """
    def __init__(self, num_classes: int, num_cells: int, num_nodes: int,
                 init_channels: int, enabled_ops: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.init_channels = init_channels
        self.enabled_ops = enabled_ops

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels * 3)
        )

        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = init_channels * 3, init_channels * 3, init_channels

        for cell_idx in range(num_cells):
            reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
            if reduction:
                C_curr *= 2
            cell = Cell(num_nodes, C_curr, reduction, enabled_ops)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def build_network(self, normal_adj: np.ndarray, normal_ops: List[int],
                      reduce_adj: np.ndarray, reduce_ops: List[int]):
        """根据架构编码构建完整网络"""
        C_prev_prev, C_prev, C_curr = self.init_channels * 3, self.init_channels * 3, self.init_channels

        for cell_idx, cell in enumerate(self.cells):
            reduction = cell_idx in [self.num_cells // 3, 2 * self.num_cells // 3]
            if reduction:
                C_curr *= 2
                cell.build_edges(reduce_adj, reduce_ops, C_prev_prev, C_prev)
            else:
                cell.build_edges(normal_adj, normal_ops, C_prev_prev, C_prev)
            C_prev_prev, C_prev = C_prev, C_curr

    def forward(self, x, normal_adj: np.ndarray, reduce_adj: np.ndarray):
        s0 = s1 = self.stem(x)
        for cell_idx, cell in enumerate(self.cells):
            reduction = cell_idx in [self.num_cells // 3, 2 * self.num_cells // 3]
            adj = reduce_adj if reduction else normal_adj
            s0, s1 = s1, cell(s0, s1, adj)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Architecture:
    """
    架构编码类
    包含: Normal Cell和Reduction Cell的邻接矩阵和操作列表
    """
    def __init__(self, num_nodes: int, enabled_ops: List[str],
                 normal_adj: Optional[np.ndarray] = None,
                 normal_op_list: Optional[List[int]] = None,
                 reduce_adj: Optional[np.ndarray] = None,
                 reduce_op_list: Optional[List[int]] = None):
        self.num_nodes = num_nodes
        self.enabled_ops = enabled_ops
        self.num_ops = len(enabled_ops)

        if normal_adj is None:
            self.normal_adj = self._random_adjacency()
            self.normal_op_list = self._random_op_list(self.normal_adj)
        else:
            self.normal_adj = normal_adj
            self.normal_op_list = normal_op_list if normal_op_list else self._random_op_list(normal_adj)

        if reduce_adj is None:
            self.reduce_adj = self._random_adjacency()
            self.reduce_op_list = self._random_op_list(self.reduce_adj)
        else:
            self.reduce_adj = reduce_adj
            self.reduce_op_list = reduce_op_list if reduce_op_list else self._random_op_list(reduce_adj)

        self.accuracy = None
        self.params = None
        self.latency = None

    def _random_adjacency(self) -> np.ndarray:
        """生成随机邻接矩阵（上三角）"""
        n = self.num_nodes
        adj = np.zeros((n, n), dtype=bool)
        for j in range(2, n):
            num_inputs = np.random.randint(1, min(j, 3) + 1)
            possible_inputs = list(range(j))
            selected = np.random.choice(possible_inputs, num_inputs, replace=False)
            for i in selected:
                adj[i, j] = True
        for i in range(n - 1):
            adj[i, n - 1] = np.random.random() > 0.5
        return adj

    def _random_op_list(self, adj: np.ndarray) -> List[int]:
        """根据邻接矩阵生成随机操作列表"""
        num_edges = int(adj.sum())
        return np.random.randint(0, self.num_ops, num_edges).tolist()

    def encode(self) -> np.ndarray:
        """将架构展平为一维向量用于代理模型"""
        def flatten(adj, op_list):
            adj_flat = adj.astype(int).flatten()
            op_padded = np.zeros(self.num_nodes * (self.num_nodes - 1) // 2, dtype=int)
            idx = 0
            op_idx = 0
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if adj[i, j]:
                        op_padded[idx] = op_list[op_idx] + 1 if op_idx < len(op_list) else 0
                        op_idx += 1
                    idx += 1
            return np.concatenate([adj_flat, op_padded])

        norm = flatten(self.normal_adj, self.normal_op_list)
        red = flatten(self.reduce_adj, self.reduce_op_list)
        return np.concatenate([norm, red])

    def copy(self) -> 'Architecture':
        """复制架构"""
        return Architecture(
            self.num_nodes, self.enabled_ops.copy(),
            self.normal_adj.copy(), self.normal_op_list.copy(),
            self.reduce_adj.copy(), self.reduce_op_list.copy()
        )

    def __repr__(self):
        return f"Arch(acc={self.accuracy:.3f}, params={self.params/1e6:.2f}M, latency={self.latency:.3f}ms)"


def get_op_list_from_adj(adj: np.ndarray, op_list: List[int], i: int, j: int) -> Optional[int]:
    """获取从节点i到节点j的操作索引"""
    if not adj[i, j]:
        return None
    idx = 0
    for ii in range(adj.shape[0]):
        for jj in range(ii + 1, adj.shape[1]):
            if adj[ii, jj]:
                if ii == i and jj == j:
                    return op_list[idx] if idx < len(op_list) else None
                idx += 1
    return None


def set_op_in_list(adj: np.ndarray, op_list: List[int], i: int, j: int, new_op: int) -> List[int]:
    """设置从节点i到节点j的操作"""
    idx = 0
    new_list = op_list.copy()
    for ii in range(adj.shape[0]):
        for jj in range(ii + 1, adj.shape[1]):
            if adj[ii, jj]:
                if ii == i and jj == j:
                    if idx < len(new_list):
                        new_list[idx] = new_op
                idx += 1
    return new_list
