"""
评估策略模块
包含: 完整训练、SynFlow零代价代理、NASWOT零代价代理、权重共享
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import List, Dict, Optional, Callable, Tuple
from tqdm import tqdm
import time

from .cell import Architecture, NASNetwork
from .metrics import estimate_architecture_params, estimate_architecture_latency


class Evaluator:
    """评估器基类"""
    def __init__(self, num_classes: int = 10, num_cells: int = 8,
                 init_channels: int = 16, device: str = 'cpu'):
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.init_channels = init_channels
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def evaluate(self, arch: Architecture) -> Architecture:
        """评估架构，设置accuracy、params、latency属性"""
        raise NotImplementedError


class FullTrainingEvaluator(Evaluator):
    """
    完整训练评估
    在CIFAR-10上训练指定epoch数后取验证精度
    """
    def __init__(self, num_classes: int = 10, num_cells: int = 8,
                 init_channels: int = 16, device: str = 'cpu',
                 epochs: int = 20, batch_size: int = 128, lr: float = 0.025,
                 data_dir: str = './data'):
        super().__init__(num_classes, num_cells, init_channels, device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_dir = data_dir
        self._data_loaded = False
        self.train_loader = None
        self.val_loader = None

    def _load_data(self):
        """加载CIFAR-10数据集"""
        if self._data_loaded:
            return

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_train = datasets.CIFAR10(root=self.data_dir, train=True,
                                      download=True, transform=transform_train)
        train_size = int(0.9 * len(full_train))
        val_size = len(full_train) - train_size
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
        val_dataset.dataset.transform = transform_val

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=2)
        self._data_loaded = True

    def evaluate(self, arch: Architecture) -> Architecture:
        """完整训练并评估"""
        self._load_data()

        model = NASNetwork(
            num_classes=self.num_classes,
            num_cells=self.num_cells,
            num_nodes=arch.num_nodes,
            init_channels=self.init_channels,
            enabled_ops=arch.enabled_ops
        )
        model.build_network(arch.normal_adj, arch.normal_op_list,
                            arch.reduce_adj, arch.reduce_op_list)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        best_acc = 0.0
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs, arch.normal_adj, arch.reduce_adj)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs, arch.normal_adj, arch.reduce_adj)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total
            if val_acc > best_acc:
                best_acc = val_acc

        arch.accuracy = best_acc
        arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
        arch.latency = estimate_architecture_latency(arch, self.num_cells)

        return arch


class SynFlowEvaluator(Evaluator):
    """
    SynFlow零代价代理指标
    计算网络参数对损失的路径归一化灵敏度
    """
    def __init__(self, num_classes: int = 10, num_cells: int = 8,
                 init_channels: int = 16, device: str = 'cpu'):
        super().__init__(num_classes, num_cells, init_channels, device)

    def evaluate(self, arch: Architecture) -> Architecture:
        """SynFlow评估"""
        model = NASNetwork(
            num_classes=self.num_classes,
            num_cells=self.num_cells,
            num_nodes=arch.num_nodes,
            init_channels=self.init_channels,
            enabled_ops=arch.enabled_ops
        )
        model.build_network(arch.normal_adj, arch.normal_op_list,
                            arch.reduce_adj, arch.reduce_op_list)
        model = model.to(self.device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = True

        input_dim = (1, 3, 32, 32)
        inputs = torch.ones(input_dim, device=self.device)

        def loss_fn(outputs):
            return torch.sum(outputs)

        signs = {}
        for name, param in model.named_parameters():
            signs[name] = torch.sign(param.data)
            param.data = torch.abs(param.data)

        outputs = model(inputs, arch.normal_adj, arch.reduce_adj)
        loss = loss_fn(outputs)
        loss.backward()

        synflow_score = 0.0
        total_params = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                synflow_score += torch.sum(torch.abs(param.data * param.grad)).item()
                total_params += param.data.numel()

        for name, param in model.named_parameters():
            param.data = signs[name] * param.data

        normalized_score = synflow_score / total_params if total_params > 0 else 0

        arch.accuracy = min(0.95, max(0.1, normalized_score / 10.0))
        arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
        arch.latency = estimate_architecture_latency(arch, self.num_cells)

        return arch


class NASWOTEvaluator(Evaluator):
    """
    NASWOT零代价代理指标
    计算训练集mini-batch上网络内核矩阵的对数行列式
    """
    def __init__(self, num_classes: int = 10, num_cells: int = 8,
                 init_channels: int = 16, device: str = 'cpu',
                 batch_size: int = 64, data_dir: str = './data'):
        super().__init__(num_classes, num_cells, init_channels, device)
        self.batch_size = batch_size
        self.data_dir = data_dir
        self._data_loaded = False
        self.sample_loader = None

    def _load_data(self):
        """加载CIFAR-10样本"""
        if self._data_loaded:
            return

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = datasets.CIFAR10(root=self.data_dir, train=True,
                                   download=True, transform=transform)

        subset = torch.utils.data.Subset(dataset, list(range(min(self.batch_size, len(dataset)))))
        self.sample_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        self._data_loaded = True

    def evaluate(self, arch: Architecture) -> Architecture:
        """NASWOT评估"""
        self._load_data()

        model = NASNetwork(
            num_classes=self.num_classes,
            num_cells=self.num_cells,
            num_nodes=arch.num_nodes,
            init_channels=self.init_channels,
            enabled_ops=arch.enabled_ops
        )
        model.build_network(arch.normal_adj, arch.normal_op_list,
                            arch.reduce_adj, arch.reduce_op_list)
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            for inputs, _ in self.sample_loader:
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                features_list = []
                hooks = []

                def hook_fn(module, input, output):
                    features_list.append(output.view(batch_size, -1))

                for module in model.modules():
                    if isinstance(module, (nn.ReLU, nn.Identity)):
                        hooks.append(module.register_forward_hook(hook_fn))

                _ = model(inputs, arch.normal_adj, arch.reduce_adj)

                for hook in hooks:
                    hook.remove()

                if len(features_list) == 0:
                    arch.accuracy = 0.5
                    arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
                    arch.latency = estimate_architecture_latency(arch, self.num_cells)
                    return arch

                F = torch.cat(features_list, dim=1)

                M = F.size(1)
                F_centered = F - F.mean(dim=0, keepdim=True)

                K = torch.matmul(F_centered, F_centered.T) / M

                try:
                    eigenvalues = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(K.size(0), device=self.device))
                    eigenvalues = torch.clamp(eigenvalues, min=1e-10)
                    logdet = torch.sum(torch.log(eigenvalues)).item()
                except:
                    logdet = 0.0

                normalized_score = max(0.0, logdet) / 1000.0

                arch.accuracy = min(0.95, max(0.1, normalized_score / 10.0))
                arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
                arch.latency = estimate_architecture_latency(arch, self.num_cells)

                return arch


class WeightSharingEvaluator(Evaluator):
    """
    权重共享评估
    维护一个超网，候选架构作为子图继承超网权重
    """
    def __init__(self, num_classes: int = 10, num_cells: int = 8,
                 init_channels: int = 16, device: str = 'cpu',
                 pretrain_epochs: int = 10, batch_size: int = 128, lr: float = 0.025,
                 data_dir: str = './data'):
        super().__init__(num_classes, num_cells, init_channels, device)
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_dir = data_dir
        self.supernet = None
        self._supernet_trained = False
        self._data_loaded = False
        self.train_loader = None
        self.val_loader = None

    def _load_data(self):
        """加载CIFAR-10数据集"""
        if self._data_loaded:
            return

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_train = datasets.CIFAR10(root=self.data_dir, train=True,
                                      download=True, transform=transform_train)
        train_size = int(0.9 * len(full_train))
        val_size = len(full_train) - train_size
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
        val_dataset.dataset.transform = transform_val

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=2)
        self._data_loaded = True

    def _pretrain_supernet(self, num_nodes: int, enabled_ops: List[str]):
        """预训练超网"""
        if self._supernet_trained:
            return

        self._load_data()

        self.supernet = NASNetwork(
            num_classes=self.num_classes,
            num_cells=self.num_cells,
            num_nodes=num_nodes,
            init_channels=self.init_channels,
            enabled_ops=enabled_ops
        )

        full_adj = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
        full_ops = np.random.randint(0, len(enabled_ops), int(full_adj.sum())).tolist()
        self.supernet.build_network(full_adj, full_ops, full_adj, full_ops)
        self.supernet = self.supernet.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.supernet.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=3e-4)

        for epoch in range(self.pretrain_epochs):
            self.supernet.train()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.supernet(inputs, full_adj, full_adj)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        self._supernet_trained = True

    def evaluate(self, arch: Architecture) -> Architecture:
        """权重共享评估"""
        self._pretrain_supernet(arch.num_nodes, arch.enabled_ops)

        model = NASNetwork(
            num_classes=self.num_classes,
            num_cells=self.num_cells,
            num_nodes=arch.num_nodes,
            init_channels=self.init_channels,
            enabled_ops=arch.enabled_ops
        )
        model.build_network(arch.normal_adj, arch.normal_op_list,
                            arch.reduce_adj, arch.reduce_op_list)
        model = model.to(self.device)

        model.load_state_dict(self.supernet.state_dict(), strict=False)

        model.eval()
        self._load_data()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs, arch.normal_adj, arch.reduce_adj)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        arch.accuracy = val_correct / val_total
        arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
        arch.latency = estimate_architecture_latency(arch, self.num_cells)

        return arch


class FastProxyEvaluator(Evaluator):
    """
    快速代理评估（用于演示）
    基于架构特征的简单预测，秒级完成
    """
    def evaluate(self, arch: Architecture) -> Architecture:
        """快速评估"""
        norm_edges = int(arch.normal_adj.sum())
        red_edges = int(arch.reduce_adj.sum())
        total_edges = norm_edges + red_edges

        conv_count = 0
        for op in arch.normal_op_list + arch.reduce_op_list:
            op_name = arch.enabled_ops[op]
            if 'conv' in op_name:
                conv_count += 1

        edge_score = min(1.0, total_edges / 20.0)
        conv_score = min(1.0, conv_count / 15.0)

        base_acc = 0.3 + 0.4 * edge_score + 0.2 * conv_score
        noise = np.random.normal(0, 0.05)
        arch.accuracy = min(0.95, max(0.1, base_acc + noise))

        arch.params = estimate_architecture_params(arch, self.num_cells, self.init_channels)
        arch.latency = estimate_architecture_latency(arch, self.num_cells)

        return arch


def get_evaluator(eval_strategy: str, **kwargs) -> Evaluator:
    """
    根据策略名称获取评估器
    """
    evaluators = {
        'full': FullTrainingEvaluator,
        'synflow': SynFlowEvaluator,
        'naswot': NASWOTEvaluator,
        'weight_sharing': WeightSharingEvaluator,
        'fast': FastProxyEvaluator,
    }

    if eval_strategy not in evaluators:
        raise ValueError(f"Unknown evaluation strategy: {eval_strategy}. "
                         f"Available: {list(evaluators.keys())}")

    evaluator_class = evaluators[eval_strategy]

    import inspect
    sig = inspect.signature(evaluator_class.__init__)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return evaluator_class(**filtered_kwargs)
