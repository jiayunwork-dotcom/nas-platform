"""
代理模型模块
使用3层MLP预筛候选架构，减少昂贵评估次数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import copy

from .cell import Architecture
from .metrics import fast_non_dominated_sort, dominates


class ArchDataset(Dataset):
    """架构数据集"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SurrogateMLP(nn.Module):
    """
    3层MLP代理模型
    输入: 架构编码向量
    输出: 预测的 [精度, 参数量, 延迟]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateModel:
    """
    代理模型包装类
    包含训练、预测、预筛功能
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-3, batch_size: int = 32, epochs: int = 100,
                 device: str = 'cpu', min_train_samples: int = 50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.min_train_samples = min_train_samples
        self.maximize = [True, False, False]

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.trained = False
        self.X_history = None
        self.y_history = None

    def _encode_architectures(self, archs: List[Architecture]) -> np.ndarray:
        """将架构列表编码为矩阵"""
        encodings = []
        for arch in archs:
            encodings.append(arch.encode())
        return np.array(encodings)

    def _get_targets(self, archs: List[Architecture]) -> np.ndarray:
        """获取目标值矩阵"""
        targets = []
        for arch in archs:
            targets.append([arch.accuracy, arch.params, arch.latency])
        return np.array(targets)

    def is_ready(self) -> bool:
        """检查是否有足够数据训练代理模型"""
        return self.X_history is not None and len(self.X_history) >= self.min_train_samples

    def train(self, evaluated_archs: List[Architecture]):
        """
        使用所有已评估数据训练代理模型
        """
        if len(evaluated_archs) < self.min_train_samples:
            self.trained = False
            return

        X = self._encode_architectures(evaluated_archs)
        y = self._get_targets(evaluated_archs)

        self.X_history = X
        self.y_history = y

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        dataset = ArchDataset(X_scaled, y_scaled)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SurrogateMLP(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.trained = True

    def predict(self, archs: List[Architecture]) -> np.ndarray:
        """
        预测架构的性能指标
        返回: [精度, 参数量, 延迟] 矩阵
        """
        if not self.trained:
            raise RuntimeError("Surrogate model not trained yet")

        self.model.eval()
        X = self._encode_architectures(archs)
        X_scaled = self.scaler_X.transform(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        y_pred[:, 0] = np.clip(y_pred[:, 0], 0.0, 1.0)
        y_pred[:, 1] = np.clip(y_pred[:, 1], 1e3, None)
        y_pred[:, 2] = np.clip(y_pred[:, 2], 0.0, None)

        return y_pred

    def _get_pareto_mask(self, points: np.ndarray) -> np.ndarray:
        """获取帕累托前沿掩码"""
        fronts = fast_non_dominated_sort(points, self.maximize)
        mask = np.zeros(len(points), dtype=bool)
        if fronts:
            mask[fronts[0]] = True
        return mask

    def pre_screen(self, candidate_archs: List[Architecture],
                   top_k: Optional[int] = None,
                   percentile: float = 30.0) -> List[Architecture]:
        """
        预筛候选架构
        只有代理预测位于帕累托前沿附近的架构才做真实评估

        Args:
            candidate_archs: 候选架构列表
            top_k: 如果指定，返回前K个最有希望的架构
            percentile: 帕累托前沿附近的百分比阈值（默认保留前30%）

        Returns:
            筛选后的架构列表
        """
        if not self.trained:
            return candidate_archs

        predictions = self.predict(candidate_archs)

        fronts = fast_non_dominated_sort(predictions, self.maximize)

        rank = np.zeros(len(candidate_archs), dtype=int)
        for front_idx, front in enumerate(fronts):
            for idx in front:
                rank[idx] = front_idx

        max_rank = int(np.percentile(rank, percentile))

        selected_indices = []
        for i in range(len(candidate_archs)):
            if rank[i] <= max_rank:
                selected_indices.append(i)

        if top_k is not None and len(selected_indices) > top_k:
            pareto_indices = [i for i in selected_indices if rank[i] == 0]
            other_indices = [i for i in selected_indices if rank[i] > 0]
            selected_indices = pareto_indices + other_indices[:top_k - len(pareto_indices)]

        selected = [copy.deepcopy(candidate_archs[i]) for i in selected_indices]
        return selected

    def get_prediction_error(self, archs: List[Architecture]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算预测误差（用于诊断）
        返回: (MAE, RMSE) 每个目标维度
        """
        if not self.trained:
            return np.zeros(3), np.zeros(3)

        predictions = self.predict(archs)
        targets = self._get_targets(archs)

        mae = np.mean(np.abs(predictions - targets), axis=0)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))

        return mae, rmse
