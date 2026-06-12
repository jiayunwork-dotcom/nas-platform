"""
代理模型模块
使用3层MLP预筛候选架构，减少昂贵评估次数
扩展: Bootstrap不确定度估计、R²/MAPE计算、学习曲线
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from typing import List, Tuple, Optional, Dict
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
    使用LayerNorm替代BatchNorm以避免小batch问题
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
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

        self.prev_model_state = None
        self.prev_scaler_X_state = None
        self.prev_scaler_y_state = None
        self.prev_val_r2 = None
        self.current_val_r2 = None

        self.incremental_train_count = 0
        self.incremental_r2_history = []
        self.rollback_count = 0
        self.full_retrain_count = 0
        self.calibration_events = []

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

    def _save_model_state(self):
        """保存当前模型状态（用于回滚）"""
        if self.model is not None:
            self.prev_model_state = copy.deepcopy(self.model.state_dict())
            self.prev_scaler_X_state = copy.deepcopy(self.scaler_X)
            self.prev_scaler_y_state = copy.deepcopy(self.scaler_y)
            self.prev_val_r2 = self.current_val_r2

    def _rollback_model(self):
        """回滚到上一个模型状态"""
        if self.prev_model_state is not None:
            self.model.load_state_dict(self.prev_model_state)
            self.scaler_X = self.prev_scaler_X_state
            self.scaler_y = self.prev_scaler_y_state
            self.current_val_r2 = self.prev_val_r2
            self.rollback_count += 1
            return True
        return False

    def _compute_avg_val_r2(self, evaluated_archs: List[Architecture]) -> float:
        """计算验证集平均R²"""
        if len(evaluated_archs) < 10:
            return 0.0
        metrics = self.compute_validation_metrics(evaluated_archs, train_ratio=0.8)
        return float(np.mean(metrics['r2']))

    def train(self, evaluated_archs: List[Architecture]):
        """
        使用所有已评估数据训练代理模型（全量训练）
        """
        if len(evaluated_archs) < self.min_train_samples:
            self.trained = False
            return

        self._save_model_state()

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
        self.current_val_r2 = self._compute_avg_val_r2(evaluated_archs)
        self.full_retrain_count += 1

    def incremental_train(self, new_archs: List[Architecture], all_archs: List[Architecture]):
        """
        增量训练代理模型

        Args:
            new_archs: 新增的已评估架构
            all_archs: 所有已评估架构（用于计算验证R²）

        Returns:
            dict: 包含训练结果信息的字典
        """
        if not self.trained or self.model is None:
            self.train(all_archs)
            return {
                'type': 'full_train',
                'success': True,
                'val_r2': self.current_val_r2,
                'rollback': False
            }

        if len(new_archs) == 0:
            return {
                'type': 'skipped',
                'success': True,
                'val_r2': self.current_val_r2,
                'rollback': False
            }

        self._save_model_state()
        prev_r2 = self.current_val_r2 if self.current_val_r2 is not None else 0.0

        new_X = self._encode_architectures(new_archs)
        new_y = self._get_targets(new_archs)

        if self.X_history is not None and self.y_history is not None:
            self.X_history = np.vstack([self.X_history, new_X])
            self.y_history = np.vstack([self.y_history, new_y])
        else:
            self.X_history = new_X
            self.y_history = new_y

        all_X = self.X_history
        all_y = self.y_history

        X_scaled = self.scaler_X.fit_transform(all_X)
        y_scaled = self.scaler_y.fit_transform(all_y)

        dataset = ArchDataset(X_scaled, y_scaled)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        incremental_epochs = 20
        incremental_lr = self.lr * 0.1
        optimizer = optim.Adam(self.model.parameters(), lr=incremental_lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(incremental_epochs):
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

        new_val_r2 = self._compute_avg_val_r2(all_archs)
        self.current_val_r2 = new_val_r2

        rollback_triggered = False
        if prev_r2 > 0 and (prev_r2 - new_val_r2) / abs(prev_r2) > 0.10:
            self._rollback_model()
            rollback_triggered = True
            self.train(all_archs)
            event = {
                'type': 'rollback_and_retrain',
                'incremental_count': self.incremental_train_count,
                'prev_r2': prev_r2,
                'new_r2': new_val_r2,
                'reason': f'R²下降{(prev_r2 - new_val_r2) / abs(prev_r2) * 100:.1f}%>10%，触发回滚并重训'
            }
            self.calibration_events.append(event)
            self.full_retrain_count += 1
        else:
            self.incremental_train_count += 1
            self.incremental_r2_history.append(new_val_r2)
            event = {
                'type': 'incremental',
                'incremental_count': self.incremental_train_count,
                'prev_r2': prev_r2,
                'new_r2': new_val_r2,
                'reason': '增量训练成功'
            }
            self.calibration_events.append(event)

        return {
            'type': 'incremental',
            'success': True,
            'val_r2': self.current_val_r2,
            'rollback': rollback_triggered,
            'prev_r2': prev_r2,
            'new_r2': new_val_r2
        }

    def get_calibration_history(self) -> Dict:
        """
        获取在线校准历史

        Returns:
            包含增量训练次数和每次R²的字典
        """
        return {
            'incremental_count': self.incremental_train_count,
            'incremental_r2_history': self.incremental_r2_history,
            'rollback_count': self.rollback_count,
            'full_retrain_count': self.full_retrain_count,
            'calibration_events': self.calibration_events
        }

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

    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[SurrogateMLP, StandardScaler, StandardScaler]:
        """训练单个MLP模型"""
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train)

        dataset = ArchDataset(X_scaled, y_scaled)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = SurrogateMLP(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model, scaler_X, scaler_y

    def _predict_with_model(self, model: SurrogateMLP, scaler_X: StandardScaler,
                            scaler_y: StandardScaler, X: np.ndarray) -> np.ndarray:
        """使用指定模型进行预测"""
        model.eval()
        X_scaled = scaler_X.transform(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = model(X_tensor).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_pred[:, 0] = np.clip(y_pred[:, 0], 0.0, 1.0)
        y_pred[:, 1] = np.clip(y_pred[:, 1], 1e3, None)
        y_pred[:, 2] = np.clip(y_pred[:, 2], 0.0, None)
        return y_pred

    def split_train_val_by_time(self, evaluated_archs: List[Architecture],
                                 train_ratio: float = 0.8) -> Tuple[List[Architecture], List[Architecture]]:
        """
        按时间顺序分割训练集和验证集
        前train_ratio为训练集，后(1-train_ratio)为验证集
        """
        n = len(evaluated_archs)
        split_idx = int(n * train_ratio)
        train_archs = evaluated_archs[:split_idx]
        val_archs = evaluated_archs[split_idx:]
        return train_archs, val_archs

    def compute_validation_metrics(self, evaluated_archs: List[Architecture],
                                    train_ratio: float = 0.8) -> Dict:
        """
        计算代理模型在验证集上的R²和MAPE

        Returns:
            包含以下键的字典:
            - r2: [accuracy_r2, params_r2, latency_r2]
            - mape: [accuracy_mape, params_mape, latency_mape]
            - train_pred: 训练集预测值
            - train_true: 训练集真实值
            - val_pred: 验证集预测值
            - val_true: 验证集真实值
            - train_archs: 训练集架构
            - val_archs: 验证集架构
        """
        if len(evaluated_archs) < 10:
            return {
                'r2': np.zeros(3),
                'mape': np.zeros(3),
                'train_pred': np.array([]),
                'train_true': np.array([]),
                'val_pred': np.array([]),
                'val_true': np.array([]),
                'train_archs': [],
                'val_archs': []
            }

        train_archs, val_archs = self.split_train_val_by_time(evaluated_archs, train_ratio)

        if len(train_archs) < self.min_train_samples:
            return {
                'r2': np.zeros(3),
                'mape': np.zeros(3),
                'train_pred': np.array([]),
                'train_true': np.array([]),
                'val_pred': np.array([]),
                'val_true': np.array([]),
                'train_archs': train_archs,
                'val_archs': val_archs
            }

        X_train = self._encode_architectures(train_archs)
        y_train = self._get_targets(train_archs)
        X_val = self._encode_architectures(val_archs)
        y_val = self._get_targets(val_archs)

        model, scaler_X, scaler_y = self._train_single_model(X_train, y_train)

        train_pred = self._predict_with_model(model, scaler_X, scaler_y, X_train)
        val_pred = self._predict_with_model(model, scaler_X, scaler_y, X_val)

        r2_scores = np.zeros(3)
        mape_scores = np.zeros(3)
        for dim in range(3):
            if len(y_val) > 1:
                try:
                    r2_scores[dim] = r2_score(y_val[:, dim], val_pred[:, dim])
                except:
                    r2_scores[dim] = 0.0
                try:
                    mape_scores[dim] = mean_absolute_percentage_error(y_val[:, dim], val_pred[:, dim]) * 100
                except:
                    mape_scores[dim] = 0.0

        return {
            'r2': r2_scores,
            'mape': mape_scores,
            'train_pred': train_pred,
            'train_true': y_train,
            'val_pred': val_pred,
            'val_true': y_val,
            'train_archs': train_archs,
            'val_archs': val_archs
        }

    def predict_with_uncertainty(self, archs: List[Architecture],
                                  n_bootstrap: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Bootstrap方法预测并估计不确定度

        Args:
            archs: 待预测的架构列表
            n_bootstrap: Bootstrap采样次数（默认10次）

        Returns:
            (predictions, uncertainties):
                predictions: 平均预测值 [N, 3]
                uncertainties: 预测标准差（作为不确定度估计） [N, 3]
        """
        if not self.trained or self.X_history is None or len(self.X_history) < self.min_train_samples:
            if self.trained:
                predictions = self.predict(archs)
                return predictions, np.zeros_like(predictions)
            return np.zeros((len(archs), 3)), np.zeros((len(archs), 3))

        X_query = self._encode_architectures(archs)
        n_samples = len(self.X_history)

        all_predictions = []

        for b in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_history[bootstrap_indices]
            y_boot = self.y_history[bootstrap_indices]

            try:
                model, scaler_X, scaler_y = self._train_single_model(X_boot, y_boot)
                pred = self._predict_with_model(model, scaler_X, scaler_y, X_query)
                all_predictions.append(pred)
            except:
                continue

        if len(all_predictions) == 0:
            predictions = self.predict(archs)
            return predictions, np.zeros_like(predictions)

        all_predictions = np.array(all_predictions)
        predictions = np.mean(all_predictions, axis=0)
        uncertainties = np.std(all_predictions, axis=0)

        return predictions, uncertainties

    def compute_learning_curve(self, evaluated_archs: List[Architecture],
                                train_ratio: float = 0.8,
                                min_samples: int = 10,
                                step: int = 10) -> Dict:
        """
        计算代理模型学习曲线
        x轴为训练样本数(从min_samples到全部训练数据)，y轴为验证集R²

        Args:
            evaluated_archs: 所有已评估的架构
            train_ratio: 训练集比例
            min_samples: 起始训练样本数
            step: 训练样本数增量步长

        Returns:
            包含以下键的字典:
            - train_sizes: 训练样本数列表
            - r2_scores: 每个训练样本数对应的R² [N, 3]
        """
        if len(evaluated_archs) < min_samples + 5:
            return {'train_sizes': [], 'r2_scores': np.array([])}

        train_archs, val_archs = self.split_train_val_by_time(evaluated_archs, train_ratio)

        if len(val_archs) == 0 or len(train_archs) < min_samples:
            return {'train_sizes': [], 'r2_scores': np.array([])}

        X_val = self._encode_architectures(val_archs)
        y_val = self._get_targets(val_archs)

        max_train = len(train_archs)
        train_sizes = list(range(min_samples, max_train + 1, step))
        if train_sizes[-1] != max_train:
            train_sizes.append(max_train)

        r2_scores = []

        for size in train_sizes:
            subset_train = train_archs[:size]
            if len(subset_train) < self.min_train_samples:
                r2_scores.append(np.zeros(3))
                continue

            X_train = self._encode_architectures(subset_train)
            y_train = self._get_targets(subset_train)

            try:
                model, scaler_X, scaler_y = self._train_single_model(X_train, y_train)
                val_pred = self._predict_with_model(model, scaler_X, scaler_y, X_val)

                r2 = np.zeros(3)
                for dim in range(3):
                    try:
                        r2[dim] = r2_score(y_val[:, dim], val_pred[:, dim])
                    except:
                        r2[dim] = 0.0
                r2_scores.append(r2)
            except:
                r2_scores.append(np.zeros(3))

        return {
            'train_sizes': train_sizes,
            'r2_scores': np.array(r2_scores)
        }
