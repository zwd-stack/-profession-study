#models.py
import math
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """模型基类"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y):
        """评估模型性能"""
        predictions = self.predict(X)
        metrics = {
            'MSE': self._calculate_mse(y, predictions),
            'RMSE': self._calculate_rmse(y, predictions),
            'MAE': self._calculate_mae(y, predictions),
            'R2': self._calculate_r2(y, predictions)
        }
        return metrics, predictions

    def _calculate_mse(self, y_true, y_pred):
        """计算均方误差"""
        return sum((pred - true) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

    def _calculate_rmse(self, y_true, y_pred):
        """计算均方根误差"""
        return math.sqrt(self._calculate_mse(y_true, y_pred))

    def _calculate_mae(self, y_true, y_pred):
        """计算平均绝对误差"""
        return sum(abs(pred - true) for true, pred in zip(y_true, y_pred)) / len(y_true)

    def _calculate_r2(self, y_true, y_pred):
        """计算R²值"""
        mean_y = sum(y_true) / len(y_true)
        ss_tot = sum((y - mean_y) ** 2 for y in y_true)
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class LinearRegression(BaseModel):
    """线性回归模型"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        # 初始化权重和偏置
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # 梯度下降
        for _ in range(self.n_iterations):
            y_pred = [self._predict_single(x) for x in X]

            # 计算梯度
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = y_pred[i] - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]

                db += error

            # 更新参数
            for j in range(n_features):
                self.weights[j] -= (self.learning_rate * dw[j]) / n_samples
            self.bias -= (self.learning_rate * db) / n_samples

        # 计算特征重要性
        self.feature_importances_ = self._calculate_feature_importance()
        return self

    def _calculate_feature_importance(self):
        """计算特征重要性"""
        if self.weights is None:
            return None
        importances = [abs(w) for w in self.weights]
        total = sum(importances)
        return [imp / total for imp in importances] if total > 0 else [1.0 / len(importances)] * len(importances)

    def predict(self, X):
        """预测新数据"""
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        """单样本预测"""
        return sum(w * val for w, val in zip(self.weights, x)) + self.bias


class DecisionTreeRegressor(BaseModel):
    """决策树回归模型"""

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_importances_ = None
        self.n_features = None

    def fit(self, X, y):
        """训练决策树模型"""
        self.n_features = len(X[0])
        self.tree = self._grow_tree(X, y)
        self.feature_importances_ = self._calculate_feature_importance()
        return self

    def _calculate_feature_importance(self):
        """计算特征重要性"""
        importances = [0.0] * self.n_features

        def _update_importance(node, depth=0):
            if node.value is not None:  # 叶节点
                return

            current_importance = 1.0 / (depth + 1)  # 重要性随深度递减
            importances[node.feature] += current_importance

            if node.left:
                _update_importance(node.left, depth + 1)
            if node.right:
                _update_importance(node.right, depth + 1)

        _update_importance(self.tree)

        # 归一化
        total_importance = sum(importances)
        if total_importance > 0:
            importances = [imp / total_importance for imp in importances]
        else:
            importances = [1.0 / self.n_features] * self.n_features

        return importances

    def _grow_tree(self, X, y, depth=0):
        """递归生长决策树"""
        n_samples = len(y)

        # 检查停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split:
            return self.Node(value=sum(y) / len(y))

        best_gain = float('-inf')
        best_split = None


        for feature_idx in range(self.n_features):
            values = sorted(set(x[feature_idx] for x in X))

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2


                left_mask = [x[feature_idx] <= threshold for x in X]
                right_mask = [not m for m in left_mask]

                if not any(left_mask) or not any(right_mask):
                    continue

                left_y = [y[i] for i in range(len(y)) if left_mask[i]]
                right_y = [y[i] for i in range(len(y)) if right_mask[i]]


                gain = self._calculate_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)

        if best_split is None:
            return self.Node(value=sum(y) / len(y))

        feature_idx, threshold = best_split


        left_mask = [x[feature_idx] <= threshold for x in X]
        right_mask = [not m for m in left_mask]

        left_X = [X[i] for i in range(len(X)) if left_mask[i]]
        left_y = [y[i] for i in range(len(y)) if left_mask[i]]
        right_X = [X[i] for i in range(len(X)) if right_mask[i]]
        right_y = [y[i] for i in range(len(y)) if right_mask[i]]


        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)

        return self.Node(feature_idx, threshold, left, right)

    def _calculate_gain(self, parent, left, right):
        """计算分裂增益"""

        def calculate_variance(y):
            if not y:
                return 0
            mean = sum(y) / len(y)
            return sum((val - mean) ** 2 for val in y) / len(y)

        n = len(parent)
        n_l = len(left)
        n_r = len(right)

        parent_variance = calculate_variance(parent)
        left_variance = calculate_variance(left)
        right_variance = calculate_variance(right)


        gain = parent_variance - (n_l / n * left_variance + n_r / n * right_variance)
        return gain

    def predict(self, X):
        """预测新数据"""
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        """单样本预测"""
        node = self.tree
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class KNNRegressor(BaseModel):
    """K近邻回归模型"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
        self.feature_importances_ = self._calculate_feature_importance()
        return self

    def _calculate_feature_importance(self):
        """计算特征重要性"""
        if self.X_train is None:
            return None

        n_features = len(self.X_train[0])
        importances = [0.0] * n_features

        # 基于特征的方差计算重要性
        for i in range(n_features):
            values = [x[i] for x in self.X_train]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            importances[i] = variance

        # 归一化
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]
        else:
            importances = [1.0 / n_features] * n_features

        return importances

    def predict(self, X):
        """预测新数据"""
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        """单样本预测"""
        # 计算到所有训练样本的距离
        distances = [(self._euclidean_distance(x, x_train), i)
                     for i, x_train in enumerate(self.X_train)]

        # 获取k个最近邻
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]

        # 计算k个最近邻的平均值作为预测值
        k_nearest_values = [self.y_train[idx] for _, idx in k_nearest]
        return sum(k_nearest_values) / self.k

    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    def feature_importance(self, X, y):
        if self.feature_importances_ is None:
            return [1.0 / len(X[0])] * len(X[0])
        return self.feature_importances_