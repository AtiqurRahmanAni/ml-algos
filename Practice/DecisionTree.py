import numpy as np
from collections import Counter


class Node:
    def __init__(self, left=None, right=None, threshold=None, feature_idx=None, label=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.label = label
        self.feature_idx = feature_idx

    def is_leaf_node(self):
        return self.label is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, depth=10):
        self.max_depth = depth
        self.min_sample_split = min_sample_split
        self.n_features = None
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[-1] if not self.n_features else min(
            X.shape[-1], self.n_features)
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0):
        n_sample, n_features = X.shape
        unique_labels = np.unique(y)
        # terminating condition
        if depth >= self.max_depth or len(unique_labels) == 1 or n_sample < self.min_sample_split:
            return Node(label=self._most_common_label(y))

        # grow tree now, need best feature and threshold
        # feature_idx = np.random.choice(
        #     n_features, self.n_features, replace=False)
        best_threshold, best_feature = self._best_split(
            X, y, np.arange(n_features))

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        X_left, X_right = X[left_idxs, :], X[right_idxs, :]
        y_left, y_right = y[left_idxs], y[right_idxs]

        left_node = self._grow_tree(X_left, y_left, depth + 1)
        right_node = self._grow_tree(X_right, y_right, depth + 1)

        return Node(left=left_node, right=right_node, threshold=best_threshold, feature_idx=best_feature)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_idx: np.ndarray):

        best_gain = -1
        best_threshold = None
        best_feature = None

        for feature in feature_idx:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature

        return best_threshold, best_feature

    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, threshold: float):
        parent_entopy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0

        y_left, y_right = y[left_idxs], y[right_idxs]
        e_l, e_r = self._entropy(y_left), self._entropy(y_right)

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entopy - child_entropy
        return ig

    def _entropy(self, y: np.ndarray):
        # [2, 1, 3, 5]
        hist = np.array(list(Counter(y).values()))
        probabilities = hist / len(y)
        return -np.sum(probabilities * np.log(probabilities))

    def _split(self, X_column: np.ndarray, threshold: float):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def _most_common_label(self, y: np.ndarray):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X: np.ndarray):
        predictions = []

        for x in X:
            prediction = self._tranvarse_tree(self.root, x)
            predictions.append(prediction)

        return np.array(predictions)

    def _tranvarse_tree(self, root: Node, x: np.ndarray):
        if root.is_leaf_node():
            return root.label

        if x[root.feature_idx] <= root.threshold:
            return self._tranvarse_tree(root.left, x)

        return self._tranvarse_tree(root.right, x)
