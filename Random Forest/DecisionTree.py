import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(
            X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_sample, n_features = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criterion
        if depth >= self.max_depth or n_labels == 1 or n_sample < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features, self.n_features, replace=False)

        '''
        find the best split
        split idx is the column number of the dataset
        split_threshold is the best value of the column that gives maximum information gain
        '''
        split_idx, split_threshold = self._best_split(X, y, feat_idx)

        # create child nodes
        left_idx, right_idx = self._split(X[:, split_idx], split_threshold)

        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(left=left, right=right, threshold=split_threshold, feature=split_idx)

    def _best_split(self, X, y, feat_idx):
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in feat_idx:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:  # here threshold is the row value of the current column
                gain = self._informative_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = threshold
                    split_idx = idx

        return split_idx, split_threshold

    def _informative_gain(self, X_column, y, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()
        return left_idx, right_idx

    def _entropy(self, y):
        '''
        entropy = -sum(p(x) * log(p(x)))
        '''
        hist = np.array(list(Counter(y).values()))
        probability = hist / len(y)
        entropy = -np.sum(probability * np.log(probability))
        return entropy

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
