import numpy as np
import scipy.linalg

def numpy_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def numpy_f1_score(y_true, y_pred, average='macro'):
    class_labels = np.unique(y_true)
    f1_sum = 0.0

    for cls in class_labels:
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_sum += f1

    return f1_sum / len(class_labels)


class NumPyScaler:
    def __init__(self): 
        self.mean_, self.std_ = None, None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-9  
        return self
    
    def transform(self, X):
        if self.mean_ is None: 
            raise RuntimeError("Must fit scaler before transforming.")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class NumPyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_
        cov_matrix = np.cov(centered, rowvar=False)

        eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
        order = np.argsort(eigen_vals)[::-1]

        eigen_vecs = eigen_vecs[:, order]
        self.components_ = eigen_vecs[:, :self.n_components]
        return self
    
    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("PCA not fitted.")
        return (X - self.mean_) @ self.components_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class NumPyLDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.classes_ = None
    
    def _as_numpy(self, X):
        return X.values if hasattr(X, 'values') else np.asarray(X)
    
    def fit(self, X, y):
        X = self._as_numpy(X)
        y = self._as_numpy(y)
        self.classes_ = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        if self.n_components is None or self.n_components >= n_classes:
            self.n_components = n_classes - 1
        
        global_mean = np.mean(X, axis=0)
        scatter_within = np.zeros((n_features, n_features))
        scatter_between = np.zeros((n_features, n_features))

        for cls in self.classes_:
            X_c = X[y == cls]
            count_c = len(X_c)
            if count_c == 0:
                continue

            class_mean = np.mean(X_c, axis=0)
            scatter_within += (X_c - class_mean).T @ (X_c - class_mean)

            mean_diff = (class_mean - global_mean).reshape(n_features, 1)
            scatter_between += count_c * (mean_diff @ mean_diff.T)
        
        if np.linalg.det(scatter_within) == 0:
            scatter_within += 1e-6 * np.eye(n_features)

        try:
            eigen_vals, eigen_vecs = scipy.linalg.eig(scatter_between, scatter_within)
        except np.linalg.LinAlgError:
            eigen_vals, eigen_vecs = scipy.linalg.eig(scatter_between, scatter_within + 1e-6 * np.eye(n_features))
        
        eigen_vals = eigen_vals.real
        eigen_vecs = eigen_vecs.real

        order = np.argsort(eigen_vals)[::-1]
        self.scalings_ = eigen_vecs[:, order][:, :self.n_components]

        return self
    
    def transform(self, X):
        X = self._as_numpy(X)
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class SoftmaxClassifier:
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=64,
                 alpha=0.01, random_state=42):
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.random_state = random_state

        self.W = None
        self.b = None
        self.classes_ = None
        self.n_classes_ = -1
        self.n_features_ = -1
        self.rng = np.random.RandomState(random_state)
    
    def _one_hot_encode(self, y):
        matrix = np.zeros((len(y), self.n_classes_))
        matrix[np.arange(len(y)), y.astype(int)] = 1
        return matrix
    
    def _softmax(self, logits):
        stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(stable)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def fit(self, X, y):
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y

        n_samples, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        y_onehot = self._one_hot_encode(y)

        init_bound = np.sqrt(6 / (self.n_features_ + self.n_classes_))
        self.W = self.rng.uniform(-init_bound, init_bound, (self.n_features_, self.n_classes_))
        self.b = np.zeros((1, self.n_classes_))

        for epoch in range(self.n_epochs):
            idx = np.arange(n_samples)
            self.rng.shuffle(idx)

            X_shuffled = X[idx]
            y_shuffled = y_onehot[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if len(X_batch) == 0:
                    continue

                logits = X_batch @ self.W + self.b
                probs = self._softmax(logits)
                error = probs - y_batch

                grad_W = (X_batch.T @ error) / len(X_batch) + self.alpha * self.W
                grad_b = np.sum(error, axis=0, keepdims=True) / len(X_batch)

                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b

    def predict_proba(self, X):
        X = X.values if hasattr(X, 'values') else X
        logits = X @ self.W + self.b
        return self._softmax(logits)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class NumPyKNNClassifier:
    def __init__(self, k=5, weights='uniform'):
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_classes_ = -1
        self.class_to_index = None
    
    def fit(self, X, y):
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_index = {c: i for i, c in enumerate(self.classes_)}
        return self
    
    def predict_proba(self, X):
        X = X.values if hasattr(X, 'values') else X

        n_test = X.shape[0]

        sq_test = np.sum(X**2, axis=1, keepdims=True)
        sq_train = np.sum(self.X_train**2, axis=1, keepdims=True).T
        dot_prod = X @ self.X_train.T

        dist_sq = np.maximum(0, sq_test - 2 * dot_prod + sq_train)
        distances = np.sqrt(dist_sq)

        knn_indices = np.argsort(distances, axis=1)[:, :self.k]
        knn_distances = np.take_along_axis(distances, knn_indices, axis=1)
        knn_labels = self.y_train[knn_indices]

        probs = np.zeros((n_test, self.n_classes_))

        for i in range(n_test):
            labels_row = knn_labels[i]
            dist_row = knn_distances[i]

            for label, dist in zip(labels_row, dist_row):
                idx = self.class_to_index[label]
                if self.weights == 'uniform':
                    probs[i, idx] += 1
                else:
                    probs[i, idx] += 1 / (dist + 1e-6)

        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class KFoldStackingEnsemble:
    def __init__(self, base_models, meta_model, k_folds=3, random_state=42):
        self.base_models_template = base_models
        self.meta_model = meta_model
        self.k_folds = k_folds
        self.random_state = random_state
        self.classes_ = None
        self.base_models_fitted = []
        
    def _clone_model(self, model):
        if isinstance(model, SoftmaxClassifier):
            params = dict(
                learning_rate=model.learning_rate,
                n_epochs=model.n_epochs,
                batch_size=model.batch_size,
                alpha=model.alpha,
                random_state=model.random_state
            )
        elif isinstance(model, NumPyKNNClassifier):
            params = dict(k=model.k, weights=model.weights)
        else:
            raise TypeError(f"Cannot clone model type {model.__class__}")

        return model.__class__(**params)

    def fit(self, X_dict, y):
        y = y.values if hasattr(y, 'values') else y
        self.classes_ = np.unique(y)

        n_samples = len(y)
        np.random.seed(self.random_state)

        idx = np.arange(n_samples)
        np.random.shuffle(idx)

        folds = np.array_split(idx, self.k_folds)

        n_meta_features = len(self.base_models_template) * len(self.classes_)
        meta_train = np.zeros((n_samples, n_meta_features))
        meta_labels = np.zeros(n_samples)

        for fold_idx in range(self.k_folds):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(self.k_folds) if i != fold_idx])

            feature_col_start = 0

            for name, model_template, feature_key in self.base_models_template:
                X_source = X_dict[feature_key]
                X_full = X_source.values if hasattr(X_source, 'values') else np.asarray(X_source)

                X_train = X_full[train_idx]
                y_train = y[train_idx]

                X_valid = X_full[val_idx]
                y_valid = y[val_idx]

                model = self._clone_model(model_template)
                model.fit(X_train, y_train)

                probas = model.predict_proba(X_valid)
                n_cols = probas.shape[1]

                meta_train[val_idx, feature_col_start:feature_col_start+n_cols] = probas
                feature_col_start += n_cols

            meta_labels[val_idx] = y[val_idx]

        self.meta_model.fit(meta_train, meta_labels)

        self.base_models_fitted = []
        for name, model_template, feature_key in self.base_models_template:
            X_source = X_dict[feature_key]
            X_full = X_source.values if hasattr(X_source, 'values') else np.asarray(X_source)

            model = self._clone_model(model_template)
            model.fit(X_full, y)
            self.base_models_fitted.append((model, feature_key))

        return self

    def predict(self, X_dict_val):
        meta_features = []

        for model, feature_key in self.base_models_fitted:
            X_val = X_dict_val[feature_key]
            probas = model.predict_proba(X_val)
            meta_features.append(probas)

        meta_test = np.hstack(meta_features)
        return self.meta_model.predict(meta_test)
