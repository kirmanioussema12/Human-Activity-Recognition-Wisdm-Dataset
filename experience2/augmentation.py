import numpy as np

def add_jitter(X, sigma=0.01):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def augment_specific_classes(X, y, class_sample_map, augment_func):
    X_augmented, y_augmented = [], []

    for cls, n_samples in class_sample_map.items():
        X_cls = X[y == cls]
        sampled_indices = np.random.choice(len(X_cls), size=n_samples, replace=True)
        X_sampled = X_cls[sampled_indices]
        X_aug = augment_func(X_sampled)
        X_augmented.append(X_aug)
        y_augmented.append(np.full(n_samples, cls))

    return np.concatenate([X] + X_augmented, axis=0), np.concatenate([y] + y_augmented, axis=0)
