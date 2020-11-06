import numpy as np


class RBFFourierBasis:
    def __init__(self, n_features: int, n_dim: int, rbf_ls: float = 1.,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        self.__n_features = n_features
        self.__n_dim = n_dim
        self.__rbf_ls = rbf_ls
        self.__rng = rng
        self.__weight = rng.normal(size=(n_dim, n_features)) / rbf_ls
        self.__offset = rng.uniform(low=0, high=2 * np.pi, size=n_features)
        return

    @property
    def n_features(self):
        return self.__n_features

    @property
    def n_dims(self):
        return self.__n_dim

    @property
    def rbf_ls(self):
        return self.__rbf_ls

    @property
    def weight(self):
        return self.__weight

    @property
    def offset(self):
        return self.__offset

    def transform(self, x):
        assert x.ndim == 2 and x.shape[1] == self.n_dims, 'x should be 2 dim'
        rff = np.sqrt(2 / self.n_features) * np.cos(x @ self.weight + self.offset)
        return rff
