import numpy as np
from scipy.linalg import cholesky, cho_solve


class GP:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, noise_var: float = 1., lscale: float = 1.,
                 k_var: float = 1., prior_mean: float = 0, standardize=True):
        self.__lscale = lscale
        self.__k_var = k_var
        self.__noise_var = noise_var
        self.__x_train = x_train
        self.__y_train = y_train
        self.__n_train = y_train.shape[0]
        if standardize:
            self.__prior_mean = np.mean(y_train)
        else:
            self.__prior_mean = prior_mean
        self.__standardize = standardize

        self.__K_inv = None
        self.__set_k_inv()

    @property
    def n_train(self):
        return self.__n_train

    @property
    def lscale(self):
        return self.__lscale

    @property
    def k_var(self):
        return self.__k_var

    @property
    def noise_var(self):
        return self.__noise_var

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def K_inv(self):
        return self.__K_inv

    def gauss_kernel(self, x1, x2):
        assert x1.ndim == 2
        assert x2.ndim == 2
        r = np.linalg.norm(x1[:, None] - x2, axis=2)
        return self.k_var * np.exp(-0.5 * np.square(r) / np.square(self.lscale))

    def __set_k_inv(self):
        K = self.gauss_kernel(self.x_train, self.x_train)
        K += self.noise_var * np.eye(self.n_train)
        self.__K_inv = cho_solve((cholesky(K, True), True), np.eye(self.n_train))

    def predict(self, x, fullcov=False):
        assert x.ndim == 2
        kx = self.gauss_kernel(self.x_train, x)  # (n,m)
        kK = kx.T @ self.__K_inv
        mean = kK @ (self.y_train - self.__prior_mean)
        var = self.gauss_kernel(x, x) + self.noise_var * np.eye(x.shape[0]) - kK @ kx
        if not fullcov:
            var = np.diag(var)
        return mean.flatten() + self.__prior_mean, var

    def add_observation(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.__x_train = np.vstack([self.__x_train, x])
        self.__y_train = np.vstack([self.__y_train, y])
        self.__n_train = self.__y_train.shape[0]
        self.__set_k_inv()
        if self.__standardize:
            self.__prior_mean = np.mean(self.y_train)
        return
