import numpy as np
from scipy.linalg import cholesky, cho_solve


class BLR:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, noise_var: float = 1.,
                 w_mean: float = 0, w_var: float = 1.):
        self.__noise_var = noise_var
        self.__x_train = x_train
        self.__y_train = y_train
        self.__n_train = y_train.shape[0]
        self.__input_dim = x_train.shape[1]
        self.__w_mean = np.full(self.input_dim, w_mean)
        self.__w_var = np.eye(self.input_dim) * w_var
        self.__w_var_inv = np.eye(self.input_dim) / w_var
        self.__set_posterior()
        return

    @property
    def n_train(self):
        return self.__n_train

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
    def input_dim(self):
        return self.__input_dim

    @property
    def w_mean(self):
        return self.__w_mean

    @property
    def w_var(self):
        return self.__w_var

    @property
    def w_var_inv(self):
        return self.__w_var_inv

    def __set_posterior(self):
        pos_var_inv = self.w_var_inv + self.x_train.T @ self.x_train / self.noise_var
        pos_var = cho_solve((cholesky(pos_var_inv, True), True), np.eye(self.input_dim))
        pos_mean = pos_var @ (self.w_var_inv @ self.w_mean + self.x_train.T @ self.y_train / self.noise_var)
        self.__w_mean = pos_mean
        self.__w_var = pos_var
        self.__w_var_inv = pos_var_inv
        return

    def predict(self, x, fullcov=False):
        assert x.ndim == 2
        mean = x @ self.w_mean
        var = self.noise_var + x @ self.w_var @ x.T
        if not fullcov:
            var = np.diag(var)
        return mean.flatten(), var

    def add_observation(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.__x_train = np.vstack([self.__x_train, x])
        self.__y_train = np.vstack([self.__y_train, y])
        self.__n_train = self.__y_train.shape[0]
        self.__set_posterior()
        return
