import functools as ft
import itertools as it
import pathlib

import numpy as np
from joblib import delayed, Parallel
from scipy.stats import norm
from sklearn.utils.extmath import cartesian

from src.models import RBFFourierBasis, GP


class PoolBO:
    def __init__(self, af: str, max_iteration: int, n_init: int = 5, seed: int = 0, noise_var: float = 1e-4,
                 save_inference_r: bool = False):
        self.__known_af = {'random': self.random,
                           'pi': self.pi,
                           'ei': self.ei,
                           'ucb': ft.partial(self.ucb, beta_rt=3),
                           'gp-ucb': ft.partial(self.ucb, beta_rt=3),
                           'mes': ft.partial(self.mes, n_sample=100)}
        # known_af = {'pi', 'ei', 'ucb', 'gp-ucb', 'random', 'gp-ucb', 'mes'}
        assert af in self.__known_af.keys(), f'Unknown acquisition function: {af}'

        self.__save_inference_r = save_inference_r
        self.__n_init = n_init
        self.__seed = seed
        self.__rng = np.random.default_rng(seed)
        self.__af = af
        self.__iteration = 0
        self.__input_dim = 2
        self.__max_iteration = max_iteration

        self.__noise_var = noise_var
        self.__noise_std = np.sqrt(noise_var)

        n_features = 2000
        self.__rff = RBFFourierBasis(n_features, self.input_dim, rng=np.random.default_rng(seed))
        self.__w = self.rng.normal(size=n_features)
        grid_1d = np.linspace(-5, 5, 70)
        self.__x_pool = cartesian([grid_1d for _ in range(self.input_dim)])
        self.__observed_idx = self.rng.choice(np.arange(self.x_pool.shape[0]), size=n_init, replace=True).tolist()
        x_train = self.__x_pool[self.observed_idx].reshape(n_init, self.input_dim)
        y_train = self.obj_f(x_train).reshape(n_init, 1)
        self.__f_best = np.max(y_train)
        y_train += self.rng.normal(scale=self.noise_std, size=(n_init, 1))
        self.__y_best = np.max(y_train)
        self.__gp = GP(x_train=x_train, y_train=y_train, noise_var=self.noise_var, standardize=False)

        self.__f_opt = np.max(self.obj_f(self.x_pool))
        self.__simple_r = [float(self.f_opt - self.f_best)]
        self.__inference_r = []
        if self.save_inference_r:
            m, _ = self.gp.predict(self.x_pool)
            max_i = np.argmax(m)
            self.__inference_r.append(float(self.f_opt - self.obj_f(self.x_pool[max_i])))
        self.__next_x = None
        self.__next_i = None
        self.__next_x_af = None
        return

    @property
    def observed_idx(self):
        return self.__observed_idx

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def x_pool(self):
        return self.__x_pool

    @property
    def af(self):
        return self.__af

    @property
    def x_train(self):
        return self.__gp.x_train

    @property
    def y_train(self):
        return self.__gp.y_train

    @property
    def y_best(self):
        return self.__y_best

    @property
    def f_best(self):
        return self.__f_best

    @property
    def f_opt(self):
        return self.__f_opt

    @property
    def gp(self):
        return self.__gp

    @property
    def rng(self):
        return self.__rng

    @property
    def noise_var(self):
        return self.__noise_var

    @property
    def noise_std(self):
        return self.__noise_std

    @property
    def next_x(self):
        return self.__next_x

    @property
    def next_i(self):
        return self.__next_i

    @property
    def next_x_af(self):
        return self.__next_x_af

    @property
    def iteration(self):
        return self.__iteration

    @property
    def max_iteration(self):
        return self.__max_iteration

    @property
    def simple_r(self):
        return self.__simple_r

    @property
    def inference_r(self):
        return self.__inference_r

    @property
    def save_inference_r(self):
        return self.__save_inference_r

    @property
    def known_af_name(self):
        return self.__known_af.keys()

    def next_observation(self):
        self.__known_af[self.af]()
        next_f = self.obj_f(self.__next_x)
        self.__f_best = float(max(self.__f_best, next_f))
        next_f += self.rng.normal(scale=self.noise_std)
        self.__y_best = float(max(self.__y_best, next_f))
        self.gp.add_observation(self.__next_x, next_f)
        self.__simple_r.append(float(self.f_opt - self.f_best))

        if self.save_inference_r:
            m, _ = self.gp.predict(self.x_pool)
            max_i = np.argmax(m)
            self.__inference_r.append(float(self.f_opt - self.obj_f(self.x_pool[max_i])))
        # self.__x_pool = self.rng.permutation(self.__x_pool)
        self.observed_idx.append(int(self.next_i))
        self.__iteration += 1
        return

    def obj_f(self, x):
        return self.__rff.transform(x.reshape(-1, self.input_dim)) @ self.__w

    def random(self):
        self.__next_i = self.rng.choice(np.setdiff1d(np.arange(self.x_pool.shape[0]), self.observed_idx))
        self.__next_x = self.x_pool[self.next_i]
        return

    def pi(self, maximize=True):
        mean, var = self.gp.predict(self.x_pool)
        std = np.sqrt(var)
        if maximize:
            z = (mean - self.y_best) / std
        else:
            z = (self.y_best - mean) / std
        af = norm.cdf(z)
        af[self.observed_idx] = -1

        max_i = np.argmax(af)
        self.__next_i = max_i
        self.__next_x = self.x_pool[max_i]
        self.__next_x_af = af[max_i]
        return

    def ei(self, maximize=True):
        mean, var = self.gp.predict(self.x_pool)
        std = np.sqrt(var)
        if maximize:
            z = (mean - self.y_best) / std
        else:
            z = (self.y_best - mean) / std
        af = std * (z * norm.cdf(z) + norm.pdf(z))
        af[self.observed_idx] = -1

        max_i = np.argmax(af)

        self.__next_i = max_i
        self.__next_x = self.x_pool[max_i]
        self.__next_x_af = af[max_i]
        return

    def ucb(self, beta_rt=3):
        mean, var = self.gp.predict(self.x_pool)
        std = np.sqrt(var)
        af = mean + beta_rt * std
        af[self.observed_idx] = np.min(af) - 1

        max_i = np.argmax(af)

        self.__next_i = max_i
        self.__next_x = self.x_pool[max_i]
        self.__next_x_af = af[max_i]
        return

    def mes(self, n_sample: int = 10, maximize=True):
        mean, cov = self.gp.predict(self.x_pool, fullcov=True)
        std = np.sqrt(np.diag(cov))

        if not maximize:
            mean = -mean

        sample_path = self.rng.multivariate_normal(mean, cov, method='cholesky', size=n_sample)
        y_max = np.max(sample_path, axis=1).flatten()
        gamma = (y_max[:, None] - mean) / std
        gamma_cdf = norm.cdf(gamma)

        af = np.mean(gamma * norm.pdf(gamma) / (2 * gamma_cdf) - np.log(gamma_cdf), axis=0)
        af[self.observed_idx] = np.min(af) - 1

        max_i = np.argmax(af)

        self.__next_i = max_i
        self.__next_x = self.x_pool[max_i]
        self.__next_x_af = af[max_i]
        return


def main():
    max_itr = 100

    af_list = ['mes-s', 'mes-g', 'pi', 'random', 'ei']
    # af_list=['ucb', 'gp-ucb']
    # af_list = ['random']
    seeds = np.arange(20)

    def one_exp(af, seed):
        bo = PoolBO(af, seed=seed, max_iteration=max_itr, n_init=10, save_inference_r=True)
        while bo.iteration < max_itr:
            # print(f'{bo.simple_r[-1]},{bo.inference_r[-1]}, x:{bo.next_x}, af:{bo.next_x_af}')
            bo.next_observation()
        np.savez(result_dir / f'{af}_seed{seed}', sr=bo.simple_r, ir=bo.inference_r)

    Parallel(n_jobs=40, verbose=10)([delayed(one_exp)(AF, SEED) for AF, SEED in it.product(af_list, seeds)])
    return


if __name__ == '__main__':
    result_dir = pathlib.Path(__file__).parent.parent / 'image/bo_result'

    main()
