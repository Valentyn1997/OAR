import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from numpy.polynomial.polynomial import polyval


# class SyntheticData:
#
#     mode = None
#     instrument = None
#
#     def __init__(self, n_samples_train, n_samples_test, **kwargs):
#         self.n_samples_train = n_samples_train
#         self.n_samples_test = n_samples_test
#
#     def get_data(self):
#         data_dicts = []
#         for n_samples in [self.n_samples_train, self.n_samples_test]:
#             if self.mode == 'normal_uniform':
#                 u = np.random.normal(0.0, 1.0, n_samples)
#                 x = np.random.uniform(-2.0, 2.0, n_samples)
#             elif self.mode == 'moons':
#                 ux, _ = make_moons(n_samples=n_samples, noise=0.125)  # noise=0.125
#                 u, x = ux[:, 0] - 0.6, 1.5 * ux[:, 1] - 0.5
#             else:
#                 raise NotImplementedError()
#             t = np.random.binomial(1, 1.0 / (1.0 + np.exp(- (0.75 * x - u + 0.5))), n_samples)
#
#             y_pot0 = self.get_mu(0, x, u) + np.random.normal(0.0, 1.0, n_samples)
#             y_pot1 = self.get_mu(1, x, u) + np.random.normal(0.0, 1.0, n_samples)
#             y = y_pot0 * (1 - t) + y_pot1 * t
#
#             data_dicts.append({
#                 'cov_f': np.stack([x, u], -1),
#                 'treat_f': t,
#                 'out_f': y,
#                 'out_pot0': y_pot0,
#                 'out_pot1': y_pot1,
#                 'mu0': self.get_mu(0, x, u),
#                 'mu1': self.get_mu(1, x, u),
#             })
#         return data_dicts
#
#     def get_mu(self, treat, x, u):
#         if treat == 0:
#             return - 1 * x - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)
#         else:
#             return 1 * x + 1 - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)
#
#     def get_gt_cate(self, x, u):
#         return self.get_mu(1, x, u) - self.get_mu(0, x, u)
#
#
# class SyntheticNormalUniformData(SyntheticData):
#     mode = 'normal_uniform'


class Synthetic:

    def __init__(self, n_samples_train, n_samples_test, cov_shift=1., C0=(0.75, 1.), C1=(1.0, -0.5, 0.5), **kwargs):
        self.cov_shift = cov_shift


        self.C0 = np.array(C0)
        self.C1 = np.array(C1)

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

    def get_data(self) -> dict:
        data_dicts = []

        for n_samples in [self.n_samples_train, self.n_samples_test]:
            n_samples_per_treat = n_samples // 2
            X0 = self.cov_shift + np.random.randn(n_samples_per_treat, 1)
            X1 = np.random.randn(n_samples_per_treat, 1)

            # noise_0 = np.random.randn(len(X0), 1)
            # noise_1 = np.random.randn(len(X1), 1)

            Y0 = self.get_mu(X0, 0) + np.random.randn(len(X0), 1)
            Y1 = self.get_mu(X1, 1) + np.random.randn(len(X1), 1)

            mu0 = np.concatenate([self.get_mu(X0, 0), self.get_mu(X1, 0)])
            mu1 = np.concatenate([self.get_mu(X0, 1), self.get_mu(X1, 1)])
            Y0_pot = np.concatenate([Y0, self.get_mu(X1, 0) + np.random.randn(len(X1), 1)])
            Y1_pot = np.concatenate([self.get_mu(X0, 1) + np.random.randn(len(X0), 1), Y1])

            out_f = np.concatenate([Y0, Y1], axis=0)

            data_dicts.append({
                'cov_f': np.concatenate([X0, X1], axis=0),
                'treat_f': np.concatenate([np.zeros((n_samples_per_treat,)), np.ones((n_samples_per_treat,))]),
                'out_f': out_f,
                'out_pot0': Y0_pot,
                'out_pot1': Y1_pot,
                'mu0': mu0,
                'mu1': mu1,
            })

        return data_dicts

    def get_mu(self, x, treat):
        if treat == 0:
            mu = polyval(x, self.C0)
        else:
            mu = polyval(x, self.C1)
        return 3 * np.cos(mu) - 2.5 * np.sin(mu)

