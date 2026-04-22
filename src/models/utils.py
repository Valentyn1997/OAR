import numpy as np
from torch import nn
import torch.nn.functional as F
import ot
import logging

import torch
from copy import deepcopy
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune

logger = logging.getLogger(__name__)


def fit_eval_kfold(args: dict, orig_hparams: DictConfig, model_cls, train_data_dict: dict, val_data_dict: dict, name: str,
                   kind: str = None, subnet_name: str = None, **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param orig_hparams: DictConfig of original hyperparameters
    :param model_cls: class of model
    :param kwargs: Other args
    """
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params[name], args)
    model_cls.set_subnet_hparams(new_params[subnet_name], args) if subnet_name is not None else None
    # new_params.exp.device = 'cuda'

    torch.set_default_device('cuda')

    if val_data_dict is None:  # KFold hparam tuning
        kf = KFold(n_splits=5, random_state=orig_hparams.exp.seed, shuffle=True)
        val_metrics = []
        for train_index, val_index in kf.split(train_data_dict['cov_f']):
            ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, train_index), \
                subset_by_indices(train_data_dict, val_index)

            model = model_cls(new_params, kind=kind, **kwargs)
            model.fit(train_data_dict=ttrain_data_dict, log=False)
            log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
            val_metrics.append(log_dict[model.val_metric])
        tune.report(val_metric=np.mean(val_metrics))

    else:  # predefined hold-out hparam tuning
        model = model_cls(new_params, kind=kind, **kwargs)
        model.fit(train_data_dict=train_data_dict, log=False)
        log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
        tune.report(val_metric=log_dict[model.val_metric])


def subset_by_indices(data_dict: dict, indices: list):
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        subset_data_dict[k] = np.copy(data_dict[k][indices])
    return subset_data_dict


class AdaptiveDropout(nn.Module):
    """
    Custom Dropout that allows a different dropout probability for each instance in the batch.
    """
    def __init__(self, inplace: bool = False):
        """
        Args:
            inplace (bool): If True, performs the operation in-place.
        """
        super(AdaptiveDropout, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the AdaptiveDropout.

        Args:
            x (torch.Tensor): Input tensor of shape (N, *).
            p (torch.Tensor): Dropout probabilities of shape (N,) or broadcastable to x.

        Returns:
            torch.Tensor: Output after applying dropout with per-instance probability.
        """
        # If we are not in training mode or p is None, just return x
        if (not self.training) or p is None:
            return x

        # assert (p <= 1.0).all()

        # if not (p >= 0.0).all():
        #     logger.warning('Negative dropout probability. Will be zeroed.')

        # Ensure p is a 1D tensor of shape (N,) or broadcastable to the first dimension of x
        # We reshape p so it can be broadcast across the remaining dimensions of x
        # e.g. if x has shape [N, C, H, W], p should become [N, 1, 1, 1].
        while p.dim() < x.dim():
            p = p.unsqueeze(-1)
        p = p.view([x.size(0)] + [1]*(x.dim() - 1))  # shape: [N, 1, 1, ...] matching x.dim()

        p[p >= 1.0] = 0.0  # Omitting division by zero
        p[p < 0.0] = 0.0

        # Generate dropout mask: 1 where we keep the neuron, 0 where we drop
        # torch.rand_like(x) -> shape is same as x
        # Compare each element with the per-sample dropout probability p
        mask = (torch.rand_like(x) > p).to(x.dtype)

        # Scale by 1 / (1 - p) to preserve expected value
        # (This is the same scaling factor as in standard dropout)
        if self.inplace:
            # Inplace version
            x.mul_(mask).div_(1 - p)
            return x
        else:
            # Out-of-place version
            return x * mask / (1 - p), 1 - mask


def wass_dist(repr_f, treat_f, weights=None):
    repr0, repr1 = repr_f[treat_f.squeeze() == 0.0, :], repr_f[treat_f.squeeze() == 1.0]
    min_size = min(repr0.shape[0], repr1.shape[0])
    repr0, repr1 = repr0[:min_size, :], repr1[:min_size, :]
    M = ot.dist(repr0, repr1)

    if weights is None:
        w0, w1 = torch.ones(min_size), torch.ones(min_size)
    else:
        w0, w1 = weights[treat_f.squeeze() == 0.0, :], weights[treat_f.squeeze() == 1.0]
        w0, w1 = w0[:min_size, 0] + 1e-9, w1[:min_size, 0] + 1e-9

    wass_dist = ot.emd2(w0 / (w0.sum()), w1 / (w1.sum()), M)
    return wass_dist


def mmd_dist(repr_f, treat_f, weights=None):

    class RBF(nn.Module):

        def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
            super().__init__()
            self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
            self.bandwidth = bandwidth

        def get_bandwidth(self, L2_distances):
            if self.bandwidth is None:
                n_samples = L2_distances.shape[0]
                return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

            return self.bandwidth

        def forward(self, X):
            L2_distances = torch.cdist(X, X) ** 2
            return torch.exp(
                -L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(
                dim=0)

    class MMDLoss(nn.Module):

        def __init__(self, kernel=RBF()):
            super().__init__()
            self.kernel = kernel

        def forward(self, X, Y, w0, w1):
            K = self.kernel(torch.vstack([X, Y]))
            w0, w1 = w0.unsqueeze(1), w1.unsqueeze(1)
            X_size = X.shape[0]
            XX = (w0 * w0.T * K[:X_size, :X_size]).sum()
            XY = (w0 * w1.T * K[:X_size, X_size:]).sum()
            YY = (w1 * w1.T * K[X_size:, X_size:]).sum()
            return XX - 2 * XY + YY

    repr0, repr1 = repr_f[treat_f.squeeze() == 0.0, :], repr_f[treat_f.squeeze() == 1.0]
    min_size = min(repr0.shape[0], repr1.shape[0])
    repr0, repr1 = repr0[:min_size, :], repr1[:min_size, :]

    if weights is None:
        w0, w1 = torch.ones(min_size), torch.ones(min_size)
    else:
        w0, w1 = weights[treat_f.squeeze() == 0.0, :], weights[treat_f.squeeze() == 1.0]
        w0, w1 = w0[:min_size, 0] + 1e-9, w1[:min_size, 0] + 1e-9

    mmd_dist = MMDLoss()(repr0, repr1, w0 / (w0.sum()), w1 / (w1.sum()))
    return mmd_dist

