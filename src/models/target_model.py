import numpy as np
from pyro.nn import DenseNN
import logging
import torch
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger
from src.models.base_net import BaseNet
from torch_ema import ExponentialMovingAverage
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist, pdist, squareform
import scipy

from src.models.utils import wass_dist, mmd_dist
from src.models.utils import AdaptiveDropout

logger = logging.getLogger(__name__)


class TargetModel:
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, target=None, **kwargs):
        # super(TargetModel, self).__init__(args, mlflow_logger, target=target)

        self.target = target
        assert target in ['cate', 'rcate', 'ivwcate']

        self.dim_cov = args.dataset.dim_cov
        self.cov_scaler = StandardScaler()
        self.treat_options = [0.0, 1.0]
        self.oracle_available = args.dataset.oracle_available
        self.dim_input = self.dim_cov

        self.regularization = args.target_net.regularization
        assert 0.0 <= self.regularization['adaptivity_coeff'] <= 1.0
        self.avg_reg_hparam = None
        self.reg_corr = None

        self.out_scaler = StandardScaler()

        self.q_trunc = args.target_net.q_trunc

        # MlFlow Logger
        self.mlflow_logger = mlflow_logger

        self.hparams = args


    def get_reg_hparams(self, prop_pred_cov, treat_f, reg_hparam_avg=None, clip_to_avg=True, return_bias_corr=False):
        reg_hparam = None
        bias_corr = None
        overlap = prop_pred_cov * (1 - prop_pred_cov)
        if self.regularization['type'] == 'dropout':
            if self.regularization['coeff'] == 'mult':
                reg_hparam = 1 - 4 * overlap
                if return_bias_corr:
                    bias_corr = - 4 * (treat_f - prop_pred_cov) * (1 - 2 * prop_pred_cov)

            elif self.regularization['coeff'] == 'mult2':
                reg_hparam = (1 - 16 * overlap ** 2)
                if return_bias_corr:
                    bias_corr = - 32 * overlap * (1 - 2 * prop_pred_cov) * (treat_f - prop_pred_cov)

            elif self.regularization['coeff'] == 'log':
                reg_hparam = 1 - 1 / (1 - (4 * overlap).log())
                if return_bias_corr:
                    bias_corr = - (treat_f - prop_pred_cov) * (1 - 2 * prop_pred_cov) / (1 - (4 * overlap).log()) ** 2 / overlap

        elif (self.regularization['type'] == 'noise') or (self.regularization['type'] == 'l2'):
            if self.regularization['coeff'] == 'mult':
                reg_hparam = 1 / (4 * overlap) - 1
                if return_bias_corr:
                    bias_corr = (2 * prop_pred_cov - 1) * (treat_f - prop_pred_cov) / overlap ** 2 / 4

            elif self.regularization['coeff'] == 'mult2':
                reg_hparam = 1 / (16 * overlap ** 2) - 1
                if return_bias_corr:
                    bias_corr = (2 * prop_pred_cov - 1) * (treat_f - prop_pred_cov) / overlap ** 3 / 8

            elif self.regularization['coeff'] == 'log':
                reg_hparam = - (4 * overlap).log() if isinstance(overlap, torch.Tensor) else - np.log(4 * overlap)
                if return_bias_corr:
                    bias_corr = (2 * prop_pred_cov - 1) * (treat_f - prop_pred_cov) / overlap

        elif self.regularization['type'] in ['none', 'bal']:
            return reg_hparam
        else:
            raise NotImplementedError()

        if clip_to_avg:
            reg_hparam[overlap < self.q_trunc ** 2] = reg_hparam[overlap >= self.q_trunc ** 2].mean() if reg_hparam_avg is None \
                else reg_hparam_avg
            if return_bias_corr:
                bias_corr[overlap < self.q_trunc ** 2] = 0.0

        if return_bias_corr:
            return reg_hparam, bias_corr
        else:
            return reg_hparam

    def prepare_train_reg(self, prop_pred_cov, treat_f):
        reg_hparam = self.get_reg_hparams(prop_pred_cov, treat_f)

        if self.regularization['type'] == 'dropout':
            # Scaling in [0, 1] and mean(.) == self.regularization['base_value']
            self.avg_reg_hparam = reg_hparam.mean()
            self.reg_corr = self.regularization['adaptivity_coeff'] * \
                            min(self.regularization['base_value'] / self.avg_reg_hparam,
                               (1 - self.regularization['base_value']) / (1.0 - self.avg_reg_hparam))
        elif (self.regularization['type'] == 'noise') or (self.regularization['type'] == 'l2'):
            # Scaling in [0, +infty] and mean(.) == self.regularization['base_value']
            self.avg_reg_hparam = reg_hparam.mean()
            self.reg_corr = self.regularization['adaptivity_coeff'] * self.regularization['base_value'] / self.avg_reg_hparam
        elif self.regularization['type'] in ['none', 'bal']:
            pass
        else:
            raise NotImplementedError()



class TargetKernelRidgeRegression(TargetModel):
    val_metric = None
    name = 'target_krr'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, target=None, **kwargs):

        super(TargetKernelRidgeRegression, self).__init__(args, mlflow_logger, target=target)

        assert self.regularization['type'] in ['none', 'l2']

        self.length_scale = args.target_net.length_scale
        self.krr = None
        self.w_pseudo_out_mean = 0.0
        self.alpha_y = self.alpha_f = self.alpha_overlap_corr = None


    def prepare_train_data(self, data_dict: dict):
        inp_f = self.cov_scaler.fit_transform(data_dict['cov_f'])

        self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        treat_f = data_dict['treat_f'].reshape(-1, 1)

        mu0_pred_cov = data_dict['mu_pred0_cov'].reshape(-1, 1)
        mu1_pred_cov = data_dict['mu_pred1_cov'].reshape(-1, 1)
        prop_pred_cov = data_dict['prop_pred_cov'].reshape(-1, 1)
        prop_pred_cov = 1 - prop_pred_cov if self.target == 'y0' else prop_pred_cov

        self.prepare_train_reg(prop_pred_cov, treat_f)

        ipwt0 = ((treat_f == 0.0) & ((1 - prop_pred_cov) >= self.q_trunc)) / (1 - prop_pred_cov + 1e-9)
        ipwt1 = ((treat_f == 1.0) & (prop_pred_cov >= self.q_trunc)) / (prop_pred_cov + 1e-9)
        pseudo_out = ipwt1 * (out_f - mu1_pred_cov) + mu1_pred_cov - (ipwt0 * (out_f - mu0_pred_cov) + mu0_pred_cov)

        return inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov

    def prepare_eval_data(self, data_dict: dict):
        inp_f = self.cov_scaler.fit_transform(data_dict['cov_f'])

        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        return inp_f, out_pot0, out_pot1, mu0, mu1

    @staticmethod
    def rbf_kernel(X, Y, length_scale):
        return np.exp(-0.5 * cdist(X, Y, metric="sqeuclidean") / length_scale)

    def fit(self, train_data_dict: dict, log: bool):
        inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = self.prepare_train_data(train_data_dict)


        alpha = 0.0
        if self.regularization['adaptive']:
            if self.regularization['type'] == 'l2':
                reg_hparam, reg_hparam_bias_corr = self.get_reg_hparams(prop_pred_cov, treat_f, self.avg_reg_hparam,
                                                                        return_bias_corr=True)
                alpha = self.regularization['base_value'] + self.reg_corr * (reg_hparam - self.avg_reg_hparam)

                if self.regularization['efficient']:
                    overlap = prop_pred_cov * (1 - prop_pred_cov)
                    clipping_mask = (overlap >= self.q_trunc ** 2).astype(float)
                    overlap_bias_corr = self.reg_corr * (reg_hparam_bias_corr * clipping_mask -
                                                         alpha * (clipping_mask * reg_hparam_bias_corr).mean() / self.avg_reg_hparam)
                    alpha = alpha + overlap_bias_corr
        else:
            alpha = self.regularization['base_value'] if self.regularization['type'] != 'none' else 0.0

        if self.target in ['cate']:
            # self.krr.fit(inp_f, pseudo_out)
            inp_f, y, w = inp_f, pseudo_out, np.ones_like(pseudo_out).reshape(-1)
        elif self.target in ['ivwcate']:
            overlap_eff = (treat_f - prop_pred_cov) ** 2
            # self.krr.fit(inp_f, pseudo_out, sample_weight=overlap_eff.reshape(-1))
            inp_f, y, w = inp_f, pseudo_out, overlap_eff.reshape(-1)
        elif self.target in ['rcate']:
            overlap_eff = (treat_f - prop_pred_cov) ** 2
            mu_pred_cov = prop_pred_cov * mu1_pred_cov + (1 - prop_pred_cov) * mu0_pred_cov
            r_pseudo_out = (out_f - mu_pred_cov) / (treat_f - prop_pred_cov + 1e-10)
            # self.krr.fit(inp_f, r_pseudo_out, sample_weight=overlap_eff.reshape(-1))
            inp_f, y, w = inp_f, r_pseudo_out, overlap_eff.reshape(-1)
        else:
            raise NotImplementedError()

        K = self.rbf_kernel(inp_f, inp_f, self.length_scale)
        W = np.diag(w)
        self.w_pseudo_out_mean = (y.reshape(-1) * w).mean() / w.mean()
        y_centered = y - self.w_pseudo_out_mean

        B = np.linalg.inv(W @ K + y.shape[0] * alpha * np.eye(y.shape[0]))
        self.alpha_y = B @ W @ y_centered
        self.inp_f_train = inp_f

        if self.regularization['efficient']:
            raise NotImplementedError()

    def predict(self, inp_f):
        K = self.rbf_kernel(inp_f, self.inp_f_train, self.length_scale)
        pseudo_out_pred = K @ self.alpha_y + self.w_pseudo_out_mean

        return pseudo_out_pred

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        inp_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)
        pseudo_out_pred = self.predict(inp_f)

        rpehe = np.sqrt(((pseudo_out_pred - (out_pot1 - out_pot0)) ** 2).mean())
        rpehe_oracle = np.sqrt((((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean()) if self.oracle_available else None

        results = {f'{prefix}_{target}_rpehe': rpehe.item(), f'{prefix}_{target}_rpehe_oracle': rpehe_oracle}
        if log:
            self.mlflow_logger.log_metrics(results, step=0)
        return results

    def get_cate(self, cov_f):
        cov_f = self.cov_scaler.transform(cov_f)
        pseudo_out_pred = self.predict(cov_f)
        return pseudo_out_pred * self.out_scaler.scale_


class TargetNet(BaseNet, TargetModel):

    val_metric = None
    name = 'target_net'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, target=None, **kwargs):

        BaseNet.__init__(self, args, mlflow_logger, target=target)
        TargetModel.__init__(self, args, mlflow_logger, target=target)

        assert self.regularization['type'] in ['none', 'dropout', 'noise', 'bal']

        self.hid_layers = args.target_net.hid_layers

        if args.target_net.dim_hid1_multiplier is None:
            self.dim_hid1 = int(args.mu_net_cov.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_input)
        else:
            self.dim_hid1 = int(args.target_net.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_input)

        if args.target_net.dim_repr_multiplier is None:
            self.dim_repr = args.mu_net_cov.dim_repr
        else:
            self.dim_repr = int(args.target_net.dim_repr_multiplier * args.dataset.extra_hid_multiplier * self.dim_input)

        if args.target_net.dim_hid2_multiplier is None:
            self.dim_hid2 = int(args.mu_net_cov.dim_hid2_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)
        else:
            self.dim_hid2 = int(args.target_net.dim_hid2_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)


        self.nn1 = DenseNN(self.dim_input,
                           self.hid_layers * [self.dim_hid1],
                           param_dims=[self.dim_repr], nonlinearity=torch.nn.ELU()).float()
        self.nn2 = DenseNN(self.dim_repr,
                           self.hid_layers * [self.dim_hid2],
                           param_dims=[1], nonlinearity=torch.nn.ELU()).float()

        self.gamma = args.target_net.gamma
        self.alpha = args.target_net.alpha
        self.ema_target = None

        self.to(self.device)

    def prepare_train_data(self, data_dict: dict):
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'])
        inp_f = torch.tensor(cov_f).float()

        self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        _, treat_f, out_f = self.prepare_tensors(None, data_dict['treat_f'], out_f)
        mu0_pred_cov = torch.tensor(data_dict['mu_pred0_cov'].reshape(-1, 1)).float()
        mu1_pred_cov = torch.tensor(data_dict['mu_pred1_cov'].reshape(-1, 1)).float()
        prop_pred_cov = torch.tensor(data_dict['prop_pred_cov'].reshape(-1, 1)).float()
        prop_pred_cov = 1 - prop_pred_cov if self.target == 'y0' else prop_pred_cov

        self.hparams.dataset.n_samples_train = inp_f.shape[0]
        self.num_train_iter = int(self.hparams.dataset.n_samples_train / self.batch_size * self.num_epochs)
        logger.info(f'Effective number of training iterations: {self.num_train_iter}.')

        self.prepare_train_reg(prop_pred_cov, treat_f)

        ipwt0 = ((treat_f == 0.0) & ((1 - prop_pred_cov) >= self.q_trunc)).float() / (1 - prop_pred_cov + 1e-9)
        ipwt1 = ((treat_f == 1.0) & (prop_pred_cov >= self.q_trunc)).float() / (prop_pred_cov + 1e-9)
        pseudo_out = ipwt1 * (out_f - mu1_pred_cov) + mu1_pred_cov - (ipwt0 * (out_f - mu0_pred_cov) + mu0_pred_cov)

        return inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov

    def prepare_eval_data(self, data_dict: dict):
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'])

        # Torch tensors
        inp_f = torch.tensor(cov_f).float()

        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        out_pot0 = torch.tensor(out_pot0).reshape(-1, 1).float()
        out_pot1 = torch.tensor(out_pot1).reshape(-1, 1).float()
        mu0 = torch.tensor(mu0).reshape(-1, 1).float() if self.oracle_available else None
        mu1 = torch.tensor(mu1).reshape(-1, 1).float() if self.oracle_available else None

        return inp_f, out_pot0, out_pot1, mu0, mu1

    def get_train_dataloader(self, *tensors):
        training_data = TensorDataset(*tensors)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        ema_target = ExponentialMovingAverage(self.parameters(), decay=self.gamma)
        return optimizer, ema_target

    def fit(self, train_data_dict: dict, log: bool):
        inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov)

        optimizer, self.ema_target = self.get_optimizer()

        for step in tqdm(range(self.num_train_iter)) if log else range(self.num_train_iter):

            inp_f, treat_f, out_f, pseudo_out, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = next(iter(train_dataloader))

            if self.regularization['type'] not in ['none', 'bal']:
                reg_hparam, reg_hparam_bias_corr = self.get_reg_hparams(prop_pred_cov, treat_f, self.avg_reg_hparam,
                                                                        return_bias_corr=True)
                if self.regularization['efficient']:
                    reg_hparam.requires_grad = True
                reg_hparam_corr = self.regularization['base_value'] + self.reg_corr * (reg_hparam - self.avg_reg_hparam)

            # Adaptive / constant dropout
            if self.regularization['type'] == 'noise':
                if self.regularization['adaptive']:
                    inp_f = inp_f + reg_hparam_corr * torch.randn_like(inp_f)
                    # if self.regularization['efficient']:
                    #     inp_f_eff = inp_f + reg_hparam_corr * torch.randn_like(inp_f)
                else:
                    inp_f = inp_f + self.regularization['base_value'] * torch.randn_like(inp_f)

            repr = self.nn1(inp_f)
            # if self.regularization['type'] == 'noise' and self.regularization['efficient']:
            #     repr_eff = self.nn1(inp_f_eff)

            # Adaptive / constant dropout
            if self.regularization['type'] == 'dropout':
                if self.regularization['adaptive']:
                    repr, drop_mask = AdaptiveDropout()(repr, reg_hparam_corr)
                    # if self.regularization['efficient']:
                    #     repr_eff, drop_mask = AdaptiveDropout()(repr, reg_hparam_corr)
                else:
                    repr = torch.nn.Dropout(p=self.regularization['base_value'])(repr)

            if self.regularization['type'] == 'bal':
                if self.regularization['coeff'] == 'wm':
                    ipm = wass_dist(repr, treat_f)
                elif self.regularization['coeff'] == 'mmd':
                    ipm = mmd_dist(repr, treat_f)
                else:
                    raise NotImplementedError()

            pred = self.nn2(repr)

            if self.target in ['cate']:
                loss = ((pred - pseudo_out) ** 2).mean()
                log_dict = {f'train_mse_target': loss.item()}
                if self.regularization['efficient']:
                    loss_weights = 1.0

            elif self.target in ['ivwcate']:
                overlap_eff = (treat_f - prop_pred_cov) ** 2
                loss = (overlap_eff * (pred - pseudo_out) ** 2).mean()
                log_dict = {f'train_wmse_target': loss.item()}
                if self.regularization['efficient']:
                    loss_weights = prop_pred_cov * (1 - prop_pred_cov)  # (treat_f - prop_pred_cov) ** 2

            elif self.target in ['rcate']:
                mu_pred_cov = prop_pred_cov * mu1_pred_cov + (1 - prop_pred_cov) * mu0_pred_cov
                loss = (((out_f - mu_pred_cov) - (treat_f - prop_pred_cov) * pred) ** 2).mean()
                log_dict = {f'train_rloss_target': loss.item()}
                if self.regularization['efficient']:
                    loss_weights = prop_pred_cov * (1 - prop_pred_cov)

            else:
                raise NotImplementedError()

            if self.regularization['type'] == 'bal':
                loss += self.regularization['base_value'] * ipm

            if self.regularization['efficient']:
                # Getting the gradient of g(.) wrt. regularization
                plug_in_loss = loss_weights * (mu1_pred_cov - mu0_pred_cov - pred) ** 2
                plug_in_loss_grad = 2 * loss_weights * (pred - mu1_pred_cov + mu0_pred_cov)
                pred.backward(torch.ones_like(reg_hparam_corr), create_graph=True, retain_graph=True)
                # pred.backward(torch.ones_like(reg_hparam_corr), create_graph=True, retain_graph=True)

                overlap = prop_pred_cov * (1 - prop_pred_cov)
                clipping_mask = (overlap >= self.q_trunc ** 2).float()

                eps = 1e-10
                reg_hparam_grad = reg_hparam.grad
                reg_hparam_grad_norm = torch.sqrt((reg_hparam_grad ** 2).mean()).detach()

                # if reg_hparam_grad_norm > self.regularization['bias_corr_threshold']:
                #     reg_hparam_grad = reg_hparam_grad * self.regularization['bias_corr_threshold'] / reg_hparam_grad_norm

                grad = plug_in_loss_grad * reg_hparam_grad

                term1 = (grad * reg_hparam_bias_corr).sum() / clipping_mask.sum()

                if self.regularization['type'] == 'noise':
                    term2 = (grad * reg_hparam_bias_corr * reg_hparam).sum() / clipping_mask.sum() / self.avg_reg_hparam
                    term3 = (grad * reg_hparam ** 2).sum() / clipping_mask.sum() / self.avg_reg_hparam
                    term4 = (grad * reg_hparam).sum() / clipping_mask.sum()
                    bias_corr = term1 - term2 - term3 + term4
                elif self.regularization['type'] == 'dropout':
                    bern_score = (drop_mask / (reg_hparam_corr - 1 + eps) + (1 - drop_mask) / (reg_hparam_corr + eps))#.mean(1, keepdim=True)
                    term1_score = self.reg_corr * (plug_in_loss * reg_hparam_bias_corr * bern_score).mean(1).sum() / clipping_mask.sum()
                    if (self.regularization['base_value'] / self.avg_reg_hparam) < ((1 - self.regularization['base_value']) / (1.0 - self.avg_reg_hparam)):
                        term2 = (grad * reg_hparam_bias_corr * reg_hparam).sum() / clipping_mask.sum() / self.avg_reg_hparam
                        term2_score = self.reg_corr * (reg_hparam_bias_corr * reg_hparam * plug_in_loss * bern_score).mean(1).sum() / clipping_mask.sum() / self.avg_reg_hparam
                        term3 = (grad * reg_hparam ** 2).sum() / clipping_mask.sum() / self.avg_reg_hparam
                        term3_score = self.reg_corr * (plug_in_loss * bern_score * reg_hparam ** 2).mean(1).sum() / clipping_mask.sum() / self.avg_reg_hparam
                        term4 = (grad * reg_hparam).sum() / clipping_mask.sum()
                        term4_score = (plug_in_loss * bern_score * reg_hparam).mean(1).sum() / clipping_mask.sum()
                    else:
                        term2 = (grad * reg_hparam_bias_corr * (1 - reg_hparam)).sum() / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                        term2_score = self.reg_corr * (reg_hparam_bias_corr * (1 - reg_hparam) * plug_in_loss * bern_score).mean(1).sum() / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                        term3 = (grad * (1 - reg_hparam) * reg_hparam).sum() / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                        term3_score = self.reg_corr * (plug_in_loss * bern_score * reg_hparam * (1 - reg_hparam)).mean(1).sum() / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                        term4 = (grad * (1 - reg_hparam)).sum() * self.avg_reg_hparam / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                        term4_score = self.reg_corr * (plug_in_loss * bern_score * (1 - reg_hparam)).mean(1).sum() * self.avg_reg_hparam / clipping_mask.sum() / (1 - self.avg_reg_hparam)
                    bias_corr = term1 + term1_score - term2 - term2_score - term3 - term3_score + term4 + term4_score
                else:
                    raise NotImplementedError()

                # lin_rise = (step + 1) / self.num_train_iter
                bias_corr[(bias_corr.abs() > self.regularization['bias_corr_threshold']) | (bias_corr.abs() > loss)] = 0.0
                loss += self.alpha * bias_corr
                log_dict[f'train_{self.target}_bias_corr'] = bias_corr.item()

            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            self.ema_target.update()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        inp_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            with self.ema_target.average_parameters():
                pseudo_out_pred = self.nn2(self.nn1(inp_f))

            rpehe = ((pseudo_out_pred - (out_pot1 - out_pot0)) ** 2).mean().sqrt()
            rpehe_oracle = (((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean().sqrt() if self.oracle_available else None

        results = {f'{prefix}_{target}_rpehe': rpehe.item(), f'{prefix}_{target}_rpehe_oracle': rpehe_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results

    def get_cate(self, cov_f):
        cov_f = self.cov_scaler.transform(cov_f)
        cov_f = torch.tensor(cov_f).reshape(-1, self.dim_cov).float()

        self.eval()
        with torch.no_grad():
            with self.ema_target.average_parameters():
                pseudo_out_pred = self.nn2(self.nn1(cov_f))

        return pseudo_out_pred.cpu().numpy() * self.out_scaler.scale_
