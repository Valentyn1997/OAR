from pyro.nn import DenseNN
import logging
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger
from src.models.base_net import BaseNet
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


class PropNet(BaseNet):

    val_metric = 'val_bce'
    name = 'prop_net'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, kind: str = None, **kwargs):
        self.kind = kind  # cov | repr
        assert self.kind in ['cov', 'repr']
        self.name = self.__class__.name + '_' + kind
        self.val_metric = self.__class__.val_metric + '_' + kind

        super(PropNet, self).__init__(args, mlflow_logger)

        self.dim_input = self.dim_cov
        self.dim_hid1 = args[self.name].dim_hid1 = int(args[self.name].dim_hid1_multiplier * args.dataset.extra_hid_multiplier *
                                                       self.dim_input)
        self.hid_layers = args[self.name].hid_layers
        self.wd = args[self.name].wd

        self.prop_nn = DenseNN(self.dim_input, self.hid_layers * [self.dim_hid1], param_dims=[1], nonlinearity=torch.nn.ELU()).float()

        self.to(self.device)

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))

        # Torch tensors
        cov_f, treat_f, _ = self.prepare_tensors(cov_f, data_dict['treat_f'], None)
        repr_f = torch.tensor(data_dict['repr_f']).float() if 'repr_f' in data_dict else None

        inp_f = cov_f if self.kind == 'cov' else repr_f

        self.hparams.dataset.n_samples_train = treat_f.shape[0]

        self.num_train_iter = int(self.hparams.dataset.n_samples_train / self.batch_size * self.num_epochs)
        logger.info(f'Effective number of training iterations: {self.num_train_iter}.')

        return inp_f, treat_f

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        cov_f, treat_f, _ = self.prepare_tensors(cov_f, data_dict['treat_f'], None)
        repr_f = torch.tensor(data_dict['repr_f']).float() if 'repr_f' in data_dict else None
        inp_f = cov_f if self.kind == 'cov' else repr_f

        mu0_unscaled = data_dict['mu0'].reshape(-1, 1)
        mu1_unscaled = data_dict['mu1'].reshape(-1, 1)

        return inp_f, treat_f, mu0_unscaled, mu1_unscaled

    def get_train_dataloader(self, inp_f, treat_f):
        training_data = TensorDataset(inp_f, treat_f)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def fit(self, train_data_dict: dict, log: bool):
        inp_f, treat_f = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(inp_f, treat_f)
        optimizer = self.get_optimizer()

        # Logging
        # self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        for step in tqdm(range(self.num_train_iter)) if log else range(self.num_train_iter):
            inp_f, treat_f = next(iter(train_dataloader))
            optimizer.zero_grad()

            prop_preds = self.prop_nn(inp_f)
            loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()
            log_dict = {f'train_bce_{self.kind}': loss.item()}

            loss.backward()

            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        inp_f, treat_f, mu0_unscaled, mu1_unscaled = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            prop_preds = self.prop_nn(inp_f)
            loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()

        prop_preds = torch.sigmoid(prop_preds).cpu().numpy()
        overlap_preds = prop_preds * (1 - prop_preds)
        tau = mu1_unscaled - mu0_unscaled
        tau_centered_abs = np.abs(tau - tau.mean())

        log_dict = {f'{prefix}_bce_{self.kind}': loss.item(),
                    f'{prefix}_mean_overlap': overlap_preds.mean().item(),
                    f'{prefix}_r_s': spearmanr(tau_centered_abs, overlap_preds).statistic,
                    f'{prefix}_r_p': pearsonr(tau_centered_abs, overlap_preds).statistic[0]}

        if log:
            self.mlflow_logger.log_metrics(log_dict, step=self.num_train_iter)

        return log_dict

    def get_prop_predictions(self, data_dict):
        inp_f, _, _, _ = self.prepare_eval_data(data_dict)
        self.eval()
        with torch.no_grad():
            prop_preds = self.prop_nn(inp_f)
        return torch.sigmoid(prop_preds).cpu().numpy()

    def get_prop_predictions_cov(self, cov_f):
        cov_f = self.cov_scaler.transform(cov_f)
        cov_f = torch.tensor(cov_f).reshape(-1, self.dim_cov).float()

        self.eval()
        with torch.no_grad():
            prop_preds = self.prop_nn(cov_f)
        return torch.sigmoid(prop_preds).cpu().numpy()
