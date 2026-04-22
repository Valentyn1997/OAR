from pyro.nn import DenseNN
import logging
import torch
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger
from src.models.base_net import BaseNet

logger = logging.getLogger(__name__)


class MuNet(BaseNet):

    val_metric = 'val_rmse'
    name = 'mu_net_cov'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(MuNet, self).__init__(args, mlflow_logger)

        self.dim_hid1 = args.mu_net_cov.dim_hid1 = int(args.mu_net_cov.dim_hid1_multiplier * args.dataset.extra_hid_multiplier *
                                                       self.dim_cov)
        self.dim_repr = args.mu_net_cov.dim_repr = int(args.mu_net_cov.dim_repr_multiplier * args.dataset.extra_hid_multiplier *
                                                       self.dim_cov)
        self.dim_hid2 = args.mu_net_cov.dim_hid2 = int(args.mu_net_cov.dim_hid2_multiplier * args.dataset.extra_hid_multiplier *
                                                       self.dim_repr)
        self.hid_layers = args[self.name].hid_layers
        self.wd = args[self.name].wd

        self.out_scaler = StandardScaler()

        self.mu_net_type = args.mu_net_cov.mu_net_type

        self.repr_nn = DenseNN(self.dim_cov, self.hid_layers * [self.dim_hid1], param_dims=[self.dim_repr], nonlinearity=torch.nn.ELU()).float()
        # Mu net init
        if self.mu_net_type == 'tnet':
            self.mu_nn = torch.nn.ModuleList([DenseNN(self.dim_repr, self.hid_layers * [self.dim_hid2], param_dims=[1],
                                                  nonlinearity=torch.nn.ELU()).float() for _ in self.treat_options])
        elif self.mu_net_type == 'snet':
            self.mu_nn = DenseNN(self.dim_repr + 1, self.hid_layers * [self.dim_hid2], param_dims=[1], nonlinearity=torch.nn.ELU()).float()
        else:
            raise NotImplementedError()
        self.to(self.device)

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        # Torch tensors
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, data_dict['treat_f'], out_f)

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        self.num_train_iter = int(self.hparams.dataset.n_samples_train / self.batch_size * self.num_epochs)
        logger.info(f'Effective number of training iterations: {self.num_train_iter}.')

        return cov_f, treat_f, out_f

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.transform(data_dict['out_f'].reshape(-1, 1))

        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, data_dict['treat_f'], out_f)
        return cov_f, treat_f, out_f

    def get_train_dataloader(self, cov_f, treat_f, out_f):
        training_data = TensorDataset(cov_f, treat_f, out_f)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def forward_train(self, repr_f, treat_f, out_f, cov_f, prefix='train'):
        repr_f = self.repr_nn(cov_f)
        if self.mu_net_type == 'tnet':
            mu_pred = torch.stack([(treat_f == t) * self.mu_nn[int(t)](repr_f) for t in self.treat_options], dim=0).sum(0)
        elif self.mu_net_type == 'snet':
            context_f = torch.cat([repr_f, treat_f], dim=1)
            mu_pred = self.mu_nn(context_f)
        else:
            raise NotImplementedError()

        mse = ((mu_pred - out_f) ** 2)
        mse = mse.mean()

        results = {f'{prefix}_mse_mu_net_cov': mse.item()}
        return mse, results

    def forward_eval(self, cov_f):
        repr_f = self.repr_nn(cov_f)
        if self.mu_net_type == 'tnet':
            mu_pred_0, mu_pred_1 = self.mu_nn[0](repr_f), self.mu_nn[1](repr_f)
        elif self.mu_net_type == 'snet':
            context_0 = torch.cat([repr_f, torch.zeros((repr_f.shape[0], 1))], dim=1)
            context_1 = torch.cat([repr_f, torch.ones((repr_f.shape[0], 1))], dim=1)
            mu_pred_0, mu_pred_1 = self.mu_nn(context_0), self.mu_nn(context_1)
        else:
            raise NotImplementedError()
        return mu_pred_0, mu_pred_1

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)
        optimizer = self.get_optimizer()

        for step in tqdm(range(self.num_train_iter)) if log else range(self.num_train_iter):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            optimizer.zero_grad()

            loss, log_dict = self.forward_train(None, treat_f, out_f, cov_f, prefix='train')
            loss.backward()

            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        cov_f, treat_f, out_f = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            _, log_dict = self.forward_train(None, treat_f, out_f, cov_f, prefix=prefix)
            log_dict[f'{prefix}_rmse'] = np.sqrt(log_dict[f'{prefix}_mse_mu_net_cov'])

        if log:
            self.mlflow_logger.log_metrics(log_dict, step=self.num_train_iter)

        return log_dict

    def get_outcomes(self, data_dict):
        cov_f, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            mu_pred0, mu_pred1 = self.forward_eval(cov_f)
        return mu_pred0.cpu().numpy(), mu_pred1.cpu().numpy()

    def get_outcomes_predictions_cov(self, cov_f):
        cov_f = self.cov_scaler.transform(cov_f)
        cov_f = torch.tensor(cov_f).reshape(-1, self.dim_cov).float()

        self.eval()
        with torch.no_grad():
            mu_pred0, mu_pred1 = self.forward_eval(cov_f)
        return mu_pred0.cpu().numpy(), mu_pred1.cpu().numpy()
