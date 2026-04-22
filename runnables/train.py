import logging
import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit
import pickle
from src.models.utils import subset_by_indices
from src import ROOT_PATH
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of dataset
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    data_dicts = dataset.get_data()
    if not args.dataset.collection:
        data_dicts = [data_dicts]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        # ================= Train / test split =================
        args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
        if args.dataset.train_test_splitted:
            data_dicts = [data_dict]
        else:
            ss = ShuffleSplit(n_splits=args.dataset.n_shuffle_splits, random_state=2 * args.exp.seed, test_size=args.dataset.test_size)
            indices = list(ss.split(data_dict['cov_f']))
            data_dicts = [(subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)) for (train_index, test_index) in indices]

        # ================= Train-validation split for downstream models =================
        for i, (train_data_dict, test_data_dict) in enumerate(data_dicts):

            # ================= Mlflow init =================
            experiment_name = f'{args.dataset.name}_reb'
            if args.target_net.regularization.efficient:
                experiment_name += '_d'

            mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                         tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

            # ================= Fitting nuisance nets =================

            prop_net_cov = instantiate(args.prop_net_cov, args, mlflow_logger, 'cov', _recursive_=True)

            # Finetuning for the first split
            if args.prop_net_cov.tune_hparams and i == 0:
                prop_net_cov.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.1})

            # Training
            logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
            prop_net_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_cov = prop_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

            # Getting propensity weights
            for data_dict in [train_data_dict, test_data_dict]:
                data_dict['prop_pred_cov'] = prop_net_cov.get_prop_predictions(data_dict=data_dict)

            if args.mu_net_cov.fit_mu:
                mu_net_cov = instantiate(args.mu_net_cov, args, mlflow_logger, _recursive_=True)

                # Finetuning for the first split
                if args.mu_net_cov.tune_hparams and i == 0:
                    mu_net_cov.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.1})

                # Training
                logger.info(f'Fitting mu-net for sub-dataset {args.dataset.dataset_ix}.')
                mu_net_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                # Evaluation
                results_out_cov = mu_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
                logger.info(f'Out-sample performance mu-net: {results_out_cov}')

                # Getting predicted mu0 and mu1
                for data_dict in [train_data_dict, test_data_dict]:
                    data_dict['mu_pred0_cov'], data_dict['mu_pred1_cov'] = mu_net_cov.get_outcomes(data_dict=data_dict)

            # Saving models
            if args.exp.save_results:
                save_dir = f'{ROOT_PATH}/reports/synthetic_krr_nonadaptive'
                np.save(f'{save_dir}/train_data_dict.npy', train_data_dict)
                np.save(f'{save_dir}/test_data_dict.npy', test_data_dict)

                torch.save(mu_net_cov, f'{save_dir}/mu_net_cov.pt')
                torch.save(prop_net_cov, f'{save_dir}/prop_net_cov.pt')

            # ================= Fitting target nets =================
            for target in args.exp.targets:

                logger.info(f'Fitting {target}-target net for sub-dataset {args.dataset.dataset_ix}.')
                target_model = instantiate(args.target_net, args, mlflow_logger, target, _recursive_=True)
                target_model.fit(train_data_dict=train_data_dict, log=args.exp.logging)


                results_in = target_model.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in_target', target=target)
                results_out = target_model.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out_target', target=target)

                if args.exp.plotting:
                    x = np.linspace(train_data_dict['cov_f'].min(), train_data_dict['cov_f'].max(), 100)
                    plt.plot(x, target_model.get_cate(x.reshape(-1, 1)), lw=2, c='r')
                    plt.plot(x, dataset.get_mu(x, 1) - dataset.get_mu(x, 0), lw=2, c='k')
                    plt.title(target + str(args.target_net.regularization.adaptive))

                    plt.show()

                if args.exp.save_results:
                    with open(f'{save_dir}/target_krr_{target}.pkl', 'wb') as f:
                        f.write(pickle.dumps(target_model))

                logger.info(f'MSE In-sample performance: {results_in}. MSE Out-sample performance: {results_out}')

            mlflow_logger.log_hyperparams(args) if args.exp.logging else None
            mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return results_in, results_out


if __name__ == "__main__":
    main()
