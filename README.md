# Overlap-Adaptive Regularization
Tackling the issue of low overlap in meta-learners with adaptive regularization.

[![Conference](https://img.shields.io/badge/ICLR2026-Paper-blue])](https://openreview.net/forum?id=HMMSnGgYOy)
[![arXiv](https://img.shields.io/badge/arXiv-2509.24962-b31b1b.svg)](https://arxiv.org/abs/2509.24962)

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/67fe5f83-2559-4b47-935c-7f593f3a2993" />

The project is built with the following Python libraries:
1. [PyTorch](https://pytorch.org/)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking


## Setup

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000 --gunicorn-opts "--timeout 280"`

To access the MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

### Semi-synthetic datasets setup

Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition): to `data/acic2016/`
```
 ── data/acic_2016
    ├── synth_outcomes
    |   ├── zymu_<id0>.csv   
    |   ├── ... 
    │   └── zymu_<id14>.csv 
    ├── ids.csv
    └── x.csv 
```


## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments, see the main configuration file `config/config.yaml` and other files in `config/` folder.



Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +model=<model> exp.seed=10
```


### Datasets
One needs to specify a dataset / dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=250`, or a subset index for ACIC 2016 data `dataset.dataset_ix=0`):
- Synthetic data (adapted from https://arxiv.org/abs/1810.02894): `+dataset=synthetic`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp100` 
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition) dataset: `+dataset=acic2016`
- [HC-MNIST](https://arxiv.org/abs/2103.04850) dataset: `+dataset=hcmnist`


### Models
Models already have the best hyperparameters saved, for each model - dataset pair. One can access them via: `+<dataset>_hparams=<dataset>` or `+<dataset>_hparams=<dataset_ix>` etc. 

#### Stage 1.
Stage 1 models are propensity networks (src/models/prop_nets.py) and outcome networks (src/models/mu_nets.py). To perform manual hyperparameter tuning, use the flags `prop_net_cov.tune_hparams=True` and `mu_net_cov.tune_hparams=True`.

#### Stage 2.
Stage 2 models are defined in config/config.yaml and src/models/target_model.py. One needs to specify a second-stage model (target net `+model=target_net` or target kernel ridge regression `+model=target_krr`) and specific parameters of the regularization:
- `target_net.regularization.adaptive`: `False` - constant regularization, `True` - overlap-adaptive regularization (OAR)
- `target_net.regularization.type`: `noise` - noise regularization (for `+model=target_net`), `dropout` - dropout (for `+model=target_net`), `l2` - RKHS norm (for `+model=target_krr`)
- `target_net.regularization.coeff`: `mult` - multiplicative regularization function ($\lambda_{\mathrm{m}}$), `mult` - logarithmic regularization function ($\lambda_{\log}$), `mult2` - squared multiplicative regularization function ($\lambda_{\mathrm{m}^2}$)
- `target_net.regularization.efficient`: `False` - OAR, `True` - dOAR
- `target_net.regularization.base_value`:  constant value of regularization $\lambda/p$ for rescaling

### Examples
Example of running target net with OAR($\lambda_{\mathrm{m}}$) noise regularization w/o hyper-parameter tuning, rescaled to $\lambda = 0.5$, based on synthetic data:
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/train.py -m +dataset=synthetic +model=target_net +synthetic_hparams=\'250\' exp.logging=True exp.device=cuda exp.seed=10 target_net.regularization.adaptive=True target_net.regularization.type=noise target_net.regularization.coeff=mult target_net.regularization.efficient=False target_net.regularization.base_value=0.5
```

Example of running target net with dOAR($\lambda_{\log}$) dropout w/ hyper-parameter tuning, rescaled to $\lambda = 0.1$, based on the IHDP dataset (first subset):
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/train.py -m +dataset=ihdp100 +model=target_net +ihdp100_hparams=ihdp100 exp.logging=True exp.device=cuda exp.seed=10 dataset.dataset_ix=0 prop_net_cov.tune_hparams=True mu_net_cov.tune_hparams=True target_net.regularization.adaptive=True target_net.regularization.type=dropout target_net.regularization.coeff=log target_net.regularization.efficient=True target_net.regularization.base_value=0.1 
```

