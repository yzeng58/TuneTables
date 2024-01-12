import time
from datetime import datetime
import argparse
import json
import os
import wandb
import optuna
import ConfigSpace

from train_loop import parse_args, reload_config, train_function, make_serializable
# from scripts.model_builder import get_model, save_model
# from scripts.model_configs import *
# from priors.utils import uniform_int_sampler_f
# from notebook_utils import *
from utils import get_wandb_api_key

def objective(trial):
    args = parse_args()
    config, model_string = reload_config(longer=1, args=args)
    for k, v in config.items():
        if isinstance(v, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            config[k] = trial.suggest_categorical(k, v.choices)
    # print("in objective, config")
    # print(make_serializable(config))

    #manually set optuna params
    config['bptt'] = trial.suggest_int('bptt', 128, 8192, log=True)
    config['lr'] = trial.suggest_float('lr', .0001, .3)
    config['aggregate_k_gradients'] = trial.suggest_int('aggregate_k_gradients', 1, 8)
    while config['bptt'] % config['aggregate_k_gradients'] != 0:
        config['aggregate_k_gradients'] -= 1
 
    print("Training model ...")

    if config['wandb_log']:
        wandb.login(key=get_wandb_api_key())
        simple_config = make_serializable(config)
        wandb.init(config=simple_config, name=model_string, group=config['wandb_group'],
                project=config['wandb_project'], entity=config['wandb_entity'])

    results_dict = train_function(config, 0, model_string)

    if config['wandb_log']:
        wandb.finish()

    return results_dict[args.optuna_objective]


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)