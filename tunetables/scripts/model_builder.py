from pathlib import Path
import argparse
from datetime import datetime
from functools import partial
import torch
import math
import numpy as np
import pdb
import tunetables.encoders as encoders
from tunetables.transformer import TransformerModel
from tunetables.utils import get_uniform_single_eval_pos_sampler, get_fixed_batch_sampler
import tunetables.priors as priors
from tunetables.train import train
from tunetables.losses import Losses

def save_model(model, path, filename, config_sample):
    config_sample = {**config_sample}

    def make_serializable(config_sample):
        if isinstance(config_sample, torch.Tensor):
            config_sample = "tensor"
        if isinstance(config_sample, dict):
            config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
        if isinstance(config_sample, list):
            config_sample = [make_serializable(v) for v in config_sample]
        if callable(config_sample):
            config_sample = str(config_sample)
        return config_sample

    config_sample = make_serializable(config_sample)
    target_path = os.path.join(path, filename)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    #Change permissions to allow group access
    os.chmod(target_path, 0o777)
    os.chmod(config_sample['base_path'], 0o777)
    try:
        #TODO: something about the target path is making the model unhappy
        torch.save((model.state_dict(), None, config_sample), target_path)
    except:
        # NOTE: This seems to work as long as you run the script from the base directory
        target_path = os.path.join("./models_diff", filename)
        torch.save((model.state_dict(), None, config_sample), target_path)



import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info

def load_model_only_inference(path, filename, device, prefix_size, n_classes):
    import tunetables.positional_encodings as positional_encodings
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """
    model_state, optimizer_state, config_sample = torch.load(os.path.join(path, filename), map_location='cpu')
    config_sample['prefix_size'] = prefix_size
    #TODO: check this, it was set to true in the training script
    config_sample['recompute_attn'] = True
    if (('nan_prob_no_reason' in config_sample and config_sample['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config_sample and config_sample['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config_sample and config_sample['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    n_out = config_sample['max_num_classes']
    print("max classes is", n_out)
    device = device if torch.cuda.is_available() else 'cpu:0'
    encoder = encoder(config_sample['num_features'], config_sample['emsize'])

    nhid = config_sample['emsize'] * config_sample['nhid_factor']
    y_encoder_generator = encoders.get_Canonical(config_sample['max_num_classes']) \
        if config_sample.get('canonical_y_encoder', False) else encoders.Linear

    assert config_sample['max_num_classes'] > 2
    loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.ones(int(config_sample['max_num_classes'])))
    # pos_enc = positional_encodings.NoPositionalEncoding(config_sample['emsize'], config_sample['bptt']*2)
    model = TransformerModel(encoder, n_out, config_sample['emsize'], 
                             config_sample['nhead'], nhid, 
                             config_sample['nlayers'], 
                             recompute_attn=config_sample['recompute_attn'],
                             y_encoder=y_encoder_generator(1, config_sample['emsize']),
                             dropout=config_sample['dropout'], 
                             # pos_encoder=pos_enc,
                             efficient_eval_masking=config_sample['efficient_eval_masking'], 
                             prefix_size=config_sample.get('prefix_size', 0),
                             n_classes=n_classes,
                             )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    if model_state.get('prefix_embedding.weight', None) is None and model.state_dict().get('prefix_embedding.weight', None) is not None:
            print('Loading prefix embedding')
            model_state['prefix_embedding.weight'] = model.state_dict()['prefix_embedding.weight']
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float('inf'), float('inf'), model), config_sample # no loss measured

def load_model(path, filename, device, eval_positions, verbose):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    print('!! Warning: GPyTorch must be installed !!')
    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location='cpu')
    if ('differentiable_hyperparameters' in config_sample
            and 'prior_mlp_activations' in config_sample['differentiable_hyperparameters']):
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values_used'] = config_sample[
                                                                                                         'differentiable_hyperparameters'][
                                                                                                         'prior_mlp_activations'][
                                                                                                         'choice_values']
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values'] = [
            torch.nn.Tanh for k in config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values']]

    config_sample['categorical_features_sampler'] = lambda: lambda x: ([], [], [])
    config_sample['num_features_used_in_training'] = config_sample['num_features_used']
    config_sample['num_features_used'] = lambda: config_sample['num_features']
    config_sample['num_classes_in_training'] = config_sample['num_classes']
    config_sample['num_classes'] = 2
    config_sample['batch_size_in_training'] = config_sample['batch_size']
    config_sample['batch_size'] = 1
    config_sample['bptt_in_training'] = config_sample['bptt']
    config_sample['bptt'] = 10
    config_sample['bptt_extra_samples_in_training'] = config_sample['bptt_extra_samples']
    config_sample['bptt_extra_samples'] = None

    #print('Memory', str(get_gpu_memory()))

    model = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model[2].load_state_dict(model_state)
    model[2].to(device)
    model[2].eval()

    return model, config_sample

def fix_loaded_config_sample(loaded_config_sample, config):
    def copy_to_sample(*k):
        t,s = loaded_config_sample, config
        for k_ in k[:-1]:
            t = t[k_]
            s = s[k_]
        t[k[-1]] = s[k[-1]]
    copy_to_sample('num_features_used')
    copy_to_sample('num_classes')
    copy_to_sample('differentiable_hyperparameters','prior_mlp_activations','choice_values')

def load_config_sample(path, template_config):
    model_state, optimizer_state, loaded_config_sample = torch.load(path, map_location='cpu')
    fix_loaded_config_sample(loaded_config_sample, template_config)
    return loaded_config_sample

def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [1000, 2000, 3000, 4000, 5000] # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max([X.shape[1] for (_, X, _, _, _, _) in test_datasets] + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets])
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits

def get_mlp_prior_hyperparameters(config):
    from tunetables.priors.utils import gamma_sampler_f
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


def get_gp_mix_prior_hyperparameters(config):
    return {'lengthscale_concentration': config["prior_lengthscale_concentration"],
            'nu': config["prior_nu"],
            'outputscale_concentration': config["prior_outputscale_concentration"],
            'categorical_data': config["prior_y_minmax_norm"],
            'y_minmax_norm': config["prior_lengthscale_concentration"],
            'noise_concentration': config["prior_noise_concentration"],
            'noise_rate': config["prior_noise_rate"]}

def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}


def get_meta_gp_prior_hyperparameters(config):
    from tunetables.priors.utils import trunc_norm_sampler_f
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if "outputscale_mean" in config:
        outputscale_sampler = trunc_norm_sampler_f(config["outputscale_mean"]
                                                   , config["outputscale_mean"] * config["outputscale_std_f"])
        config['outputscale'] = outputscale_sampler
    if "lengthscale_mean" in config:
        lengthscale_sampler = trunc_norm_sampler_f(config["lengthscale_mean"],
                                                   config["lengthscale_mean"] * config["lengthscale_std_f"])
        config['lengthscale'] = lengthscale_sampler

    return config


def get_model(config, device, should_train=True, verbose=False, state_dict=None, epoch_callback=None, is_wrapper = False, x_wrapper = None, y_wrapper = None, cat_idx = None):
    extra_kwargs = {}
    n_features = config['max_features']

    if 'aggregate_k_gradients' not in config or config['aggregate_k_gradients'] is None:
        config['aggregate_k_gradients'] = 1

    # config['num_steps'] = math.ceil(config['num_steps'] * config['aggregate_k_gradients'])
    # config['batch_size'] = math.ceil(config['batch_size'] / config['aggregate_k_gradients'])

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(batch_size, seq_len, num_features, hyperparameters
                , device, model_proto=model_proto
                , **kwargs):
            kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size
                , seq_len=seq_len
                , device=device
                , hyperparameters=hyperparameters
                , num_features=num_features, **kwargs)
        return new_get_batch

    #Real Data Training
    if config['prior_type'] == 'real':
        from tunetables.priors.real import TabularDataset

        print("is_wrapper",is_wrapper)

        if is_wrapper:
            num_classes = len(np.unique(y_wrapper))
            target_type = "classification"

            total_data_points = len(x_wrapper) 
            indices = np.arange(total_data_points)
            
            if config['epochs']==0: #only process test_loader
                train_indices = indices
                val_indices = indices
                num_classes = 2 #we just want the dataloader
            else:
                np.random.shuffle(indices)
                train_size = int(0.85 * total_data_points)
                eval_size = total_data_points - train_size
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]            

            split_indeces = [train_indices, val_indices]

            dataset = TabularDataset(name = "user_dataset",
                                            X = x_wrapper,
                                            y = y_wrapper,
                                            cat_idx = cat_idx,
                                            target_type = target_type,
                                            num_classes = num_classes,
                                            split_indeces = [train_indices, val_indices]
                                        )
            prior_hyperparameters = {}
            use_style = False
        else:
            dataset = TabularDataset.read(Path(config['data_path']).resolve())
            prior_hyperparameters = {}
            use_style = False

    #Priors == DataLoaders (synthetic)
    if config['prior_type'] == 'prior_bag':
        # Prior bag combines priors
        get_batch_gp = make_get_batch(priors.fast_gp)
        get_batch_mlp = make_get_batch(priors.mlp)
        if 'flexible' in config and config['flexible']:
            get_batch_gp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_gp})
            get_batch_mlp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_mlp})
        prior_bag_hyperparameters = {'prior_bag_get_batch': (get_batch_gp, get_batch_mlp)
            , 'prior_bag_exp_weights_1': 2.0}
        prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config)
            , **prior_bag_hyperparameters}
        model_proto = priors.prior_bag
    elif config['prior_type'] == 'real':
        pass
    else:
        if config['prior_type'] == 'mlp':
            prior_hyperparameters = get_mlp_prior_hyperparameters(config)
            model_proto = priors.mlp
        elif config['prior_type'] == 'gp':
            prior_hyperparameters = get_gp_prior_hyperparameters(config)
            model_proto = priors.fast_gp
        elif config['prior_type'] == 'gp_mix':
            prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
            model_proto = priors.fast_gp_mix
        else:
            raise Exception()

        if 'flexible' in config and config['flexible']:
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs['get_batch'] = get_batch_base
            model_proto = priors.flexible_categorical
    
    if config['prior_type'] == 'real':
        pass
    else:
        if config.get('flexible'):
            prior_hyperparameters['normalize_labels'] = True
            prior_hyperparameters['check_is_compatible'] = True
        prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config[
            'prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
        prior_hyperparameters['rotate_normalized_labels'] = config[
            'rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

        use_style = False

        if 'differentiable' in config and config['differentiable']:
            get_batch_base = make_get_batch(model_proto, **extra_kwargs)
            extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
            model_proto = priors.differentiable_prior
            use_style = True
        print(f"Using style prior: {use_style}")

    if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    # check_is_compatible = False if 'multiclass_loss_type' not in config else (config['multiclass_loss_type'] == 'compatible')

    epochs = 0 if not should_train else config['epochs']

    args = argparse.Namespace(**config)

    if config['prior_type'] == 'real':
        dataloader = dataset

        config['num_classes'] = len(set(dataloader.y))
        config['num_steps'] = None

    else:
        dataloader = model_proto.DataLoader

    if config['max_num_classes'] == 2:
        loss = Losses.bce
    elif config['max_num_classes'] > 2:
        loss = Losses.ce(config['max_num_classes'])
    elif config['prior_type'] == 'real':
        loss = Losses.ce(config['num_classes'])
    
    epkd = {
                        'prior_type': config['prior_type']
                        , 'num_features': n_features
                        , 'split': config['split']
                        , 'hyperparameters': prior_hyperparameters
                        , 'num_eval_fitting_samples': config.get('num_eval_fitting_samples', 1000)
                        , 'val_subset_size' : config.get('val_subset_size', 10)
                        , 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None)
                        , 'prompt_tuning': config.get('prompt_tuning', False)
                        , 'tuned_prompt_size': config.get('tuned_prompt_size', 0)
                        , 'model_string': config.get('model_string', datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
                        , 'save_path': config.get('base_path', '.')
                        , 'rand_seed': config.get('rand_seed', 135798642)
                        , 'summerize_after_prep': config.get('summerize_after_prep', False)
                        , 'average_ensemble': config.get('average_ensemble', False)
                        , 'permute_feature_position_in_ensemble': config.get('permute_feature_position_in_ensemble', False)
                        , 'bagging': config.get('bagging', False)
                        , 'private_model': config.get('private_model', False)
                        , 'private_data': config.get('private_data', False)
                        , 'epsilon': config.get('epsilon', 50)
                        , 'delta': config.get('delta', 1e-5)
                        , 'gradnorm': config.get('gradnorm', 1.2)
                        , 'tuned_prompt_label_balance': config.get('tuned_prompt_label_balance', 'equal')
                        , 'reseed_data': config.get('reseed_data', False)
                        , 'zs_eval_ensemble': config.get('zs_eval_ensemble', 0)
                        , 'pad_features': config.get('pad_features', False)
                        , 'early_stopping_patience': config.get('early_stopping_patience', 2)
                        , 'num_classes' : config.get('num_classes', 2)
                        , 'uniform_bptt': config.get('uniform_bptt', False)
                        , 'min_batches_per_epoch': config.get('min_batches_per_epoch', 10)
                        , 'keep_topk_ensemble': config.get('keep_topk_ensemble', 0)
                        , 'topk_key': config.get('topk_key', 'Val_Accuracy')
                        , 'do_preprocess' : config.get('do_preprocess', False)
                        , 'preprocess_type' : config.get('preprocess_type', 'none')
                        , 'wandb_log': config.get('wandb_log', False)
                        , 'shuffle_every_epoch': config.get('shuffle_every_epoch', False)
                        , 'real_data_qty': config.get('real_data_qty', False)
                        , 'max_time': config.get('max_time', 0)
                        , 'kl_loss': config.get('kl_loss', False)
                        , 'subset_rows_bagging': config.get('subset_rows_bagging', 0)
                        , 'bptt_search' : config.get('bptt_search', False)
                        , 'workers' : config.get('workers', 1)
                        , **extra_kwargs
    }

    if config['boosting'] or config.get('uniform_bptt', False):
        sep_samp = get_fixed_batch_sampler(config.get('bptt', 1024) + config.get('bptt_extra_samples', 128))
    else:
        sep_samp = get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']), min_len=config.get('min_eval_pos', 0))
        
    model, results_dict, data_for_fitting, test_loader = train(args
                  , dataloader
                  , loss
                  , encoder
                  , style_encoder_generator = encoders.StyleEncoder if use_style else None
                  , emsize=config['emsize']
                  , nhead=config['nhead']
                  # For unsupervised learning change to NanHandlingEncoder
                  , y_encoder_generator= encoders.get_Canonical(config['max_num_classes']) if config.get('canonical_y_encoder', False) else encoders.Linear
                  , pos_encoder_generator=None
                  , batch_size=config['batch_size']
                  , nlayers=config['nlayers']
                  , nhid=config['emsize'] * config['nhid_factor']
                  , epochs=epochs
                  , warmup_epochs=config['warmup_epochs']
                  , bptt=config['bptt']
                  , gpu_device=device
                  , dropout=config['dropout']
                  , steps_per_epoch=config['num_steps']
                  , single_eval_pos_gen=sep_samp
                  , load_weights_from_this_state_dict=state_dict
                  , validation_period=config['validation_period']
                  , aggregate_k_gradients=config['aggregate_k_gradients']
                  , recompute_attn=config['recompute_attn']
                  , epoch_callback=epoch_callback
                  , bptt_extra_samples = config['bptt_extra_samples']
                  , extra_prior_kwargs_dict=epkd
                  , lr=config['lr']
                  , verbose=config['verbose']
                  , boosting = config['boosting']
                  , boosting_lr = config.get('boosting_lr', 1e-3)
                  , boosting_n_iters = config.get('boosting_n_iters', 10)
                  , rand_init_ensemble = config.get('rand_init_ensemble', False)
                  , do_concat = config.get('concat_method', '')
                  , weight_decay=config.get('weight_decay', 0.0)
                  , is_wrapper = is_wrapper
                  , x_wrapper = x_wrapper, 
                  y_wrapper = y_wrapper)
    return model, results_dict, data_for_fitting, test_loader

    return model, results_dict