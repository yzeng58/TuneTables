import time
from datetime import datetime
import argparse

from scripts.model_builder import get_model, save_model
from scripts.model_configs import *
from priors.utils import uniform_int_sampler_f
from notebook_utils import *

def train_function(config_sample, i=0, add_name=''):
    start_time = time.time()
    save_every_k = 5
    epochs = []
    
    def save_callback(model, epoch):
        #NOTE: I think the 'epoch' value is actually 1 / config['epochs']
        epochs.append(epoch)
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        if len(epochs) % save_every_k == 0:
            print('Saving model..')
            config_sample['epoch_in_training'] = epoch
            save_model(model, config_sample['base_path'], f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt',
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
            print("Done saving.")
    model = get_model(config_sample
                      , config_sample["device"]
                      , should_train=True
                      , verbose=1
                      , state_dict=config_sample["state_dict"]
                      , epoch_callback = save_callback)
    
    return

def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = ''
    
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string

def train_loop():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to resume from.')
    parser.add_argument('--save_path', type=str, default=".", help='Path to save new checkpoints.')
    args = parser.parse_args()

    config, model_string = reload_config(longer=1)

    if args.resume is not None:
        model_state, optimizer_state_load, config_sample_load = torch.load(args.resume, map_location='cpu')
        module_prefix = 'module.'
        config["state_dict"] = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    else:
        config["state_dict"] = None

    config['bptt_extra_samples'] = None

    # config["large_datasets"] = True
    # config["max_samples"] = 10000 if config["large_datasets"] else 5000
    # config["bptt"] = 10000 if config["large_datasets"] else 3000
    # config["suite"]='cc'
    # config["max_features"] = 100

    config["device"] = 'cuda'
    config["base_path"] = args.save_path

    # diff
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['sampling'] = 'normal' # vielleicht schlecht?
    del config['differentiable_hyperparameters']['sampling']

    config['pre_sample_causes'] = True
    # end diff

    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['normalize_to_ranking'] = False # False

    config['categorical_feature_p'] = .2 # diff: .0

    # turn this back on in a random search!?
    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.

    config['normalize_with_sqrt'] = False

    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None

    config['normalize_ignore_label_too'] = False

    config['differentiable_hps_as_style'] = False
    config['max_eval_pos'] = 1000

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['bptt'] = 1024+128
    config['canonical_y_encoder'] = False

        
    config['aggregate_k_gradients'] = 8
    config['batch_size'] = 8*config['aggregate_k_gradients']
    config['num_steps'] = 1024//config['aggregate_k_gradients']
    config['epochs'] = 400
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    #TODO: check whether config_sample should be iterated within train_function
    config_sample = evaluate_hypers(config)

    config_sample['batch_size'] = 4

    print("Training model ...")

    train_function(config_sample, 0, model_string)

    print("Done")

if __name__ == '__main__':
    train_loop()