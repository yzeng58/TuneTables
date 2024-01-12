from collections import defaultdict
import copy

base_dict = {
    'prior_type' : 'real',
    'pad_features' : '',
    'resume' : '/home/benfeuer/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt',
    'epochs' : 31,
    'validation_period' : 3,
    'save_every_k_epochs' : 15,
    'aggregate_k_gradients' : 1,
    'lr' : 1e-3,
    'feature_subset_method' : 'pca',
    'wandb_log' : '',
    'do_preprocess' : '',
    'verbose' : '',
}

pt10_dict = {
        'prompt_tuning' : '',
        'tuned_prompt_size' : 10,
        'lr' : 0.1,
}

ens_bagged_avg_dict = {
        'epochs': 61,
        'save_every_k_epochs' : 62,
        'bagging' : '',
        'ensemble_lr' : 1.0,
        'average_ensemble' : '',
        'ensemble_size' : 5,
}

ens_randinit_avg_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_randinit_avg_dict.pop('bagging')
ens_randinit_avg_dict['rand_init_ensemble'] = ''

pt10_dict = dict(base_dict, **pt10_dict)
pt10_dict['save_every_k_epochs'] = base_dict['epochs'] + 1
pt10_uniform_dict = copy.deepcopy(pt10_dict)
pt10_uniform_dict['uniform_bptt'] = ''
pt10_powerall_dict = copy.deepcopy(pt10_dict)
pt10_powerall_dict['preprocess_type'] = 'power_all'
pt100_dict = copy.deepcopy(pt10_dict)
pt100_dict['tuned_prompt_size'] = 100
pt100_short_dict = copy.deepcopy(pt100_dict)
pt100_short_dict['epochs'] = 7
pt100_short_dict['save_every_k_epochs'] = 12
pt100_short_dict['early_stopping'] = 7
pt100_short_dict['validation_period'] = 3
pt100_short_dict['lr'] = 0.2
pt100_shorter_dict = copy.deepcopy(pt100_dict)
pt100_shorter_dict['epochs'] = 4
pt100_shorter_dict['save_every_k_epochs'] = 5
pt100_shorter_dict['early_stopping'] = 5
pt100_shorter_dict['validation_period'] = 3
pt100_shorter_dict['lr'] = 0.3
pt100_short_unif_dict = copy.deepcopy(pt100_short_dict)
pt100_short_unif_dict['uniform_bptt'] = ''
pt100_shorter_unif_dict = copy.deepcopy(pt100_shorter_dict)
pt100_shorter_unif_dict['uniform_bptt'] = ''
pt100_unif_dict = copy.deepcopy(pt100_dict)
pt100_unif_dict['uniform_bptt'] = ''
pt100_prop_dict = copy.deepcopy(pt100_dict)
pt100_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt100_pca_dict = copy.deepcopy(pt100_dict)
pt100_pca_dict['feature_subset_method'] = 'pca'
pt100_rand_dict = copy.deepcopy(pt100_dict)
pt100_rand_dict['feature_subset_method'] = 'random'
pt100_powerall_dict = copy.deepcopy(pt100_dict)
pt100_powerall_dict['preprocess_type'] = 'power_all'
pt1000_dict = copy.deepcopy(pt10_dict)
pt1000_dict['tuned_prompt_size'] = 1000
pt1000_unif_dict = copy.deepcopy(pt1000_dict)
pt1000_unif_dict['uniform_bptt'] = ''
pt1000_unif_highep_lowlr_dict = copy.deepcopy(pt1000_unif_dict)
pt1000_unif_highep_lowlr_dict['epochs'] = 101
pt1000_unif_highep_lowlr_dict['lr'] = 0.05
pt1000_unif_highep_lowlr_dict['save_every_k_epochs'] = 31
pt1000_unif_highep_lowlr_dict['validation_period'] = 5
pt1000_unif_highep_lowlr_dict['early_stopping'] = 10
pt1000_powerall_dict = copy.deepcopy(pt1000_dict)
pt1000_powerall_dict['preprocess_type'] = 'power_all'

ens_bagged_avg_dict = dict(pt1000_dict, **ens_bagged_avg_dict)
ens_randinit_avg_dict = dict(pt1000_dict, **ens_randinit_avg_dict)
ens_bagged_avg_top2_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_bagged_avg_top2_dict['keep_topk_ensemble'] = 2
ens_randinit_avg_top2_dict = copy.deepcopy(ens_randinit_avg_dict)
ens_randinit_avg_top2_dict['keep_topk_ensemble'] = 2
ens_bagged_avg_top2_powerall_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_powerall_dict['preprocess_type'] = 'power_all'
ens_bagged_avg_top2_reseed_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_reseed_dict['reseed_data'] = ''
ens_bagged_avg_top2_reseed_dict['ensemble_size'] = 10
ens_randinit_avg_top2_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_dict)
ens_randinit_avg_top2_reseed_dict['reseed_data'] = ''
ens_randinit_avg_top2_reseed_dict['ensemble_size'] = 10
ens_randinit_avg_top2_unif_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_unif_reseed_dict['uniform_bptt'] = ''

all_tasks = {
    'ft' : base_dict,
    'pt10' : pt10_dict,
    'pt10-powerall' : pt10_powerall_dict,
    'pt10-uniform' : pt10_uniform_dict,
    'pt100': pt100_dict,
    'pt100-prop' : pt100_prop_dict,
    'pt100-pca' : pt100_pca_dict,
    'pt100-rand' : pt100_rand_dict,
    'pt100-powerall' : pt100_powerall_dict,
    'pt100-uniform' : pt100_unif_dict,
    'pt100-uniform-short' : pt100_short_unif_dict,
    'pt100-uniform-shorter' : pt100_shorter_unif_dict,
    'pt100-short' : pt100_short_dict,
    'pt100-shorter' : pt100_shorter_dict,
    'pt1000': pt1000_dict,
    'pt1000-uniform' : pt1000_unif_dict,
    'pt1000-uniform-highep-lowlr' : pt1000_unif_highep_lowlr_dict,
    'pt1000-powerall' : pt1000_powerall_dict,
    'pt1000-5ens-bagged-avg' : ens_bagged_avg_dict,
    'pt1000-5ens-randinit-avg' : ens_randinit_avg_dict,
    'pt1000-5ens-bagged-avg-top2' : ens_bagged_avg_top2_dict,
    'pt1000-5ens-bagged-avg-top2-powerall' : ens_bagged_avg_top2_powerall_dict,
    'pt1000-5ens-randinit-avg-top2' : ens_randinit_avg_top2_dict,
    'pt1000-10ens-bagged-avg-top2-reseed' : ens_bagged_avg_top2_reseed_dict,
    'pt1000-10ens-randinit-avg-top2-reseed' : ens_randinit_avg_top2_reseed_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed' : ens_randinit_avg_top2_unif_reseed_dict,
}

def get_all_tasks():
    return all_tasks