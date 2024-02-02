#!/bin/bash
# sudo /opt/deeplearning/install-driver.sh; \

instance_repo_dir=/home/yyyyyyyy/TabPFN-pt

gcloud compute ssh --ssh-flag="-A" tabpfn-pt-20240118174904 --zone=us-central1-a --project=tabular-400321 \
    --command="\
    sudo -s; \
    /opt/deeplearning/install-driver.sh; \
    cd ${instance_repo_dir}; \
    source /home/bf996/.bashrc; \
    find . -type f -exec chmod 777 {} \;; \
    find . -type d -exec chmod 777 {} \;; \
    git config --global --add safe.directory /home/yyyyyyyy/TabPFN-pt; \
    cd ${instance_repo_dir}/tabpfn; \
    python train_loop.py --data_path /home/yyyyyyyy/TabPFN-pt/tabpfn/data/openml__dilbert__168909 --split 0 --real_data_qty 0 --wandb_group openml__dilbert__168909_pt1000-uniform-short_bptt_128_rdq_0_split_0 --prior_type real --pad_features --resume /home/yyyyyyyy/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt --epochs 7 --validation_period 3 --save_every_k_epochs 12 --aggregate_k_gradients 1 --lr 0.2 --feature_subset_method pca --wandb_log --do_preprocess --verbose --max_time 36000 --prompt_tuning --tuned_prompt_size 1000 --early_stopping 7 --uniform_bptt --bptt 128; \
    "