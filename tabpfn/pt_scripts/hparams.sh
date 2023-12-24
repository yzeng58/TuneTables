
# nohup python -u train_loop.py --prior_type real --data_path "/home/colin/TabPFN-pt/tabpfn/data/openml__airlines__189354" --resume "/home/colin/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt" --prompt_tuning --tuned_prompt_size 1000 --lr 0.1 --save_every_k_epochs=30 --epochs 31 --aggregate_k_gradients 1 >> dec13.out &
# openml__airlines__189354  openml__albert__189356  openml__connect-4__146195  openml__higgs__146606  openml__poker-hand__9890


DATA_PATH="/home/colin/TabPFN-pt/tabpfn/data/openml__higgs__146606"
RESUME="/home/colin/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
EPOCHS=31

# Learning rate, gradient aggregation (epochs, AdamW)
# batch size, tuned_prompt_size

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.01 --aggregate_k_gradients 1 --batch_size 4 --seed 100 --wandb_name "higgs_lr01"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.2 --aggregate_k_gradients 1 --batch_size 4 --seed 100 --wandb_name "higgs_lr2"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 2 --batch_size 4 --seed 100 --wandb_name "higgs_agg2"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 8 --seed 100 --wandb_name "higgs_batch8"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 2 --seed 100 --wandb_name "higgs_batch2"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 4 --bptt 512 --seed 100 --wandb_name "higgs_bptt512"

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 4 --bptt 2048 --seed 100 --wandb_name "higgs_bptt2048"

