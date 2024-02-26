import subprocess
import asyncio
import os
import time
import datetime
import argparse
import json
import shutil
import copy
from pathlib import Path

from tqdm.auto import tqdm

from all_tasks import get_all_tasks

import wandb
import torch

MAX_CLASSES = 10
MAX_FEATURES = 100
MAX_SAMPLES = 3000

def is_json_serializable(obj):
    """
    Test if an object is JSON serializable.

    Args:
    obj (any): The object to test for JSON serialization.

    Returns:
    bool: True if the object is JSON serializable, False otherwise.
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def make_serializable(config_sample):
    if isinstance(config_sample, torch.Tensor):
        config_sample = "tensor"
    if isinstance(config_sample, dict):
        config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
    if isinstance(config_sample, list):
        config_sample = [make_serializable(v) for v in config_sample]
    if callable(config_sample):
        config_sample = str(config_sample)
    if not is_json_serializable(config_sample):
        config_sample = str(config_sample)
    return config_sample


def get_wandb_api_key(api_key_file="./config/wandb_api_key.txt"):
    # todo: if we make a config folder, put wandb_api_key.txt into the config folder
    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()
    
def wandb_init(config, model_string):
    mkey = get_wandb_api_key()
    wandb.login(key=mkey)
    simple_config = make_serializable(config)
    if simple_config['state_dict'] is not None:
        simple_config['state_dict'] = 'omitted'
    wandb.init(config=simple_config, name=model_string, group=config['wandb_group'],
            project=config['wandb_project'], entity=config['wandb_entity'])

async def run_command(cmd):
    # Start the subprocess
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # Wait for the command to finish
    stdout, stderr = await process.communicate()

    return process.returncode, stdout, stderr

def main_f(args):

    def run_tunetables(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt, do_wandb):
        if task == "tunetables-long":
            UPPER_CUTOFF = 1e10
        elif task == "tunetables-short":
            UPPER_CUTOFF = 10000
        elif "tunetables" in task:
            UPPER_CUTOFF = 100000
            print(f"Using default upper cutoff of 100000 for task {task}")
        args.real_data_qty = MAX_SAMPLES
        metadata_path = Path(dataset_path[1:-1]) / 'metadata.json'
        with open(metadata_path) as f:
            metadata = json.load(f)
        n_classes = metadata['num_classes']
        n_features = metadata['num_features']
        n_samples = metadata['num_instances']
        all_res = {}
        all_res_d = {}
        if n_features > MAX_FEATURES:
            print("Sweeping feature subselection methods.")
            #NOTE: Other options: zs-pca_white-32, zs-isomap-32, zs-ica-32, zs-random-32, zs-sparse_random_projection-32
            tt_tasks = ['zs-pca-16', 'zs-mutual_information-16']
            for task in tt_tasks:
                try:
                    res, _ = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
                    all_res[task] = res["Val_Accuracy"]
                    all_res_d[task] = res
                except:
                    pass
            best_task = max(all_res, key=all_res.get)
            feat_sel_method = best_task.split('-')[1]
        else:
            feat_sel_method = ''
        if n_classes > 25:
            raise NotImplementedError("Please add a task to all_tasks for the correct number of classes (modify task pt1000-10ens-randinit-avg-top2-unif-reseed-25cl-long).")
        if n_classes > MAX_CLASSES and n_classes < 25:
            tt_tasks = [f'pt1000-10ens-randinit-avg-top2-unif-reseed-25cl-long']
        elif n_samples <= MAX_SAMPLES:
            if feat_sel_method == '':
                tt_tasks = [
                    'zs-random-32',
                    'zs-preproc-random-32',
                    'pt10-short-lowlr-prop',
                    'pt10-uniform-kl-nopp',
                    'pt10-uniform-kl-nopp-prop',
                ]
            else:
                tt_tasks = [f'pt10-short-lowlr-prop-{feat_sel_method}', f'pt10-uniform-kl-nopp-{feat_sel_method}', f'pt10-uniform-kl-nopp-prop-{feat_sel_method}']
        elif n_samples > UPPER_CUTOFF:
            if feat_sel_method != '':
                tt_tasks = [
                    f'pt1000-{feat_sel_method}', 
                    f'pt1000-sumafter-{feat_sel_method}', 
                    f'pt1000-uniform-{feat_sel_method}', 
                    f'pt1000-uniform-sumafter-{feat_sel_method}'
                ]
            else:
                tt_tasks = [
                    'pt1000', 
                    'pt1000-uniform',
                ]
        else:
            if feat_sel_method != '':
                tt_tasks = [f'zs-{feat_sel_method}-32',
                            f'zs-preproc-{feat_sel_method}-32',
                            f'pt1000-10ens-randinit-avg-top2-unif-reseed-{feat_sel_method}', 
                            f'pt1000-10ens-randinit-avg-top2-unif-reseed-sumafter-{feat_sel_method}',
                            #f'pt100-10ens-randinit-avg-top2-reseed-{feat_sel_method}',
                            #f'pt100-10ens-randinit-avg-top2-reseed-sumafter-{feat_sel_method}',
                            ]
            else:
                tt_tasks = [
                            'zs-random-32',
                            'zs-preproc-random-32',
                            'pt1000-10ens-randinit-avg-top2-unif-reseed',
                            # 'pt1000-10ens-randinit-avg-top2-reseed',
                            ]
        if args.verbose:
            print("For dataset", dataset_path, "split", split, "with", n_classes, "classes, and", n_features, "features, and", n_samples, "samples, running tasks:", tt_tasks)
        
        
        # args.bptt_backup = args.bptt
        
        #wandb logging for tunetables meta-optimization
        if do_wandb:
            model_string = task_str = "tunetables" + '_split_' + str(split)
            wandb_group = dataset.strip() + "_" + task_str
            config = dict()
            config['wandb_group'] = wandb_group
            config['wandb_project'] = args.wandb_project
            config['wandb_entity'] = args.wandb_entity
            config['tt_tasks'] = tt_tasks
            config['dataset_path'] = dataset_path
            config['dataset'] = dataset.strip()
            config['split'] = split
            config['n_classes'] = n_classes
            config['n_features'] = n_features
            config['n_samples'] = n_samples
            config['upper_cutoff'] = UPPER_CUTOFF
            config['state_dict'] = None
            wandb_init(config, model_string)
            
        start_time = time.time()
        for i, task in enumerate(tt_tasks):
            
            if all_res_d.get(task, None) is not None:
                continue
            # args.bptt = args.bptt_backup
            # if 'unif' in task:
            #     args.bptt_backup = args.bptt
            #     args.bptt = 128
            res, _ = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
            if do_wandb:
                wandb.log(res, step=i, commit=True)
            if args.verbose:
                print("Best epoch results for", dataset.strip(), "split", split, "task", task.strip(), ":", res)
            all_res_d[task] = res
            all_res[task] = max(res.get("Val_Accuracy", 0.0), res.get("Val_nc_Accuracy", 0.0), res.get("Ens_Val_Accuracy", 0.0), res.get("Ens_Val_Accuracy_NC", 0.0))
        best_task = max(all_res, key=all_res.get)
        time_taken = time.time() - start_time
        if do_wandb:
            wandb.log({"tunetables_runtime": time_taken})
            wandb.finish()
        return all_res_d[best_task], best_task
        

    def run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt):
        # Get task name
        task = task.strip()
        task_str = task
        if args.run_optuna:
            task_str += '_optuna'
        if args.bptt > -1:
            task_str += '_bptt_' + str(args.bptt)
        if args.shuffle_every_epoch:
            task_str += '_shuffleep_'
        task_str += '_rdq_' + str(args.real_data_qty)
        task_str += '_split_' + str(split)
        if args.epochs > 0:
            task_str += '_epochs_' + str(args.epochs)
        if task.startswith('zs'):
            #zero-shot logic
            ensemble_size = int(task.split('-')[-1])
            subset_ft_method = task.split('-')[-2]
            command = ['python', base_cmd, 
                    '--data_path', dataset_path,
                    '--subset_features_method', subset_ft_method,
                    '--split', str(split),
                    '--real_data_qty', str(args.real_data_qty),
                    '--zs-eval-ensemble', str(ensemble_size),
                    '--workers', "1"]
            if "preproc" in task:
                command.append("--do_preprocess")
            if args.wandb_log:
                command = command + [
                    '--wandb_log',
                    '--wandb_group', "\"" + dataset.strip() + "_" + task_str + "_" + subset_ft_method + "\"", 
                ]
        else:
            # Get task args
            npp = False
            npad = False
            if '-npad' in task:
                npad = True
                task = task.replace('-npad', '')
            if '-nopreproc' in task:
                npp = True
                task = task.replace('-nopreproc', '')
            next_task = all_tasks[task]
            if not args.wandb_log:
                try:
                    next_task.pop('wandb_log')
                except:
                    pass
            if args.wandb_project != '':
                next_task['wandb_project'] = args.wandb_project
            if args.resume != '':
                next_task['resume'] = args.resume
            if args.epochs > 0:
                next_task['epochs'] = args.epochs
            if args.validation_period > 0:
                next_task['validation_period'] = args.validation_period
            if npp:
                try:
                    next_task.pop('do_preprocess')
                except:
                    pass
                task_str += '_nopreproc'
            if npad:
                try:
                    next_task.pop('pad_features')
                except:
                    pass
                task_str += '_npad'
            addl_args = []
            for k, v in next_task.items():
                addl_args.append("--" + k)
                val = str(v)
                if val != '':
                    addl_args.append(val)
            command = ['python', base_cmd, '--data_path', dataset_path, '--split', str(split), '--real_data_qty', str(args.real_data_qty), '--wandb_group', "\"" + dataset.strip() + "_" + task_str + "\""] + addl_args
            if args.run_optuna:
                if args.wandb_log and args.wandb_project == '':
                    command = command + ["--wandb_project", args.wandb_project] 
                elif args.wandb_log:
                    command = command + ["--wandb_project", "tabpfn-pt-optuna"]
        if args.bptt > -1:
            command.append("--bptt")
            command.append(str(args.bptt))     
        if args.shuffle_every_epoch:
            command.append("--shuffle_every_epoch")
        if args.verbose:
            command.append("--verbose")
            print("Running command:", ' '.join(command))
        job_str = "\'" + ' '.join(command) + '\'\n'
        if args.gcp_run:
            return {}, job_str
        else:
            returncode, stdout, stderr = asyncio.run(run_command(' '.join(command)))
            stdout = stdout.decode()
            print("Stderr:", stderr.decode())
            if args.print_stdout:
                print("Stdout:", stdout)
            # Initialize an empty dictionary to hold the parsed output
            output_dict = {}
            # Define the marker indicating the start of the JSON output
            json_start_marker = "^RESULTS\n"
            # Check if the marker is in the stdout
            if json_start_marker in stdout:
                # Extract the JSON string part. Assume the JSON starts immediately after the marker
                json_str = stdout.split(json_start_marker, 1)[1]
                
                # Attempt to parse the JSON string into a Python dictionary
                try:
                    output_dict = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    output_dict = {}
            # Parse and relocate logs
            new_outputs = Path('logs').glob('_multiclass*')
            updated_outputs = []
            for output in new_outputs:
                new_name = output.name.replace('_multiclass', task_str)
                new_path = os.path.join(output.parent, new_name)
                os.rename(output, new_path)
                updated_outputs.append(new_path)
            for output in updated_outputs:
                shutil.move(output, log_dir)
            return output_dict, job_str

    #START OF MAIN_F

    with open(args.datasets) as f:
        datasets = f.readlines()

    with open(args.tasks) as f:
        tasks = f.readlines()

    all_tasks = get_all_tasks()

    if args.run_optuna:
        base_cmd = 'run_optuna.py'
    else:
        base_cmd = 'train_loop.py'

    gcp_txt = "run_commands=(\n"

    for dataset in tqdm(datasets):
        dataset_path = "\"" + os.path.join(args.base_path, dataset.strip()) + '\"'
        #sanitize name
        # dataset_path = dataset_path.replace(r'(', r'\(').replace(r')', r'\)')
        # print("Dataset path:", dataset_path)
        log_dir = './logs/' + dataset.strip()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        for split in args.splits:
            for task in tasks:
                if 'tunetables' in task and args.gcp_run:
                    task_args = [
                        "--splits", str(split),
                        "--print_stdout",
                        "--verbose",
                    ]
                    if args.bptt > -1:
                        task_args = task_args + ["--bptt", str(args.bptt)]
                    if args.epochs > 0:
                        task_args = task_args + ["--epochs", str(args.epochs)]
                    if args.validation_period > 0:
                        task_args = task_args + ["--validation_period", str(args.validation_period)]
                    if args.wandb_log:
                        task_args = task_args + [
                            "--wandb_log",
                            "--wandb_project", args.wandb_project,
                            "--wandb_entity", args.wandb_entity,
                        ]
                    task_str = task + "_dataset_" + dataset.strip() + "_args_" + " ".join(task_args)
                    res = None
                elif 'tunetables' in task:
                    do_wandb = args.wandb_log
                    tt_args = copy.deepcopy(args)
                    tt_args.wandb_log = False
                    res, task_str = run_tunetables(dataset_path, task, split, log_dir, tt_args, base_cmd, gcp_txt, do_wandb)
                else:
                    res, task_str = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
                if args.gcp_run:
                    gcp_txt += "\"" + task_str + "\"" "\n"
                    if res:
                        print("Results for", dataset.strip(), "split", split, "task", task.strip(), ":", res)
    if args.gcp_run:
        task_str = dataset.strip() + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        gcp_txt += ")"
        with open("run_commands.sh", "w") as f:
            f.write(gcp_txt)
        start_time = time.time()
        print("Starting GCP run with gcp text:", gcp_txt)
        returncode, stdout, stderr = asyncio.run(run_command('bash batch/run_gcp_expt.sh'))
        print("GCP run finished in", time.time() - start_time, "seconds.")
        if args.print_stdout:
            print("Stdout:")
            print(stdout.decode())
            print("Stderr:")
            print(stderr.decode())
        target_dir = os.path.join(log_dir, task_str)
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "stdout.txt"), "w") as f:
            f.write(stdout.decode())
        with open(os.path.join(target_dir, "stderr.txt"), "w") as f:
            f.write(stderr.decode())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TabPFN')
    parser.add_argument('--base_path', type=str, default='/home/benfeuer/TabPFN-pt/tunetables/data', help='Path to TabPFN-pt dataset directory')
    parser.add_argument('--datasets', type=str, default='/home/benfeuer/TabPFN-pt/tunetables/metadata/subset.txt', help='Path to datasets text file')
    parser.add_argument('--tasks', type=str, default='/home/benfeuer/TabPFN-pt/tunetables/metadata/subset_tasks.txt', help='Tasks to run')
    parser.add_argument('--resume', type=str, default='/home/benfeuer/TabPFN-pt/tunetables/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt', help='TabPFN checkpoint to resume from')
    parser.add_argument('--bptt', type=int, default=-1, help='bptt batch size')
    parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')
    parser.add_argument('--shuffle_every_epoch', action='store_true', help='Whether to shuffle the order of the data every epoch (can help when bptt is large).')
    parser.add_argument('--run_optuna', action='store_true', help='Whether to run optuna hyperparameter search.')
    parser.add_argument('--real_data_qty', type=int, default=0, help='Number of real data points to use for fitting.')
    parser.add_argument('--gcp_run', action='store_true', help='Whether to launch the job on a GCP instance.')
    parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb.')
    parser.add_argument('--wandb_project', type=str, default='', help='Project name for wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='nyu-dice-lab', help='Entity for wandb logging.')
    parser.add_argument('--print_stdout', action='store_true', help='Whether to print stdout from each run.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output.')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to run.')
    parser.add_argument('--validation_period', type=int, default=0, help='Number of epochs between validation runs.')
    args = parser.parse_args()
    main_f(args)
