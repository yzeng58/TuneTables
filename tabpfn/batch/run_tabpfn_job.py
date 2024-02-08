import subprocess
import asyncio
import os
import time
import argparse
from tqdm.auto import tqdm
from all_tasks import get_all_tasks
import shutil
from pathlib import Path

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

    with open(args.datasets) as f:
        datasets = f.readlines()

    with open(args.tasks) as f:
        tasks = f.readlines()

    all_tasks = get_all_tasks()

    if args.run_optuna:
        base_cmd = 'run_optuna_n.py'
    else:
        base_cmd = 'train_loop.py'

    gcp_txt = "run_commands=(\n"

    for dataset in tqdm(datasets):
        print("Starting dataset: ", dataset.strip())
        dataset_path = os.path.join(args.base_path, dataset.strip())
        log_dir = './logs/' + dataset.strip()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        for split in args.splits:
            for task in tasks:
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
                if task.startswith('zs'):
                    ensemble_size = int(task.split('-')[-1])
                    subset_ft_method = task.split('-')[-2]
                    command = ['python', base_cmd, 
                            '--data_path', dataset_path,
                            '--subset_features_method', subset_ft_method,
                            '--split', str(split),
                            '--wandb_log',
                            '--wandb_group', dataset.strip() + "_" + task_str + "_" + subset_ft_method, 
                            '--real_data_qty', str(args.real_data_qty),
                            '--zs-eval-ensemble', str(ensemble_size)]
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
                    if args.wandb_project != '':
                        next_task['wandb_project'] = args.wandb_project
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
                    if args.gcp_run:
                        #dataset_path = dataset_path.replace(r'(', r'\(').replace(r')', r'\)')
                        command = ['python', base_cmd, '--data_path \"' + dataset_path + "\"", '--split', str(split), '--real_data_qty', str(args.real_data_qty), '--wandb_group', dataset.strip().replace("(", "").replace(")", "") + "_" + task_str] + addl_args
                    else:
                        command = ['python', base_cmd, '--data_path', dataset_path, '--split', str(split), '--real_data_qty', str(args.real_data_qty), '--wandb_group', dataset.strip() + "_" + task_str] + addl_args
                    if args.run_optuna:
                        command = command + ["--wandb_project", "tabpfn-pt-optuna"]
                if args.bptt > -1:
                    command.append("--bptt")
                    command.append(str(args.bptt))     
                if args.shuffle_every_epoch:
                    command.append("--shuffle_every_epoch")           
                print("Running command:", ' '.join(command))
                if args.gcp_run:
                    gcp_txt += "\'" + ' '.join(command) + '\'\n'

                    # Check if there are already 10 jobs running
                    # current_op_count = int(subprocess.check_output("gcloud compute instances list --filter='status=RUNNING' | wc -l", shell=True))
                    # while current_op_count > 10:
                    #     print("Waiting for a slot to open up...")
                    #     time.sleep(30)
                    #     # await(running_tasks[-1])
                    #     current_op_count = int(subprocess.check_output("gcloud compute instances list --filter='status=RUNNING' | wc -l", shell=True))
                    # with open("run_command.txt", "w") as f:
                    #     f.write(' '.join(command))
                    # 
                    # # running_tasks.append(t)
                    # # time.sleep(30)
                    # if returncode != 0:
                    #     print("GCP launch failed.")
                    #     print("Stdout:")
                    #     print(stdout.decode())
                    #     print("Stderr:")
                    #     print(stderr.decode())
                    #     exit()
                    # else:

                else:
                    subprocess.call(command)
                    new_outputs = Path('logs').glob('_multiclass*')
                    updated_outputs = []
                    for output in new_outputs:
                        new_name = output.name.replace('_multiclass', task_str)
                        new_path = os.path.join(output.parent, new_name)
                        os.rename(output, new_path)
                        updated_outputs.append(new_path)
                    for output in updated_outputs:
                        shutil.move(output, log_dir)
    if args.gcp_run:
        gcp_txt += ")"
        with open("run_commands.sh", "w") as f:
            f.write(gcp_txt)
        start_time = time.time()
        print("Starting GCP run.")
        returncode, stdout, stderr = asyncio.run(run_command('bash batch/run_gcp_expt.sh'))
        print("GCP run finished in", time.time() - start_time, "seconds.")
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
    parser.add_argument('--base_path', type=str, default='/home/yyyyyyyy/TabPFN-pt/tabpfn/data', help='Path to TabPFN-pt dataset directory')
    parser.add_argument('--datasets', type=str, default='/home/yyyyyyyy/TabPFN-pt/tabpfn/metadata/subset.txt', help='Path to datasets text file')
    parser.add_argument('--tasks', type=str, default='/home/yyyyyyyy/TabPFN-pt/tabpfn/metadata/subset_tasks.txt', help='Tasks to run')
    parser.add_argument('--bptt', type=int, default=-1, help='bptt batch size')
    parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')
    parser.add_argument('--shuffle_every_epoch', action='store_true', help='Whether to shuffle the order of the data every epoch (can help when bptt is large).')
    parser.add_argument('--run_optuna', action='store_true', help='Whether to run optuna hyperparameter search.')
    parser.add_argument('--real_data_qty', type=int, default=0, help='Number of real data points to use for fitting.')
    parser.add_argument('--gcp_run', action='store_true', help='Whether to launch the job on a GCP instance.')
    parser.add_argument('--wandb_project', type=str, default='', help='Project name for wandb logging')

    args = parser.parse_args()
    main_f(args)
