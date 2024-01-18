#!/bin/bash

#defines run_experiment_gpu
source ./batch/utils.sh

set -e

# Read the contents of 'run_command'
file_contents=$(cat run_command.txt)

echo "Running command: $file_contents"

trap "delete_instances" EXIT
INSTANCE_LIST=()
instance_name="tabpfn-pt-$(date +%Y%m%d%H%M%S)"
# add instance name to the instance list
INSTANCE_LIST+=("${instance_name}")
# Run the command
run_experiment_gpu "$file_contents" "$instance_name" >> "runlog-$(date +%Y%m%d%H%M%S).txt" 2>&1 &

wait