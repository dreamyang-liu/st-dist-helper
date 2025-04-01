#!/bin/bash

# This script is used to set up the environment for distributed training in SageMaker Training Job.
# It extracts the necessary information from the resource config file and sets the appropriate environment variables.

set -e
resource_config_path="/opt/ml/input/config/resourceconfig.json"

if [[ ! -f "$resource_config_path" ]]; then
    echo "Error: Resource config file not found at $resource_config_path"
    exit 1
fi

cat "$resource_config_path"

# Use jq to extract all needed information in one call
# Extract current_host, hosts array, and gpu_count from the resource config file
current_host=$(jq -r '.current_host' "$resource_config_path")
hosts=$(jq '.hosts | length' "$resource_config_path")

# Print extracted values for verification
echo "Current host: $current_host"
echo "Hosts: $hosts"

# Extract the host number from current_host and calculate rank_id
if [[ $current_host =~ algo-([0-9]+) ]]; then
    host_number="${BASH_REMATCH[1]}"
    rank_id=$((host_number - 1))
else
    echo "Error: Unable to extract host number from current_host: $current_host"
    exit 1
fi
nnodes=$hosts
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

echo "nnodes: $nnodes"
echo "nproc_per_node: $nproc_per_node"
echo "rank_id: $rank_id"

exec torchrun --nnodes="$nnodes" \
         --nproc_per_node="$nproc_per_node" \
         --rdzv_id=resnet50 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=algo-1:29500 \
         --node_rank="$rank_id" \
         "$@"
