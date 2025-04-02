#!/bin/bash
set -ex
resource_config_path="/opt/ml/input/config/resourceconfig.json"
if [[ ! -f "$resource_config_path" ]]; then
    echo "Error: Resource config file not found at $resource_config_path"
    exit 1
fi
cat "$resource_config_path"
current_host=$(jq -r '.current_host' "$resource_config_path")
hosts=$(jq '.hosts | length' "$resource_config_path")
echo "Current host: $current_host"
echo "Total Hosts: $hosts"
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

cd /opt/ml/input/data/code





# Environment Setup
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py_312_torch_transformers

# Entrypoint



exec torchrun --nnodes="$nnodes" \
--nproc_per_node="$nproc_per_node" \
--rdzv_id=4224 \
--rdzv_backend=c10d \
--max_restarts=3 \
--rdzv_endpoint=algo-1:29500 \
--node_rank="$rank_id" \
main.py --epochs=10 --batch-size=256
