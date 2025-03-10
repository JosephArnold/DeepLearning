#!/bin/bash -x
#SBATCH --output=/p/project1/sdlrs/arnold6/fwdgrad/slurm_outputs/fwdgrad-out_gpu.%j
#SBATCH --error=/p/project1/sdlrs/arnold6/fwdgrad/slurm_outputs/fwdgrad-err_gpu.%j
#SBATCH --account=####
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:60:00
#SBATCH --partition=####


ml  Stages/2024  GCCcore/.12.3.0
ml Hydra/1.3.2
ml PyTorch/2.1.2
ml torchvision
ml tensorboard
ml matplotlib/3.7.2

export UCX_RC_MLX5_FAILURE=INFO
export UCX_RC_MLX5_FC_ENABLE=y
export UCX_RC_MLX5_TIMEOUT=10000000.00us
export UCX_RC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_DC_MLX5_FAILURE=INFO
export UCX_DC_MLX5_FC_ENABLE=y
export UCX_DC_MLX5_TIMEOUT=10000000.00us
export UCX_DC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_UD_MLX5_FAILURE=INFO
export UCX_UD_MLX5_FC_ENABLE=y
export UCX_UD_MLX5_TIMEOUT=10000000.00us
export UCX_UD_MLX5_RNR_TIMEOUT=10000.00us
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=10s
export NCCL_IB_RETRY_CNT=10

RDV_ADDR=$(hostname)
FQDN=$(hostname --fqdn)  # <--- Seems to provide the same result as `hostname` without parameters
echo "FQDN: ""$FQDN"
WORLD_SIZE=$SLURM_JOB_NUM_NODES
echo "Number of nodes is ${SLURM_JOB_NUM_NODES}"
# # ----- "Without this, srun does not inherit cpus-per-task from sbatch" -----
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
echo "SRUN_CPUS_PER_TASK: ""$SRUN_CPUS_PER_TASK"
MASTER_ADDRi="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# # Allow communication over InfiniBand cells.
MASTER_ADDRi="${MASTER_ADDRi}i"
# # Get IP for hostname.
export MASTER_ADDRi="$(nslookup "$MASTER_ADDRi" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=33523

# ----- ATTENTION -- REVIEW -----
MASTER_ADDR2="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR_NAME=$MASTER_ADDR2
export MASTER_ADDR2="$(nslookup "$MASTER_ADDR2" | grep -oP '(?<=Address: ).*')"
# -----

echo "Hostname: ""$RDV_ADDR"
echo "MASTER_ADDR[i]: ""$MASTER_ADDRi"
echo "MASTER_ADDR2: ""$MASTER_ADDR2"
echo "MASTER_ADDR_NAME: ""$MASTER_ADDR_NAME"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
RESUME_DIR="/p/project1/geofm4eo/pretrained_code/Prithvi-global-v1-experimental-fsdp/submit_scripts/trained_models/${DATETIME}_JOSEPH_TEST_ViTL"
echo "RESUME_DIR: ""$RESUME_DIR"
mkdir $RESUME_DIR


#srun --nodes=1  --gpus-per-task=1 python fwdgrad-nn.py
srun -l bash -c "torchrun \
    --nproc_per_node=1\
    --nnodes=$SLURM_JOB_NUM_NODES\\
    --rdzv_id=$SLURM_JOB_ID\
    --rdzv_backend=c10d\
    --rdzv_endpoint=$MASTER_ADDR2:$MASTER_PORT \
    --rdzv_conf=is_host=\"\$(if((SLURM_PROCID)); then echo 0; else echo 1; fi)\" \
    --max-restarts=3 \
    fwdgrad-nn-ddp.py"
