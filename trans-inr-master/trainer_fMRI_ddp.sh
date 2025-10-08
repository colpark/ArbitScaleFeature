#!/bin/bash
#SBATCH -A m4727
#SBATCH -N 1
#SBATCH -C gpu # gpu&hbm80g
#SBATCH -D /pscratch/sd/t/tbalasoo/trans-inr # change here
#SBATCH -q regular # preempt #regular #shared #regular, shared,  ...
#SBATCH --job-name=DDP_example
#SBATCH --output=./logs/250810_%x_%j.out
#SBATCH --error=./logs/250810_%x_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:10:00 #02:00:00 #for single epoch training 

# to test inetactively, run the command below
# salloc --nodes 1 --time 00:30:00 -C gpu --account m4727 -q interactive --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32

root_path="$PSCRATCH/MAMBAINR/trans-inr-master"  # change here
cd $root_path

module load conda
conda activate /global/cfs/cdirs/m4727/mamba-env

run() {
    config="$1"
    fmri_cfg="$2"
    
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    export MASTER_PORT=${MASTER_PORT:-29500} # Use defined port or default
    export OMP_NUM_THREADS=1
    echo "--- Rank ${RANK} | LocalRank ${LOCAL_RANK} | Master ${MASTER_ADDR}:${MASTER_PORT} | WorldSize ${WORLD_SIZE} | Nodes ${SLURM_NNODES} | GPUs-Node ${SLURM_GPUS_ON_NODE} | CUDA_VISIBLE ${CUDA_VISIBLE_DEVICES} ---"

    # python simple_ddp.py 
    TORCH_DISTRIBUTED_DEBUG=DETAIL python run_trainer_slurm.py --cfg "$PSCRATCH/MAMBAINR/trans-inr-master/cfgs/$config" --fmri-data-cfg "$PSCRATCH/MAMBAINR/trans-inr-master/cfgs/$fmri_cfg"

}
export -f run

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(srun --nodelist="$MASTER_NODE" --ntasks=1 --nodes=1 bash -c "ip -4 addr show hsn0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'| head -n 1 " 2>/dev/null)

export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=29500 # Ensure this matches the port in the function

echo "** MASTER_ADDR = ${MASTER_ADDR}"
echo "** MASTER_PORT = ${MASTER_PORT}"

# Calculate total ranks
ranks_per_node=$SLURM_GPUS_ON_NODE # Should match --ntasks-per-node
ranks_total=$(( ranks_per_node * SLURM_NNODES ))

echo "** NUM_NODES      = $SLURM_NNODES"
echo "** Ranks per node = ${ranks_per_node}"
echo "** Total ranks    = ${ranks_total}"

# -o0 option lets you print stdout/stderr from only rank 0 
# to help you avoid making too lengty output
srun -o0 -u \
    -c 31 \
    -n $SLURM_NTASKS \
    -N $SLURM_NNODES \
    --gpus-per-node=$ranks_per_node \
    bash -c "run $1 $2"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo elapsed_time: $elapsed_time