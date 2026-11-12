#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x-%j.out
# Usage (via run_all.sh, which adds --account and --partition from local.env):
#   sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --job-name=<name> --qos=<qos> --time=<hh:mm:ss> code/slurm_stage.sh <stage[,stage...]> <run_dir> [extra args]
set -euo pipefail
STAGES=$1; RUN_DIR=$2; shift 2
source ./local.env
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /dev/shm | tail -1

# Weights are copied once per stage into node RAM (/dev/shm); the four workers
# then memory-map the same pages instead of each streaming 58 GB from Lustre.
export LOCAL_MODEL_ROOT=/dev/shm/models-$SLURM_JOB_ID
declare -A REPO=([t2i]=Qwen/Qwen-Image-2512 [edit]=Qwen/Qwen-Image-Edit-2511 [video]=Wan-AI/Wan2.2-TI2V-5B-Diffusers)
stage_model() {
  local repo=$1 dst=$LOCAL_MODEL_ROOT/$1
  local snap=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$repo', local_files_only=True))")
  mkdir -p "$dst"
  (cd "$snap" && find . -type d -exec mkdir -p "$dst/{}" \; && find . -type l -o -type f | xargs -P 16 -I{} cp -L {} "$dst/{}")
  basename "$snap" > "$dst/.revision"
  echo "staged $repo ($(du -sh "$dst" | cut -f1)) at $(date -Is)"
}
trap 'rm -rf "$LOCAL_MODEL_ROOT"' EXIT

for STAGE in ${STAGES//,/ }; do
  echo "=== stage $STAGE start $(date -Is)"
  rm -rf "$LOCAL_MODEL_ROOT"; stage_model "${REPO[$STAGE]}"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i "$PY" code/02_generate.py --stage "$STAGE" --run-dir "$RUN_DIR" --shard $i --nshards 4 "$@" \
      > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${STAGE}-g$i.log" 2>&1 &
  done
  wait
  echo "=== stage $STAGE end $(date -Is)"
  tail -n 3 logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${STAGE}-g*.log
  CUDA_VISIBLE_DEVICES=0 "$PY" code/03_qc_score.py --run-dir "$RUN_DIR" --stage "$STAGE" --device cuda \
    > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${STAGE}-qc.log" 2>&1 || echo "QC failed for $STAGE"
done
