#!/bin/bash
#SBATCH --job-name=smatable_grid
#SBATCH --partition=GPU-big
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# 0) Basics and context

cd "$SLURM_SUBMIT_DIR"

echo "JobID: ${SLURM_JOB_ID}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Node list: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"

# 1) Configuration for container script

IMAGE="${SLURM_SUBMIT_DIR}/run_container.sif"
PY_SCRIPT="run_smatable_nas.py"

STUDY_NAME="smatable-explorer-test"
N_WORKERS=12
DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
MAX_TRIALS=999999
TOP_N_MODELS=5

DATASET_SRC="${HPC_HOME}/data/smatable-data-recording/official-recording-labeled"

WS_NAME="smatable-ws"
WS_DAYS=30

# 2) Prepare SCRATCH workspace

if ws_find "${WS_NAME}" >/dev/null 2>&1; then
  WS_PATH="$(ws_find "${WS_NAME}" | tail -n 1)"
  echo "Workspace already exists: ${WS_NAME} -> ${WS_PATH}"
else
  echo "Workspace not found -> creating: ws_allocate ${WS_NAME} ${WS_DAYS}"
  ws_allocate "${WS_NAME}" "${WS_DAYS}"
  WS_PATH="$(ws_find "${WS_NAME}" | tail -n 1)"
  echo "Created workspace: ${WS_NAME} -> ${WS_PATH}"
fi

JOB_SCRATCH="${WS_PATH}/jobs/${SLURM_JOB_ID}"
DATA_CACHE_BASE="${WS_PATH}/datasets"
DATA_DIR="${DATA_CACHE_BASE}/official-recording-labeled"

mkdir -p "${JOB_SCRATCH}" "${DATA_CACHE_BASE}"

# 3) study state dir, so journals and checkpoints persist across jobs
STUDY_STATE_DIR="${WS_PATH}/studies/${STUDY_NAME}"
mkdir -p "${STUDY_STATE_DIR}"

# 4) Copy Dataset to workspace (if not there yet)

if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null || true)" ]; then
  echo "Dataset not found in workspace -> copy dataset to workspace."
  mkdir -p "${DATA_DIR}"
  rsync -a "${DATASET_SRC}/" "${DATA_DIR}/"
else
  echo "Dataset already exists -> using ${DATA_DIR}"
fi

# 5) Run apptainer

echo "Starting Container..."
echo "IMAGE: ${IMAGE}"
echo "PY_SCRIPT: ${PY_SCRIPT}"
echo "DATA_DIR (host): ${DATA_DIR}"
echo "Workspace:  ${WS_PATH}"

srun --ntasks=1 apptainer run --nv --writable-tmpfs \
  --bind "${DATA_DIR}:/data" \
  --bind "${STUDY_STATE_DIR}:/study" \
  --env STUDY_NAME="${STUDY_NAME}" \
  --env JOURNAL_FILE="/study/journal.log" \
  --env SAMPLER_CHECKPOINT_DIR="/study/checkpoints" \
  --env N_WORKERS="${N_WORKERS}" \
  --env DEVICES="${DEVICES}" \
  --env MAX_TRIALS="${MAX_TRIALS}" \
  --env TOP_N_MODELS="${TOP_N_MODELS}" \
  --env DATA_DIR="/data" \
  "${IMAGE}" \
  "${PY_SCRIPT}"

echo "Container run finished."