# HPC Apptainer Example

This directory contains an example setup for running `elastic-ai.explorer` on an
HPC system with SLURM, Apptainer, and NVIDIA GPUs.

The files are not a generic HPC interface or a reusable deployment tool. They
document one concrete setup and are intended as a starting point for adapting the
explorer to a local HPC environment.

## Files

- `run_container.def` is an example Apptainer definition for packaging the
  project and one experiment entry point into a container image.
- `run_parallized_search.sh` is an example SLURM job script. It uses local HPC
  conventions such as workspaces, dataset locations, and GPU allocation settings.


## Build the Container

Build the image from the project root:

```bash
apptainer build hpc/run_container.sif hpc/run_container.def
```

The SLURM script expects the image at:

```text
hpc/run_container.sif
```

If the image is stored somewhere else, update the `IMAGE` variable in
`run_parallized_search.sh`.

## Run the Search Job

```bash
sbatch run_parallized_search.sh
```

The job script creates or reuses the configured workspace, copies the dataset
there if needed, keeps the study state in that workspace, and starts the
container via `srun apptainer run`.

## Container Bind Mounts

The job script binds two host directories into the container:

- `${DATA_DIR}:/data` provides the training and test data inside the container.
- `${STUDY_STATE_DIR}:/study` stores the Optuna journal and sampler checkpoints.

Keeping `/study` outside the container makes interrupted or timed-out searches
resumable across later SLURM jobs.

## Environment Variables Passed to the Container

The SLURM script passes these values into the container:

- `STUDY_NAME` names the Optuna study.
- `JOURNAL_FILE` points to the Optuna journal file, usually
  `/study/journal.log`.
- `SAMPLER_CHECKPOINT_DIR` points to the sampler checkpoint directory, usually
  `/study/checkpoints`.
- `N_WORKERS` sets the number of parallel worker processes.
- `DEVICES` lists the devices assigned to workers, for example
  `cuda:0,cuda:1,cuda:2,cuda:3`.
- `MAX_TRIALS` sets the total trial limit for the search.
- `TOP_N_MODELS` controls how many top models are collected after the search.
- `DATA_DIR` is set to `/data` inside the container.

Adjust these values in `run_parallized_search.sh` before submitting the job.
