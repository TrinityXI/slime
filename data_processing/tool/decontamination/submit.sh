#!/bin/bash
#SBATCH --job-name=decontamination
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=64
#SBATCH --output=/work/projects/polyullm/slu/logs/stem/decontamination/%j-%x.out
#SBATCH --error=/work/projects/polyullm/slu/logs/stem/decontamination/%j-%x.err
#SBATCH --exclude=kb3-a1-nv-dgx[01-07]

set -x

container_image=/work/projects/polyullm/slu/containers/dedup-decon-251217.sqsh
container_name=dedup-decon-251217.sqsh
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm
work_dir=$(pwd)

config_script=$1
task_script=$2
echo "================ run task ========================"
echo "config_script: $config_script"
echo "task_script: $task_script"
echo "slurm_job_id: $SLURM_JOB_ID"
echo "=================================================="

SCRIPTS="
export SLURM_JOB_ID='$SLURM_JOB_ID'

cd '$work_dir' &&
source '$config_script' &&
bash '$task_script'
"

srun --container-name=$container_name \
     --container-mounts=$container_mounts \
     --container-image=$container_image \
     --container-workdir=$work_dir \
     --container-writable \
     bash -c "$SCRIPTS"