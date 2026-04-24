#!/bin/bash
#SBATCH --job-name=dedup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1200GB
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --output=/work/projects/polyullm/slu/logs/stem/dedup/query-level/stage2/%j-%x.out
#SBATCH --error=/work/projects/polyullm/slu/logs/stem/dedup/query-level/stage2/%j-%x.err
#SBATCH --exclude=kb3-a1-nv-dgx[01-07]
# 注意，非8卡机器只能申请01-07开发节点，如果是cpu任务则不受限制，可在脚本中删除--nodelist这一行
# 如果申请8卡，建议把--nodelist=kb3-a1-nv-dgx[01-07]改成--exclude=kb3-a1-nv-dgx[01-07]，申请资源时排除开发节点，避免和debug任务争抢资源
set -x

container_image=/lustre/projects/polyullm/posttrain/container/dedup-decon-251217.sqsh
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