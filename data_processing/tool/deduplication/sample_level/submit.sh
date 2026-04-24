#!/bin/bash
#SBATCH --job-name=spark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1200GB
#SBATCH --cpus-per-task=144
#SBATCH --output=/work/projects/polyullm/slu/logs/stem/dedup/sample-level/%j-%x.out
#SBATCH --error=/work/projects/polyullm/slu/logs/stem/dedup/sample-level/%j-%x.err
#SBATCH --exclude=kb3-a1-nv-dgx[01-07]

set -x

# ========================================================
# replace these information with your own
# ========================================================
workdir=$(pwd)
tmp_dir=/lustre/projects/polyullm/slu/tmp/spark-${SLURM_JOB_ID}
cache_dir=/lustre/projects/polyullm/slu/tmp/chukonu_cache
container_image=/lustre/projects/polyullm/container/chukonu+3.4.1-jdk11-2026010601.sqsh
container_name=chukonu+3.4.1-jdk11-2026010601
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm,$cache_dir:/opt/chukonu_cache
# ========================================================

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Spark Master
port=7077
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# create tmp folder
mkdir -p ${tmp_dir}
chmod -R 777 ${tmp_dir}
mkdir -p ${cache_dir}

printenv

echo "Starting Spark Master at $head_node"
mkdir -p ${tmp_dir}/${head_node}
chmod -R 777 ${tmp_dir}/${head_node}
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts,${tmp_dir}/${head_node}:/tmp,${tmp_dir}/${head_node}:/opt/spark/logs \
    --container-image=$container_image \
    --container-writable \
    bash -c "bash /opt/spark/sbin/start-master.sh -h $head_node_ip  --webui-port 8031  && tail -f /dev/null" &

sleep 5

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "create tmp folder"
    mkdir -p ${tmp_dir}/${node_i}
    chmod -R 777 ${tmp_dir}/${node_i}
    echo "Starting Spark Worker $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --container-name=$container_name \
        --container-mounts=$container_mounts,${tmp_dir}/${head_node}:/tmp,${tmp_dir}/${head_node}:/opt/spark/logs \
        --container-image=$container_image \
        --container-writable \
            bash -c "bash /opt/spark/sbin/start-worker.sh spark://$head_node_ip:$port  && tail -f /dev/null" &
        sleep 5
done

config_script=$1
task_script=$2

master_url=spark://$head_node_ip:$port
executor_cores=$(( $SLURM_CPUS_PER_TASK ))
executor_memory=$(( $SLURM_MEM_PER_NODE / 1024 ))G
default_parallelism=$(( $SLURM_CPUS_PER_TASK * $worker_num ))

echo "================ run task ========================"
echo "config_script: $config_script"
echo "task_script: $task_script"
echo "master_url: $master_url"
echo "executor_cores: $executor_cores"
echo "executor_memory: $executor_memory"
echo "default_parallelism: $default_parallelism"
echo "slurm_job_id: $SLURM_JOB_ID"
echo "=================================================="

echo "sleep 60 seconds"
sleep 60


SCRIPTS="
export MASTER_URL='$master_url'
export EXECUTOR_CORES='$executor_cores'
export EXECUTOR_MEMORY='$executor_memory'
export DEFAULT_PARALLELISM='$default_parallelism'
export SLURM_JOB_ID='$SLURM_JOB_ID'
export LOG_PATH='$tmp_dir'

cd '$workdir' &&
source '$config_script' &&
bash '$task_script'"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    bash -c "$SCRIPTS"

# Clean up Spark processes
cleanup() {
    echo "Shutting down Spark cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        --container-name=$container_name \
        --container-image=$container_image \
        --container-writable \
        bash -c "bash /opt/spark/sbin/stop-master.sh"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
            --container-name=$container_name \
            --container-image=$container_image \
            --container-writable \
            bash -c "bash /opt/spark/sbin/stop-worker.sh"
    done
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT
