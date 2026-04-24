#!/bin/bash

# ====================================================================
# Resource Environment variables
# ===============================================================
# Ensure MASTER_URL is not empty
if [ -z "${MASTER_URL}" ]; then
    echo "Error: MASTER_URL is required but not set"
    exit 1
fi

set -x

# Read from environment variables, use defaults if not set
EXECUTOR_CORES="${EXECUTOR_CORES:-4}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-32G}"
EXECUTOR_MEMORY_OVERHEAD="${EXECUTOR_MEMORY_OVERHEAD:-4G}"
DEFAULT_PARALLELISM="${DEFAULT_PARALLELISM:-16}"
# MASTER_URL="${MASTER_URL:-spark://kb3-a1-nv-dgx08:7077}"
LOG_PATH=${LOG_PATH:-/work/projects/polyullm/wtf/spark/logs/slurm}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_PATH}/spark-runtime-${TIMESTAMP}.log"
ERR_FILE="${LOG_PATH}/spark-runtime-${TIMESTAMP}.err"

# ====================================================================
# Application parameters
# ===============================================================
# Calculate SQL_SHUFFLE_PARTITIONS based on DEFAULT_PARALLELISM
SQL_SHUFFLE_PARTITIONS=$(awk "BEGIN {print int(${DEFAULT_PARALLELISM} * 1.5)}")

# Application parameters
INPUT_PATH=${INPUT_PATH:-"/work/projects/polyullm/congkai/pretrain_data/Nemotron-CC-Math-v1/*/*.parquet"}
OUTPUT_PATH=${OUTPUT_PATH:-"/work/projects/polyullm/wtf/data/Nemotraon-CC-Math-v1-dedup"}
FILE_TYPE=${FILE_TYPE:-"parquet"}
QUERY_KEY=${QUERY_KEY:-"query"}

# Additional parameters (using default values from the script)
THRESHOLD=${THRESHOLD:-"0.85"}
NGRAM_SIZE=${NGRAM_SIZE:-"5"}
MIN_LENGTH=${MIN_LENGTH:-"2"}
NUM_PERM=${NUM_PERM:-"128"}
B=${B:-"8"}
R=${R:-"16"}
WITH_SPLIT=${WITH_SPLIT:-"False"}
RUN_CHUKONU=${RUN_CHUKONU:-"True"}

# set +x

# Build spark-submit command with redirection
spark-submit \
    --master ${MASTER_URL} \
    --executor-cores ${EXECUTOR_CORES} \
    --executor-memory ${EXECUTOR_MEMORY} \
    --conf spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD} \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --conf spark.sql.adaptive.skewJoin.enabled=true \
    --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=256m \
    --conf spark.sql.shuffle.partitions=${SQL_SHUFFLE_PARTITIONS} \
    --conf spark.default.parallelism=${DEFAULT_PARALLELISM} \
    --conf spark.plugins=org.pacman.chukonu.ChukonuPlugin \
    --conf spark.chukonu.enableNativeCodegen=true \
    --conf spark.chukonu.root=/opt/chukonu_install \
    --conf spark.chukonu.cxx=/usr/bin/g++ \
    --conf spark.chukonu.stagingdir=/opt/chukonu_staging \
    --conf spark.chukonu.compileCacheDir=/opt/chukonu_cache \
    --conf spark.chukonu.buildType=Release \
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
    --conf spark.sql.pyspark.jvmStacktrace.enabled=true \
    --conf spark.kryo.unsafe=true \
    --conf spark.shuffle.service.enabled=false \
    --conf spark.memory.offHeap.enabled=true \
    --conf spark.memory.offHeap.size=1g \
    --conf spark.local.dir="/lustre/projects/polyullm/slu/tmp" \
    dedup.py \
    --query_key ${QUERY_KEY} \
    --num-parallel ${DEFAULT_PARALLELISM} \
    --threshold ${THRESHOLD} \
    --ngram_size ${NGRAM_SIZE} \
    --min_length ${MIN_LENGTH} \
    --num_perm ${NUM_PERM} \
    --b ${B} \
    --r ${R} \
    --with_split ${WITH_SPLIT} \
    --run_chukonu ${RUN_CHUKONU} \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --file_type ${FILE_TYPE} \
    > "${LOG_FILE}" 2> "${ERR_FILE}"

echo "Job completed. Logs saved to:"
echo "  Standard output: ${LOG_FILE}"
echo "  Standard error:  ${ERR_FILE}"
