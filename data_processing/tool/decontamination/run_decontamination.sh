#!/bin/bash
set -x

INPUT_PATH=${INPUT_PATH:-"/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage2/2025-12-16_21-17-28/result"}
OUTPUT_DIR=${OUTPUT_DIR:-"/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-decon/tmp"}
BENCHMARK_CONFIG_PATH=${BENCHMARK_CONFIG_PATH:-"./benchmarks.yaml"}
N_GRAM_SIZE=${N_GRAM_SIZE:-"32"}
QUERY_KEY=${QUERY_KEY:-"query"}
NUM_PROC=${NUM_PROC:-"1"}

python decontamination.py \
    --benchmark_config_path "${BENCHMARK_CONFIG_PATH}" \
    --ngram_size ${N_GRAM_SIZE} \
    --input_path "${INPUT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --query_key ${QUERY_KEY} \
    --num_proc ${NUM_PROC}

echo "Decontamination job completed."