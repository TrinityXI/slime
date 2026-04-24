#!/bin/bash
set -x

DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
export INPUT_PATH="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage2/2025-12-16_21-17-28/result"
export OUTPUT_DIR="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-decon/${DATETIME}"
export BENCHMARK_CONFIG_PATH="./benchmarks.yaml"
export QUERY_KEY="query"
export N_GRAM_SIZE=32
export NUM_PROC=1