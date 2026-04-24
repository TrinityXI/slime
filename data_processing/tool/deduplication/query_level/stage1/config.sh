#!/bin/bash
set -x

DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')

# dataset parameters
export INPUT_PATH="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/sample-level/2025-12-10_17-32-15/result/*.parquet"
export OUTPUT_PATH="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage1/${DATETIME}"
export FILE_TYPE="parquet"
export QUERY_KEY="query"

# dedup parameters
export THRESHOLD="0.9"
export NGRAM_SIZE="5"
export MIN_LENGTH="2"
export NUM_PERM="128"
export B="8"
export R="16"
export WITH_SPLIT="False"
export RUN_CHUKONU="True"