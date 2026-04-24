#!/bin/bash
set -x

DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')

# dataset parameters
export INPUT_PATH="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts/*/*.parquet"
export OUTPUT_PATH="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/${DATETIME}"
export FILE_TYPE="parquet"
export TEXT_KEY="conversations"

# dedup parameters
export THRESHOLD="0.85"
export NGRAM_SIZE="5"
export MIN_LENGTH="2"
export NUM_PERM="128"
export B="8"
export R="16"
export WITH_SPLIT="False"
export RUN_CHUKONU="True"