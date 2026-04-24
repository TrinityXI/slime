#!/bin/bash
set -x

INPUT_DIR=${INPUT_DIR:-"/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage1/2025-12-15_13-44-27/result"}
OUTPUT_DIR=${OUTPUT_DIR:-"/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage2/tmp"}
EMBEDDING_DIR=${EMBEDDING_DIR:-""}
EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-"/work/projects/polyullm/slu/models/bge-m3"}
THRESHOLD=${THRESHOLD:-"0.9"}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-"256"}
SEARCH_BATCH_SIZE=${SEARCH_BATCH_SIZE:-"100"}
QUERY_KEY=${QUERY_KEY:-"query"}

python dedup.py \
    --threshold ${THRESHOLD} \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --embedding_dir "${EMBEDDING_DIR}" \
    --embedding_model_path "${EMBEDDING_MODEL_PATH}" \
    --embedding_batch_size ${EMBEDDING_BATCH_SIZE} \
    --search_batch_size ${SEARCH_BATCH_SIZE} \
    --query_key ${QUERY_KEY}

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Job completed."
else
    echo "Job executed error, exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi