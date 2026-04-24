#!/bin/bash
set -x

DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
export INPUT_DIR="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage1/2025-12-15_13-44-27/result"
export OUTPUT_DIR="/work/projects/polyullm/slu/data/data-pipeline-20251209/stem/Mixture-of-Thoughts-dedup/query-level/stage2/${DATETIME}"
export EMBEDDING_DIR="" # 如果没有算过数据的embedding vector则置空，脚本会计算embedding vector并存储在output_dir中
export EMBEDDING_MODEL_PATH="/work/projects/polyullm/slu/models/bge-m3"
export THRESHOLD=0.9
# 每张GPU encode embedding vector的batch size。经测试，使用bge-m3作为embedding model，单张H800可支持的batch size一般为128或256，再大可能会oom (视query长度而定)
export EMBEDDING_BATCH_SIZE=128
# 每张GPU进行相似度搜索的batch size
export SEARCH_BATCH_SIZE=100
export QUERY_KEY="query"