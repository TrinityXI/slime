# InfiAlign Data Pipeline

Post-train 数据处理流水线，包含 **去重 (Deduplication)** 和 **去污染 (Decontamination)** 模块。

## 目录结构

```
infialign/
├── deduplication/
│   ├── sample_level/          # Sample 级去重 (query+response)
│   │   ├── dedup.py
│   │   ├── config.sh
│   │   ├── run_dedup.sh
│   │   └── submit.sh
│   └── query_level/
│       ├── stage1/            # Query 级去重 — MinHash/LSH/WCC (Spark)
│       │   ├── dedup.py
│       │   ├── config.sh
│       │   ├── run_dedup_stage1.sh
│       │   └── submit.sh
│       └── stage2/            # Query 级去重 — 语义相似度 (FAISS)
│           ├── dedup.py
│           ├── config.sh
│           ├── run_dedup_stage2.sh
│           └── submit.sh
└── decontamination/           # 评测集去污染
    ├── decontamination.py
    ├── benchmarks.yaml
    ├── config.sh
    ├── run_decontamination.sh
    └── submit.sh
```

## 整体流程

```
原始数据 (Parquet/JSON)
  │
  ▼
Sample-Level 去重 — Exact + MinHash/LSH/WCC (Spark)
  │
  ▼
Query-Level Stage1 — Query 聚合 + Exact + MinHash/LSH/WCC (Spark)
  │
  ▼
Query-Level Stage2 — Embedding 语义去重 (FAISS, GPU)
  │
  ▼
Decontamination — N-gram 匹配去除评测集污染
  │
  ▼
清洗后数据 (Parquet)
```

---

## Deduplication

### Sample-Level 去重

对完整的 query+response 文本进行去重。

#### 算法

1. **Exact Matching** — 去除文本内容完全相同的 sample
2. **MinHash + LSH + WCC 相似去重**:
   - **MinHash + LSH**: 将文本转为 n-gram，计算 MinHash 签名，通过 LSH 分桶找到直接相似的 sample 对
   - **WCC (Weakly Connected Component)**: 基于弱连通分量找到间接相似的 sample（如 a\~b, b\~c → a\~c）

文本预处理：NFKC Unicode 归一化 → 小写 → 标点去除/分割 → n-gram 分词。

#### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_path` | (必填) | 输入文件/目录路径 |
| `--output_path` | (必填) | 输出目录根路径 |
| `--file_type` | `parquet` | 输入格式，`parquet` 或 `json` |
| `--text_key` | `text` | 去重文本列名 |
| `--threshold` | `0.85` | Jaccard 相似度阈值 |
| `--ngram_size` | `5` | n-gram 大小 |
| `--num_perm` | `128` | MinHash 置换数（签名维度） |
| `--b` | `8` | LSH band 数量（可选，None 时自动计算最优值） |
| `--r` | `16` | 每个 band 的行数（可选，None 时自动计算最优值） |
| `--min_length` | `2` | 最短文档长度 |
| `--with_split` | `False` | 是否启用 CJK 字符分割 |
| `--run_chukonu` | `True` | 使用 Chukonu 原生加速 |
| `--num-parallel` | `3200` | WCC 计算并行度 |

#### 输出

```
{output_path}/
├── dedup/    # Exact 去重后的中间数据
├── wcc/      # WCC 连通分量结果 (vid, component)
├── dup/      # 被标记为重复的 sample
└── result/   # 最终去重结果
```

#### 提交任务

```bash
cd deduplication/sample_level
sbatch submit.sh config.sh run_dedup.sh
```

> **注意**：启动 Spark 集群至少需要 2 个节点（1 管理 + 1 worker）。默认申请 2 节点，可按需增加（如 4, 8, 16）。

---

### Query-Level 去重

只对 query 进行去重，同时保留每个 query 的所有 response，分为两个阶段。

#### Stage 1 — MinHash/LSH/WCC (Spark)

##### 算法

1. **Query Aggregation** — 将相同 query 的所有记录聚合到 `sample_list` 字段（`[record1, record2, ...]`），保留全部 response 以便后续质量校验
2. **Exact Matching** — 去除完全相同的 query
3. **MinHash + LSH + WCC** — 相似 query 去重（同 sample-level）

##### 参数

与 sample-level 基本一致，区别：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--query_key` | `query` | 去重的 query 列名（替代 `--text_key`） |

> Stage 1 目前仅支持 Chukonu 加速模式（`--run_chukonu=True`）。

##### 提交任务

```bash
cd deduplication/query_level/stage1
sbatch submit.sh config.sh run_dedup_stage1.sh
```

---

#### Stage 2 — 语义去重 (FAISS)

##### 算法

1. 使用 **Embedding Model**（默认 BGE-M3）将所有 query 编码为 embedding 向量
2. L2 归一化后使用 **FAISS IndexFlatIP** 构建索引
3. Range search 找到余弦相似度高于阈值的 query 对
4. 去除重复 query（保留 ID 较小的条目）

支持多 GPU 并行编码和检索，无 GPU 时自动回退到 CPU。

##### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_dir` | (必填) | 输入目录（parquet） |
| `--output_dir` | (必填) | 输出目录根路径 |
| `--embedding_dir` | (可选) | 预计算 embedding 目录（.npy），为空则现场计算 |
| `--embedding_model_path` | BGE-M3 路径 | Sentence Transformer 模型路径 |
| `--threshold` | `0.9` | 余弦相似度阈值 |
| `--embedding_batch_size` | `1024` | Embedding 编码 batch size |
| `--search_batch_size` | `4096` | 相似度检索 batch size |
| `--query_key` | `query` | Query 列名 |

##### 输出

```
{output_dir}/
├── npy/embeddings.npy                   # 原始 embedding
├── result/                              # 去重后的 parquet 和 embedding
│   └── deduplicated_embeddings.npy
└── dup/similar_queries_*.json           # 相似 query 对（JSON）
```

相似 query 对格式：
```json
{
  "id": "sample_id",
  "query": "query_text",
  "similar_queries": [
    {"id": "neighbor_id", "query": "neighbor_query", "similarity": 0.95}
  ]
}
```

##### 提交任务

```bash
cd deduplication/query_level/stage2
sbatch submit.sh config.sh run_dedup_stage2.sh
```

> 默认申请单节点 8 GPU。

---

## Decontamination

通过 word-level N-gram 匹配检测并移除与评测基准重叠的数据。

### 算法

1. 从 `benchmarks.yaml` 加载各评测集，对文本做归一化（小写 + 空格归一化）
2. 构建 N-gram 查找表：`ngram → {包含该 ngram 的评测条目集合}`
3. 对每条输入数据的 query 生成 word N-gram，匹配是否命中任一评测集
4. 输出清洁数据和污染数据

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_path` | (必填) | 输入文件/目录（JSON/JSONL/Parquet） |
| `--output_dir` | (必填) | 输出目录根路径 |
| `--benchmark_config_path` | `./benchmarks.yaml` | 评测集配置文件 |
| `--query_key` | `query` | 待检测列名 |
| `--ngram_size` | `32` | Word-level N-gram 大小 |
| `--num_proc` | `1` | 并行处理进程数 |

### 输出

```
{output_dir}/
├── contaminated_data/
│   └── contaminated_data.json   # 被污染的样本（含匹配详情）
└── result/                      # 清洁数据 (parquet)
```

污染数据会附加字段：
- `contaminated_{benchmark_name}` (bool) — 是否匹配该评测集
- `contaminated_details` — 匹配的 N-gram 和对应评测条目

### 评测集配置

[benchmarks.yaml](./decontamination/benchmarks.yaml) 中定义要检测的评测集：

```yaml
BENCHMARK_NAME:
  local_path: /path/to/dataset     # HuggingFace 数据集路径
  subset: optional_subset           # 可选，数据子集
  split: test                       # 可选，数据划分
  prompt_key: question              # 文本列名
```

当前已配置的评测集：MMLU-Pro、MMLU、SuperGPQA、AIME-2024/2025、MATH-500、GSM8K、GPQA-Diamond。

### 提交任务

```bash
cd decontamination
sbatch submit.sh config.sh run_decontamination.sh
```

> 单节点 CPU 运行，无需 GPU。

---

## 运行环境

### 依赖

| 包 | 用途 |
|---|---|
| pyspark | 分布式去重 (sample-level, query-level stage1) |
| chukonu | Spark 原生加速插件 |
| sentence-transformers | Embedding 编码 (stage2) |
| faiss-cpu / faiss-gpu | 向量相似度检索 (stage2) |
| torch | GPU 加速 |
| pyarrow, pandas | 数据读写 |
| datasets (HuggingFace) | 加载评测集 |
| numpy, scipy | 数值计算 |
| pyyaml | 配置解析 |

### SLURM 资源参考

| 任务 | 节点数 | GPU | 内存 | 容器 |
|---|---|---|---|---|
| Sample-level 去重 | ≥2 | 无 | 1200G | chukonu+3.4.1-jdk11 |
| Query-level Stage1 | ≥2 | 无 | 1200G | chukonu+3.4.1-jdk11 |
| Query-level Stage2 | 1 | 8 | 1200G | dedup-decon |
| Decontamination | 1 | 无 | 512G | dedup-decon |

### Spark 配置

Spark 任务默认启用：
- Adaptive Query Execution (AQE)
- Kryo 序列化
- Chukonu 插件加速
- 可调参数：`EXECUTOR_CORES` (默认 4)、`EXECUTOR_MEMORY` (默认 32G)、`DEFAULT_PARALLELISM` (默认 16)
