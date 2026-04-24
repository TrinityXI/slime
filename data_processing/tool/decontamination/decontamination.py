# TODO: 如果后面有数据量大处理不了的情况，改为使用spark
import argparse
import collections
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time
import yaml

from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, Value
from tqdm import tqdm
from typing import Dict


def read_data(input_path: str):
    if os.path.isfile(input_path):
        dataset = load_dataset(data_files=input_path, split="train")
        return dataset

    dataset_list = []
    json_files = glob.glob(os.path.join(input_path, "*.json")) + glob.glob(os.path.join(input_path, "*.jsonl"))
    if json_files:
        ds = load_dataset("json", data_files=json_files, split="train")
        dataset_list.append(ds)

    parquet_files = glob.glob(os.path.join(input_path, "*.parquet"))
    if parquet_files:
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        dataset_list.append(ds)
    
    if not dataset_list:
        raise ValueError(f"No supported data files (json, jsonl, parquet) found in '{input_path}'")
    
    dataset = concatenate_datasets(dataset_list)
    return dataset

# Referenced from https://github.com/Qihoo360/Light-R1/blob/main/decontaminate/n_gram_check.py
def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text

def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(document)

    # example: {ngram1(str): query1, ngram2(str): query1, ngram3(str): query2,...}
    return lookup

def build_ngram_single(document: str, ngram_size: int = 32) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)

def load_benchmark(benchmark_cfg: Dict[str, any], ngram_size: int = 32) -> Dict[str, any]:
    ngram_lookups = {}
    for name, cfg in benchmark_cfg.items():
        query_key = cfg['prompt_key']
        dataset = load_dataset(path=cfg['local_path'], name=cfg.get('subset', None), split=cfg.get('split', None))
        print(f"Loaded benchmark {name} from {cfg['local_path']}, shape: {dataset.shape}")
        ngram_lookups[name] = build_ngram_lookup(dataset[query_key], ngram_size)
    return ngram_lookups

def main():
    benchmark_cfg = yaml.safe_load(open(args.benchmark_config_path, "r"))
    print(f"Loaded benchmark configuration from {args.benchmark_config_path}")
    print(benchmark_cfg)

    print("-----Build benchmarks' ngram lookup------")
    ngram_lookups = load_benchmark(benchmark_cfg, ngram_size=args.ngram_size)

    print("-----Read data------")
    ds = read_data(args.input_path)
    print(f"Loaded data from {args.input_path}, shape: {ds.shape}")

    print("-----Start decontamination------")
    contamination_counts = defaultdict(int)

    # 指定schema
    features = ds.features.copy()
    for benchmark in ngram_lookups:
        features[f"contaminated_{benchmark}"] = Value("bool")
    features["contaminated_details"] = [{
        "benchmark": Value("string"),
        "matched_ngrams": [Value("string")],
        "eval_dataset_entries": [Value("string")]
    }]

    def find_contaminated(row):
        # For each example we have to build the ngrams and check for all of them on each row
        ngrams = build_ngram_single(row[args.query_key], ngram_size=args.ngram_size)

        details = []
        for benchmark, ngram_lookup in ngram_lookups.items():
            is_contaminated = any(ngram in ngram_lookup for ngram in ngrams)
            matched_ngrams = [ngram for ngram in ngrams if ngram in ngram_lookup] if is_contaminated else []
            
            row[f"contaminated_{benchmark}"] = is_contaminated
            if is_contaminated:
                sample = {
                    "benchmark": benchmark,
                    "matched_ngrams": matched_ngrams,
                    "eval_dataset_entries": list(ngram_lookup[matched_ngrams[0]]) if matched_ngrams else []
                }
                details.append(sample)
        
        row["contaminated_details"] = details if details else None
        return row

    ds = ds.map(find_contaminated, num_proc=args.num_proc, features=features)

    for row in ds:
        for benchmark in ngram_lookups:
            if row[f"contaminated_{benchmark}"]:
                contamination_counts[benchmark] += 1

    print("----------Contaminated Data Statistics----------")
    if not contamination_counts:
        print("No data contamination")
    else:
        for eval_name, count in contamination_counts.items():
            print(f"Dataset: {eval_name}, Contaminated Data Number: {count}")

    print("----------Save Data----------")
    # Save contaminated data
    if contamination_counts:
        contaminated_ds = ds.filter(
            lambda x: x.get("contaminated_details") is not None,
            num_proc=args.num_proc
        )
        contaminated_ds = contaminated_ds.select_columns(["id", args.query_key, "contaminated_details"])

        output_path = os.path.join(args.output_dir, "contaminated_data", "contaminated_data.json")
        contaminated_ds.to_json(output_path, orient="records", lines=False, force_ascii=False, indent=4)
        print(f"Saved {len(contaminated_ds)} contaminated records to {output_path}")

    # Save clean data
    added_keys = [f"contaminated_{benchmark}" for benchmark in ngram_lookups.keys()]
    if contamination_counts:
        clean_ds = ds.filter(
            lambda x: x.get("contaminated_details") is None,
            num_proc=args.num_proc
        )
        clean_ds = clean_ds.remove_columns(added_keys)
    else:
        clean_ds = ds.remove_columns(added_keys)

    output_dir = os.path.join(args.output_dir, "result")
    rows_per_file = 200000
    df = clean_ds.to_pandas()
    # 将 Pandas DataFrame 转换为 PyArrow Table
    table = pa.Table.from_pandas(df)
    # pyarrow.parquet保存parquet文件
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        max_rows_per_file=rows_per_file,
        row_group_size=rows_per_file,
        existing_data_behavior='overwrite_or_ignore', # 覆盖旧数据
        basename_template='decontaminated_data_{i}.parquet'
    )

    print(f"Saved {len(clean_ds)} clean records to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_config_path", type=str, default="./benchmarks.yaml")
    parser.add_argument("--ngram_size", type=int, default=32)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--query_key", type=str, default="query")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes for multiprocessing decontamination")
    args = parser.parse_args()
    print(args)

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Decontamination completed in {end_time - start_time:.2f} seconds, {(end_time - start_time) / 60:.2f} minutes")