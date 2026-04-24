import argparse
import json
import faiss
import numpy as np
import os
import itertools
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import torch
import torch.multiprocessing as mp

from collections import defaultdict
from multiprocessing import Process, Manager, set_start_method
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_data(parquet_file_dir):
    table = pq.read_table(parquet_file_dir)
    df = table.to_pandas()
    print(f"rows: {len(df)}, columns: {len(df.columns)}, column names: {df.columns.tolist()}")
    return df

def read_partial_data(parquet_file_dir, num_rows=2000000):
    tables = []
    remaining = num_rows

    for fname in sorted(os.listdir(parquet_file_dir)):
        if not fname.endswith(".parquet"):
            continue

        pf = pq.ParquetFile(os.path.join(parquet_file_dir, fname))

        for rg_id in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(rg_id)
            rg_rows = rg_meta.num_rows

            if rg_rows <= remaining:
                table = pf.read_row_group(rg_id)
                tables.append(table)
                remaining -= rg_rows
            else:
                # row group 内再 slice
                table = pf.read_row_group(rg_id).slice(0, remaining)
                tables.append(table)
                remaining = 0

            if remaining <= 0:
                return pa.concat_tables(tables).to_pandas()

    return pa.concat_tables(tables).to_pandas()

def save_parquet(df, output_dir, rows_per_file=200000):
    """
    按n行分片存储parquet文件
    """
    # 如果df类型是Pandas DataFrame，则将其转换为PyArrow Table
    if isinstance(df, pd.DataFrame):
        table = pa.Table.from_pandas(df)
    elif isinstance(df, pa.Table):
        table = df
    else:
        raise ValueError("df must be a Pandas DataFrame or a PyArrow Table")

    # pyarrow.parquet保存parquet文件
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        max_rows_per_file=rows_per_file,
        row_group_size=rows_per_file,
        existing_data_behavior='overwrite_or_ignore', # 覆盖旧数据
        basename_template='deduplicated_data_{i}.parquet'
    )

def encode_on_cpu(queries, model_path, batch_size):
    model = SentenceTransformer(model_path, trust_remote_code=True, device="cpu")
    embeddings = model.encode(queries, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    print(f"Embedding on CPU done. Shape: {embeddings.shape}")
    return embeddings

# referenced https://github.com/InfiXAI/SFT_Data_Pipeline/blob/a1a7b354a78b21a859803a4b60fc48db66f87670/embedding_compute/embedding_gpus.py
def encode_on_gpu(gpu_id: str, df_chunk, query_key, model_path, batch_size, process_idx, return_dict):
    print(f"[GPU {gpu_id}] Process {process_idx} loading model...")
    model = SentenceTransformer(model_path, trust_remote_code=True, device=f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Encoding {len(df_chunk)} rows...")
    embeddings = model.encode(df_chunk[query_key].astype(str).tolist(), show_progress_bar=True, 
                              batch_size=batch_size, convert_to_numpy=True)
    
    return_dict[gpu_id] = embeddings
    print(f"[GPU {gpu_id}] Embedding on process {process_idx} done. Shape: {embeddings.shape}")

def encode_on_gpus(gpu_ids, df_query, query_key, model_path, batch_size):
    # 平均分片
    num_gpus = len(gpu_ids)
    chunks = np.array_split(df_query, num_gpus)

    # 多进程分配到各GPU
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    for idx, gpu_id in enumerate(gpu_ids):
        chunk = chunks[idx]

        p = Process(
            target=encode_on_gpu,
            args=(gpu_id, chunk, query_key, model_path, batch_size, idx, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # 合并各分片的embedding结果
    embeddings = np.vstack([return_dict[str(i)] for i in range(num_gpus)])
    print(f"All GPU embedding done. Shape: {embeddings.shape}")
    return embeddings

def calculate_embedding(df, query_key, output_dir, model_path, batch_size=1024):
    available_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    if not available_gpus:
        print("No GPU available, using CPU for embedding calculation.")
        embeddings = encode_on_cpu(df[query_key].astype(str).tolist(), model_path, batch_size)
    else:
        gpu_ids = available_gpus.split(",")
        embeddings = encode_on_gpus(gpu_ids, df, query_key, model_path, batch_size)

    # 保存embedding结果
    output_dir = os.path.join(output_dir, "npy")
    output_path = os.path.join(output_dir, "embeddings.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")
    return embeddings

def build_faiss_index(embeddings):
    print("Building FAISS index...")
    start_time = time.time()

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    end_time = time.time()
    print(f"FAISS index built in {end_time - start_time:.2f} seconds, {(end_time - start_time) / 60:.2f} minutes")
    return index

def deduplicate_embedding_by_faiss_cpu(df, embeddings, output_dir, query_key="query", threshold=0.9, batch_size=1024):
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = build_faiss_index(embeddings)

    to_remove = np.zeros(embeddings.shape[0], dtype=bool)
    duplicate_cnt = 0
    # key: 被删除的 query_id
    # value: List[(neighbor_id, similarity)], 跟该query相似的那些query
    similar_map = defaultdict(list)

    total_batches = (embeddings.shape[0] + batch_size - 1) // batch_size
    start_time = time.time()
    for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Searching & Filtering", total=total_batches):
        batch_end = min(i + batch_size, embeddings.shape[0])
        batch = embeddings[i:batch_end]
        # 执行范围搜索
        # lims: 每个查询的结果在labels和distances中的起止位置
        # distances: 距离(相似度)数组
        # labels: 邻居id数组
        lims, distances, labels = index.range_search(batch, threshold)

        # 解析结果并标记重复
        # 注意: range_search肯定会包含vector自身(相似度 1.0)
        # 逻辑: 对于query的embedding向量(全局ID为global_id)，如果在结果中发现了相似度大于threshold的neighbor_id，
        # 且neighbor_id < global_id，则说明query是重复的 (保留id小的，删除id大的)
        for j in range(len(batch)):
            global_id = i + j
            
            # 获取当前query的所有邻居
            start = lims[j]
            end = lims[j+1]

            # 只有自己一个结果，说明没有重复
            if end - start <= 1:
                continue
            
            neighbors = labels[start:end]
            sims = distances[start:end]
            
            mask = neighbors < global_id
            if not np.any(mask):
                continue

            to_remove[global_id] = True
            duplicate_cnt += 1
            for nid, sim in zip(neighbors[mask], sims[mask]):
                similar_map[global_id].append((int(nid), float(sim)))

        # 显式清理内存，防止 batch 过大时 Python GC 不及时
        del lims, distances, labels
        
    keep_indices = np.where(~to_remove)[0]
    deduplicated_embeddings = embeddings[keep_indices]
    deduplicated_df = df.iloc[keep_indices].reset_index(drop=True)

    # Save data
    res_dir = os.path.join(output_dir, "result")
    os.makedirs(res_dir, exist_ok=True)

    save_parquet(deduplicated_df, res_dir)
    print(f"Deduplicated data saved to {res_dir}")
    
    dedup_embedding_path = os.path.join(res_dir, "deduplicated_embeddings.npy")
    np.save(dedup_embedding_path, deduplicated_embeddings)
    print(f"Deduplicated embeddings saved to {dedup_embedding_path}")

    # Collect and save similar records
    dup_dir = os.path.join(output_dir, "dup")
    os.makedirs(dup_dir, exist_ok=True)

    chunk_size = 100000
    chunk_counter = 0
    similar_records_buffer = []
    for gid, neighbors in similar_map.items():
        record = {
            "id": df.iloc[gid]["id"],
            "query": df.iloc[gid][query_key],
            "similar_queries": [
                {
                    "id": df.iloc[nid]["id"],
                    "query": df.iloc[nid][query_key],
                    "similarity": sim
                }
                for nid, sim in neighbors
            ]
        }
        similar_records_buffer.append(record)

        if len(similar_records_buffer) >= chunk_size:
                chunk_path = os.path.join(dup_dir, f"similar_queries_{chunk_counter}.json")
                with open(chunk_path, "w", encoding="utf-8") as f:
                    json.dump(similar_records_buffer, f, indent=2, ensure_ascii=False)
                print(f"Saved chunk {chunk_counter} with {len(similar_records_buffer)} records to {chunk_path}")
                total_records += len(similar_records_buffer)
                similar_records_buffer = []
                chunk_counter += 1

    if len(similar_records_buffer) > 0:
        chunk_path = os.path.join(dup_dir, f"similar_queries_{chunk_counter}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(similar_records_buffer, f, indent=2, ensure_ascii=False)
        print(f"Saved chunk {chunk_counter} with {len(similar_records_buffer)} records to {chunk_path}")
        total_records += len(similar_records_buffer)
        similar_records_buffer = []
    
    end_time = time.time()
    print(f"Searching & Filtering completed in {end_time-start_time:.2f} seconds, {(end_time-start_time)/60:.2f} minutes")
    return deduplicated_embeddings.shape[0]

def l2_normalize_torch(t: torch.Tensor, dim=1, eps=1e-12):
    norm = t.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return t.div_(norm)

def gpu_deduplication_worker(embeddings, rank, start_idx, end_idx, return_dict, output_dir, threshold: float = 0.9, batch_size: int = 100):
    """
    GPU工作进程: 计算分配到的Query片段与其他所有向量的相似度
    """
    print(f"[GPU {rank}] Initializing... Processing query rows {start_idx} to {end_idx}")
    device = torch.device(f"cuda:{rank}")
    start_time = time.time()
    
    # 如果单张卡放不下整个矩阵，可能要改成PP形式
    embeddings_gpu = torch.tensor(embeddings, dtype=torch.float32, device=device)
    # L2 normalize
    embeddings_gpu = l2_normalize_torch(embeddings_gpu)
    print(f"[GPU {rank}] Embedding matrix loaded to VRAM. Shape: {embeddings_gpu.shape}, dtype: {embeddings_gpu.dtype}")

    local_rows = []
    local_cols = []
    local_scores = []
    for current_idx in tqdm(range(start_idx, end_idx, batch_size), desc=f"[GPU {rank}]", position=rank):
        batch_end = min(current_idx + batch_size, end_idx)
        # 取出一个batch的query向量
        queries = embeddings_gpu[current_idx:batch_end] # Shape: [batch, 1024]

        # 矩阵乘计算相似度: [batch, 1024] @ [1024, 20M] -> [batch, 20M]
        sim_matrix = torch.matmul(queries, embeddings_gpu.T)
        
        rows, cols = torch.where(sim_matrix > threshold)
        global_rows = rows + current_idx
        # 只保留 global_row < col (即只看上三角)，去除了自身对比和重复对比(j, i)
        mask = global_rows < cols
        if mask.any():
            valid_local_rows = rows[mask]
            valid_cols = cols[mask]
            valid_scores = sim_matrix[valid_local_rows, valid_cols]
            
            local_rows.append(global_rows[mask].cpu().numpy().astype(np.int32))
            local_cols.append(valid_cols.cpu().numpy().astype(np.int32))
            local_scores.append(valid_scores.cpu().numpy().astype(np.float16))
        
        # 清理显存
        del sim_matrix, rows, cols, mask, global_rows

    # 结果汇总
    if local_rows:
        ret_rows = np.concatenate(local_rows)
        ret_cols = np.concatenate(local_cols)
        ret_scores = np.concatenate(local_scores)

        # Save intermediate results to disk to avoid IPC issues with large data
        save_path = os.path.join(output_dir, f"gpu_ret_{rank}.npz")
        np.savez_compressed(save_path, rows=ret_rows, cols=ret_cols, scores=ret_scores)

        return_dict[rank] = save_path
    else:
        return_dict[rank] = None

    end_time = time.time()    
    print(f"[GPU {rank}] Finished, find {len(ret_rows)} duplicated pairs, execution time: {end_time - start_time:.2f} seconds, "
          f"{(end_time - start_time) / 60:.2f} minutes")

def deduplicate_embedding_by_gpus(df, embeddings, output_dir, gpu_ids, query_key="query", threshold=0.9, batch_size=100):
    total_samples = len(embeddings)
    num_gpus = len(gpu_ids)
    chunk_size = total_samples // num_gpus
    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()

    start_time = time.time()
    print("Starting GPU tasks...")

    # Create temp dir for IPC
    temp_ipc_dir = os.path.join(output_dir, "tmp_ipc")
    os.makedirs(temp_ipc_dir, exist_ok=True)

    for i in range(num_gpus):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_gpus - 1 else total_samples
        
        p = mp.Process(target=gpu_deduplication_worker, args=(embeddings, int(gpu_ids[i]), start, end, return_dict, temp_ipc_dir, threshold, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("All GPU tasks finished. Merging results and calculating removal mask...")

    final_rows = []
    final_cols = []
    final_scores = []
    for i in range(num_gpus):
        npz_path = return_dict.get(i)
        if npz_path and os.path.exists(npz_path):
            # Load from disk
            try:
                data = np.load(npz_path)
                final_rows.append(data['rows'])
                final_cols.append(data['cols'])
                final_scores.append(data['scores'])
            except Exception as e:
                print(f"Error loading temp file for gpu {i}: {e}")
                continue
    
    if not final_rows:
        print("No duplicates found.")

    all_rows = np.concatenate(final_rows)
    all_cols = np.concatenate(final_cols)
    all_scores = np.concatenate(final_scores)
    print(f"Total duplicate pairs found: {len(all_rows)}")

    # i4 = int32 (4 bytes), f2 = float16 (2 bytes) -> 每条记录 10 bytes
    dtype = [('keep_idx', 'i4'), ('remove_idx', 'i4'), ('score', 'f2')]
    structured_data = np.zeros(len(all_rows), dtype=dtype)
    structured_data['keep_idx'] = all_rows
    structured_data['remove_idx'] = all_cols
    structured_data['score'] = all_scores
    print(f"structured_data: shape: {structured_data.shape}, dtype: {structured_data.dtype}")

    # 按keep_idx排序，保证处理顺序，即先处理前面的embedding vector，再处理后面的
    sort_idx = np.argsort(structured_data['keep_idx'])
    structured_data = structured_data[sort_idx]
    
    # 取第二列的所有值，也就是按连通图的连通分量去重
    remove_indices = np.unique(structured_data['remove_idx'])
    print(f"Find {len(remove_indices)} samples to remove.")

    # Generate mask
    all_indices = np.arange(len(embeddings))
    keep_mask = np.ones(total_samples, dtype=bool)
    keep_mask[remove_indices] = False
    
    keep_indices = all_indices[keep_mask]
    deduplicated_embeddings = embeddings[keep_indices]
    deduplicated_df = df.iloc[keep_indices].reset_index(drop=True)
    print(f"Deduplicated embeddings shape: {deduplicated_embeddings.shape}, dtype: {deduplicated_embeddings.dtype}")
    print(f"Deduplicated data shape: {deduplicated_df.shape}")

    end_time = time.time()
    print(f"Searching & Filtering completed in {end_time-start_time:.2f} seconds, {(end_time-start_time)/60:.2f} minutes")

    # Save data
    res_dir = os.path.join(output_dir, "result")
    os.makedirs(res_dir, exist_ok=True)

    save_parquet(deduplicated_df, res_dir)
    print(f"Deduplicated data saved to {res_dir}")
    
    dedup_embedding_path = os.path.join(res_dir, "deduplicated_embeddings.npy")
    np.save(dedup_embedding_path, deduplicated_embeddings)
    print(f"Deduplicated embeddings saved to {dedup_embedding_path}")
    res_num = deduplicated_embeddings.shape[0]
    del deduplicated_df, deduplicated_embeddings

    # Collect and save similar records
    print(f"Begin to collect similar records...")
    sim_start_time = time.time()

    dup_dir = os.path.join(output_dir, "dup")
    os.makedirs(dup_dir, exist_ok=True)
    
    chunk_size = 10000
    chunk_counter = 0
    similar_records_buffer = []
    all_ids = df["id"].tolist()
    all_queries = df[query_key].tolist()
    # Vectorized grouping to avoid slow itertools.groupby and row-wise numpy access
    keep_idxs = structured_data['keep_idx']
    remove_idxs = structured_data['remove_idx']
    scores = structured_data['score']
    total_records = 0
    if len(keep_idxs) > 0:
        # Find indices where keep_idx changes (structured_data is already sorted)
        change_points = np.flatnonzero(keep_idxs[1:] != keep_idxs[:-1]) + 1
        start_indices = np.concatenate(([0], change_points))
        end_indices = np.concatenate((change_points, [len(keep_idxs)]))
        
        # TODO: 遍历所有相似对会非常非常慢，因此先只存一个chunk用来人工校验，后续再优化全量相似对数据存储
        unique_gids = keep_idxs[start_indices].tolist()
        for i, gid in enumerate(unique_gids):
            start = start_indices[i]
            end = end_indices[i]
            
            record = {
                "id": all_ids[gid],
                "query": all_queries[gid],
                "similar_queries": [
                    {
                        "id": all_ids[r],
                        "query": all_queries[r],
                        "similarity": float(s)
                    }
                    for r, s in zip(remove_idxs[start:end].tolist(), scores[start:end].tolist())
                ]
            }
            similar_records_buffer.append(record)

            if len(similar_records_buffer) >= chunk_size:
                chunk_path = os.path.join(dup_dir, f"similar_queries_{chunk_counter}.json")
                with open(chunk_path, "w", encoding="utf-8") as f:
                    json.dump(similar_records_buffer, f, indent=2, ensure_ascii=False)
                print(f"Saved chunk {chunk_counter} with {len(similar_records_buffer)} records to {chunk_path}")
                total_records += len(similar_records_buffer)
                similar_records_buffer = []
                chunk_counter += 1
                break # 先只存一个chunk用来人工校验

    if len(similar_records_buffer) > 0:
        chunk_path = os.path.join(dup_dir, f"similar_queries_{chunk_counter}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(similar_records_buffer, f, indent=2, ensure_ascii=False)
        print(f"Saved chunk {chunk_counter} with {len(similar_records_buffer)} records to {chunk_path}")
        total_records += len(similar_records_buffer)
        similar_records_buffer = []

    end_time = time.time()
    print(f"Collected {total_records} similar records (partial) in {end_time-sim_start_time:.2f} seconds, "
          f"{(end_time-sim_start_time)/60:.2f} minutes")
    return res_num


def main():
    start_time = time.time()
    df = read_data(args.input_dir)
    end_time = time.time()
    print(f"Loaded {len(df)} samples from {args.input_dir} in {end_time - start_time:.2f} seconds, "
          f"{(end_time - start_time) / 60:.2f} minutes")

    # 如果没有输入embedding files dir，则计算embedding
    if not args.embedding_dir:
        print("Calculating embedding...")
        start_time = time.time()
        embeddings = calculate_embedding(df, args.query_key, args.output_dir, args.embedding_model_path, 
                                         args.embedding_batch_size)
        end_time = time.time()
        print(f"Embedding shape: {embeddings.shape}, calculation time: {end_time - start_time:.2f} seconds, "
              f"{(end_time - start_time) / 60:.2f} minutes")
    else:
        # 读取embedding file
        print("Loading embedding files...")
        start_time = time.time()
        embedding_files = [f for f in os.listdir(args.embedding_dir) if f.endswith(".npy")]
        if not embedding_files:
            raise ValueError(f"No .npy files found in {args.embedding_dir}")
        embeddings_list = []
        for file in embedding_files:
            file_path = os.path.join(args.embedding_dir, file)
            embeddings = np.load(file_path)
            embeddings_list.append(embeddings)
            print(f"Loaded embedding indice from {file_path}, shape: {embeddings.shape}")
        embeddings = np.vstack(embeddings_list)

        end_time = time.time()
        print(f"Total loaded embeddings shape: {embeddings.shape}, loading time: {end_time - start_time:.2f} seconds, "
              f"{(end_time - start_time) / 60:.2f} minutes")
    
    start_time = time.time()
    available_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    gpu_ids = available_gpus.split(",") if available_gpus else []
    if gpu_ids:
        print(f"Calculate similarity and filter by GPUs: {gpu_ids}")
        res_num = deduplicate_embedding_by_gpus(df, embeddings, args.output_dir, gpu_ids, args.query_key, args.threshold, 
                                                args.search_batch_size)
    else:
        print(f"Calculate similarity and filter by CPU")
        res_num = deduplicate_embedding_by_faiss_cpu(df, embeddings, args.output_dir, args.query_key, args.threshold, 
                                                     args.search_batch_size)

    print("======Statistics======")
    print(f"原始数量: {embeddings.shape[0]}")
    print(f"保留数量: {res_num}")
    print(f"删除数量: {embeddings.shape[0] - res_num}")
    print(f"去重率: {(embeddings.shape[0] - res_num) /embeddings.shape[0]:.2%}")
    
    end_time = time.time()
    print(f"Embedding dedup completed in {end_time - start_time:.2f} seconds, {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--embedding_dir", type=str, help="Embedding files directory")
    parser.add_argument("--embedding_model_path", type=str, default="/work/projects/polyullm/slu/models/bge-m3", help="Embedding model path")
    parser.add_argument("--embedding_batch_size", type=int, default=1024, help="Embedding batch size")
    parser.add_argument("--search_batch_size", type=int, default=4096, help="Similarity calculating & search batch size")
    parser.add_argument("--query_key", type=str, default="query", help="Query key")
    
    args = parser.parse_args()
    print(args)

    set_start_method("spawn")
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Deduplication completed in {end_time - start_time:.2f} seconds, {(end_time - start_time) / 60:.2f} minutes")
