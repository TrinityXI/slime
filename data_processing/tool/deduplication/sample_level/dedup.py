import argparse
import hashlib
import re
import string
from unicodedata import normalize
from struct import unpack as byrunpack
from itertools import tee
from typing import List, Text, Tuple

import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType, MapType
from scipy.integrate import quad as integrate
import time
from chukonu import invoke
from chukonu.config import find_chukonu_lib_path
from chukonu.library import weakly_connected_component_spark
from pyspark.sql.types import StructType, StructField, LongType, StringType

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


SEED = 42
NON_ALPHA = re.compile("\W", re.UNICODE)
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
        The first 4 bytes (32 bits) of the SHA1 hash of the input data.

    Examples
    --------
    >>> sha1_hash32(b"hello")
    499578026
    >>> bin(sha1_hash32(b"hello"))
    '0b11101110001101111010010101010'
    >>> sha1_hash32(b"hello world").bit_length()
    30
    """
    return byrunpack("<I", hashlib.sha1(data).digest()[:4])[0]


def generate_hash_values(
        content: str,
        idx: int,
        num_perm: int,
        ngram_size: int,
        min_length: int,
        hashranges: List[Tuple[int, int]],
        permutations: np.ndarray,
) -> List[Tuple[int, bytes, int]]:
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size, min_length)}
    hv = np.array([sha1_hash32(token.lower().encode("utf-8")) for token in tokens], dtype=np.uint64)
    a, b = permutations
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]

def optimal_param(
        threshold: float,
        num_perm: int,
        false_positive_weight: float = 0.5,
        false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(area, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(area, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.

    Examples
    --------
    >>> generate_edges([1, 2, 3])
    [(2, 1), (3, 1)]
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏" + string.punctuation
pattern = None

def set_pattern(punc, use_split=False):
    global pattern
    if use_split:
        pattern = re.compile(r"[\s%s]+|(?<=[\u4e00-\u9fff])(?=[\u4e00-\u9fff])" % re.escape(punc))
    else:
        pattern = re.compile(r"[\s%s]+" % re.escape(punc))


def normal_str_udf(content: str):
    if content is None:
        return None
    # return punc_pattern.sub(" ", normalize("NFKC", content))
    return pattern.sub(" ", normalize("NFKC", content.lower()))


def run_chukonu(spark: SparkSession, args, parquet_file_path: str, result_path: str, dup_path: str, wcc_path: str, dedup_path: str):
    if args.b is None or args.r is None:
        B, R = optimal_param(args.threshold, args.num_perm)
    else:
        B, R = args.b, args.r

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(args.num_perm)
        ],
        dtype=np.uint64,
    ).T

    if args.file_type == "json":
        df = spark.read.json(parquet_file_path)
    else:
        df = spark.read.parquet(parquet_file_path)

    print(f"Readed {df.count()} samples from {parquet_file_path}")
    print(f"DataFrame columns: {df.columns}")
    print(f"DataFrame schema: {df.schema}")

    text_col_name = "text"
    # list等复杂对象需转为字符串再去重，否则去重时会报错
    if isinstance(df.schema[args.text_key].dataType, (ArrayType, StructType, MapType)):
        df = df.withColumn(text_col_name, F.to_json(F.col(args.text_key)))
    else:
        df = df.withColumn(text_col_name, F.col(args.text_key))

    df = df.dropDuplicates(subset=[text_col_name])
    # 存储数据前把添加的text_col_name列删掉
    dedup_df = df.drop(text_col_name)
    dedup_df.write.mode("overwrite").parquet(dedup_path)
    print(f"Total number of samples after deduplication: {dedup_df.count()}")

    normal_str = F.udf(normal_str_udf, StringType())
    uid_df = df.withColumn("uid", F.monotonically_increasing_id())
    records = uid_df.withColumn("content", normal_str(F.col(text_col_name))).select("uid", "content")

    minhash_schema = StructType(
        [
            StructField("src", LongType(), False),
            StructField("dst", LongType(), False),
        ]
    )
    
    wcc_schema = StructType(
        [
            StructField("vid", LongType(), False),
            StructField("component", LongType(), False),
        ]
    )
    
    perm0 = ",".join([str(it) for it in PERMUTATIONS[0]])
    perm1 = ",".join([str(it) for it in PERMUTATIONS[1]])

    components = invoke(
        spark,
        f"{find_chukonu_lib_path()}/libminhash_chukonu_mod.so",
        [records],
        [str(B), str(R), str(args.num_perm), str(args.ngram_size), str(args.min_length), perm0, perm1],
        minhash_schema,
    )

    wcc = invoke(
        spark,
        f"{find_chukonu_lib_path()}/wcc.so",
        [components],
        [str(args.num_parallel)],
        wcc_schema,
    )
    
    wcc.write.mode("overwrite").parquet(wcc_path) # wcc存盘
    wcc_filter = wcc.filter(F.col("vid") > F.col("component"))
    
    records_filter = uid_df.join(wcc_filter, uid_df.uid == wcc_filter.vid, "left_anti")
    records_dup = uid_df.join(wcc_filter, uid_df.uid == wcc_filter.vid, "left_semi")
    
    # 存储数据前把添加的text_col_name、uid列删掉
    records_filter = records_filter.drop(text_col_name, "uid")
    records_dup = records_dup.drop(text_col_name, "uid")
    records_filter.write.mode("overwrite").parquet(result_path)
    records_dup.write.mode("overwrite").parquet(dup_path)


def run_spark(spark: SparkSession, args, parquet_file_path: str, result_path: str, dup_path: str, wcc_path: str, dedup_path: str):
    if args.b is None or args.r is None:
        B, R = optimal_param(args.threshold, args.num_perm)
    else:
        B, R = args.b, args.r

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(args.num_perm)
        ],
        dtype=np.uint64,
    ).T

    if args.file_type == "json":
        df = spark.read.json(parquet_file_path)
    else:
        df = spark.read.parquet(parquet_file_path)
    
    text_col_name = "text"
    # list等复杂对象需转为字符串再去重，否则去重时会报错
    if isinstance(df.schema[args.text_key].dataType, (ArrayType, StructType, MapType)):
        df = df.withColumn(text_col_name, F.to_json(F.col(args.text_key)))
    else:
        df = df.withColumn(text_col_name, F.col(args.text_key))

    df = df.dropDuplicates(subset=[text_col_name])
    # 存储数据前把添加的text_col_name列删掉
    dedup_df = df.drop(text_col_name)
    dedup_df.write.mode("overwrite").parquet(dedup_path)
    print(f"Total number of samples after deduplication: {dedup_df.count()}")

    normal_str = F.udf(normal_str_udf, StringType())
    uid_df = df.withColumn("uid", F.monotonically_increasing_id())
    records = uid_df.withColumn("content", normal_str(F.col(text_col_name))).select("uid", "content")

    edges = (
        records
        .rdd
        .flatMap(
            lambda x: generate_hash_values(
                content=x[1],
                idx=x[0],
                num_perm=args.num_perm,
                ngram_size=args.ngram_size,
                min_length=args.min_length,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
            )
        )
        .groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
        .distinct()
    )

    components = spark.createDataFrame(edges, schema=["src", "dst"])

    wcc = weakly_connected_component_spark(components, args.num_parallel)

    wcc.write.mode("overwrite").parquet(wcc_path) # wcc存盘
    wcc_filter = wcc.filter(F.col("vid") > F.col("component"))
    
    records_filter = uid_df.join(wcc_filter, uid_df.uid == wcc_filter.vid, "left_anti")
    records_dup = uid_df.join(wcc_filter, uid_df.uid == wcc_filter.vid, "left_semi")
    # 存储数据前把添加的text_col_name、uid列删掉
    records_filter = records_filter.drop(text_col_name, "uid")
    records_dup = records_dup.drop(text_col_name, "uid")
    records_filter.write.mode("overwrite").parquet(result_path)
    records_dup.write.mode("overwrite").parquet(dup_path)


def redundancy(parquet_file_path, dup_path):
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    if args.file_type == "json":
        df = spark.read.json(parquet_file_path)
    else:
        df = spark.read.parquet(parquet_file_path)

    dup_records = spark.read.parquet(dup_path)
    num_doc = df.count()
    
    num_dup = dup_records.count()
    print(f"重复样本数: {num_dup}, 总样本数: {num_doc}, 重复率: {num_dup * 100/num_doc}%")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Minhash_wcc with PySpark or Chukonu")
    parser.add_argument("--num-parallel", type=int, default=3200, help="Number of parallel tasks")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument("--min_length", type=int, default=2, help="Minimum length of document to be considered")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations")
    parser.add_argument("--b", type=int, default=8, help="Number of bands")
    parser.add_argument("--r", type=int, default=16, help="Number of rows per band")
    parser.add_argument("--with_split", type=str2bool, default=False, help="Use split or not")
    parser.add_argument("--run_chukonu", type=str2bool, default=True, help="Use chukonu or not")
    parser.add_argument("--input_path", type=str, required=True, help="Input path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--file_type", type=str, default="parquet", choices=["json", "parquet"], help="File type (json or parquet)")
    parser.add_argument("--text_key", type=str, default="text", help="Text key")
    args = parser.parse_args()
    
    print(args)

    parquet_file_path = args.input_path
    dedup_path = args.output_path + "/dedup" # 存储经过精确去重(Exact Deduplication)后的数据
    wcc_path = args.output_path + "/wcc" # 存储弱连通分量(Weakly Connected Components)的计算结果表，记录哪些数据被MinHash+LSH算法判定为相似并归为了一组
    dup_path = args.output_path + "/dup" # 存储被判定为重复的数据(经过MinHash+LSH、WCC计算后认为相似的数据)
    result_path = args.output_path + "/result" # 存储完成去重(精确去重+相似去重)后的最终结果

    conf = SparkConf()
    conf.set("spark.app.name", "MinHashLSH.dedup_all")
    conf.set("spark.debug.maxToStringFields", "100")
    conf.set("hive.exec.dynamic.partition", "true")
    conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    mid_time = time.time()

    set_pattern(punc, args.with_split)
    if args.run_chukonu:
        run_chukonu(spark, args, parquet_file_path, result_path, dup_path, wcc_path, dedup_path)
    else:
        run_spark(spark, args, parquet_file_path, result_path, dup_path, wcc_path, dedup_path)

    end_time = time.time()

    # 计算冗余率
    redundancy(parquet_file_path, dup_path)
    
    print(f"准备时间: {mid_time - start_time:.4f} 秒，执行时间: {end_time - mid_time:.4f} 秒，总时间: {end_time - start_time:.4f} 秒")
