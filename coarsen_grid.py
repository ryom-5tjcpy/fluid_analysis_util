import glob
import json
import numpy as np
import polars as pl
from polars import col


def main():
    dataset_key = "keta2_4096"

    with open("dataset_options.json", "r") as f:
        options = json.load(f)
    
    dataset = options[dataset_key]

    coarsen_size = 64

    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"
    files = glob.glob(data_pattern)

    import time
    start = time.perf_counter()

    lf = pl.scan_ipc(files)

    expr_i = ((col("gnx") - 1) // coarsen_size + 1).alias("i")
    expr_j = ((col("gny") - 1) // coarsen_size + 1).alias("j")
    expr_k = ((col("nn") - 1) // coarsen_size + 1).alias("k")

    lf = lf.with_columns([expr_i, expr_j, expr_k])

    h = coarsen_size // 2

    local_x = (col("gnx") - 1) % coarsen_size + 1
    local_y = (col("gny") - 1) % coarsen_size + 1
    local_z = (col("nn")  - 1) % coarsen_size + 1

    # 提示された6つのパターンを条件式として定義
    c1 = (local_x == 1) & (local_y == h) & (local_z == h)
    c2 = (local_x == h) & (local_y == 1) & (local_z == h)
    c3 = (local_x == h) & (local_y == h) & (local_z == 1)
    c4 = (local_x == coarsen_size) & (local_y == h) & (local_z == h)
    c5 = (local_x == h) & (local_y == coarsen_size) & (local_z == h)
    c6 = (local_x == h) & (local_y == h) & (local_z == coarsen_size)

    # 6つのパターンのいずれかに合致する行だけをフィルター
    lf_uvw = lf.filter(c1 | c2 | c3 | c4 | c5 | c6).sort(["k", "j", "i", "gnx", "gny", "nn"])

    df_uvw = lf_uvw.collect()
    print(df_uvw)
    print(len(df_uvw))

    #lf_eps = lf.group_by(["i", "j", "k"]).agg(col("eps").mean().alias("eps_mean"), col("eps").sum().alias("eps_sum")).sort(["k", "j", "i"])
    #lf_eps.collect(engine='streaming').write_csv("coarsened_data.csv")

    elapse = time.perf_counter() - start
    print(f"Elapsed time: {elapse}")

if __name__ == "__main__":
    main()