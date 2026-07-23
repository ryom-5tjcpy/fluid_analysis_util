import glob
import json
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

    df = pl.scan_ipc(files)

    expr_i = ((col("gnx") - 1) // coarsen_size + 1).alias("i")
    expr_j = ((col("gny") - 1) // coarsen_size + 1).alias("j")
    expr_k = ((col("nn") - 1) // coarsen_size + 1).alias("k")

    df = df.with_columns([expr_i, expr_j, expr_k])
    df = df.group_by(["i", "j", "k"]).agg(col("eps").count().alias("eps_count"), col("eps").mean().alias("eps_mean"), col("eps").std().alias("eps_std"), col("eps").sum().alias("eps_sum"), col("eps").min().alias("eps_min"), col("eps").max().alias("eps_max"), col("eps").median().alias("eps_median")).sort(["i", "j", "k"])
    df.collect(engine='streaming').write_csv("coarsened_data.csv")

    elapse = time.perf_counter() - start
    print(f"Elapsed time: {elapse}")

if __name__ == "__main__":
    main()