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

    N = dataset['data_size'] // coarsen_size

    k_grid, j_grid, i_grid = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))

    i_flat = np.ravel(i_grid)
    j_flat = np.ravel(j_grid)
    k_flat = np.ravel(k_grid)

    h = coarsen_size // 2

    gnx = np.vstack([
        i_flat * coarsen_size + 1,
        i_flat * coarsen_size + h,
        i_flat * coarsen_size + h,
        (i_flat + 1) * coarsen_size,
        i_flat * coarsen_size + h,
        i_flat * coarsen_size + h
    ])

    gnx = gnx.T.reshape(-1)

    gny = np.vstack([
        j_flat * coarsen_size + h,
        j_flat * coarsen_size + 1,
        j_flat * coarsen_size + h,
        j_flat * coarsen_size + h,
        (j_flat + 1) * coarsen_size,
        j_flat * coarsen_size + h
    ])

    gny = gny.T.reshape(-1)

    nn = np.vstack([
        k_flat * coarsen_size + h,
        k_flat * coarsen_size + h,
        k_flat * coarsen_size + 1,
        k_flat * coarsen_size + h,
        k_flat * coarsen_size + h,
        (k_flat + 1) * coarsen_size
    ])

    nn = nn.T.reshape(-1)

    targets = pl.DataFrame({
        "gnx": gnx,
        "gny": gny,
        "nn": nn
    }, schema={"gnx": pl.Int64, "gny": pl.Int64, "nn": pl.Int64})

    lf_uvw = lf.join(targets, on=["gnx", "gny", "nn"], how="inner")
    df_uvw = lf_uvw.collect(engine='streaming')
    print(df_uvw)
    print(len(df_uvw))

    lf_eps = lf.group_by(["i", "j", "k"]).agg(col("eps").mean().alias("eps_mean"), col("eps").sum().alias("eps_sum")).sort(["k", "j", "i"])
    lf_eps.collect(engine='streaming').write_csv("coarsened_data.csv")

    elapse = time.perf_counter() - start
    print(f"Elapsed time: {elapse}")

if __name__ == "__main__":
    main()