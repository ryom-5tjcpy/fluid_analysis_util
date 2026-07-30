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

    #file = f"{base_path}/data/keta2_4096_00000_1_1.arrow"

    import time
    start = time.perf_counter()

    lf = pl.scan_ipc(files)
    #lf = pl.scan_ipc(file)

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
    lf_uvw = lf.filter(c1 | c2 | c3 | c4 | c5 | c6).with_columns(
        pl.when(local_x == 1).then(pl.lit("x_low")).when(local_x == coarsen_size).then(pl.lit("x_high"))
        .when(local_y == 1).then(pl.lit("y_low")).when(local_y == coarsen_size).then(pl.lit("y_high"))
        .when(local_z == 1).then(pl.lit("z_low")).when(local_z == coarsen_size).then(pl.lit("z_high"))
        .alias("face"))

    df_uvw = lf_uvw.collect()

    df_uvw = df_uvw.pivot(on="face", index=["i", "j", "k"], values=["u", "v", "w"])

    df_uvw = df_uvw.with_columns([
        ((col('u_x_high') - col('u_x_low')) / (coarsen_size - 1)).alias('u_x_grad'),
        ((col('u_y_high') - col('u_y_low')) / (coarsen_size - 1)).alias('u_y_grad'),
        ((col('u_z_high') - col('u_z_low')) / (coarsen_size - 1)).alias('u_z_grad'),
        ((col('v_x_high') - col('v_x_low')) / (coarsen_size - 1)).alias('v_x_grad'),
        ((col('v_y_high') - col('v_y_low')) / (coarsen_size - 1)).alias('v_y_grad'),
        ((col('v_z_high') - col('v_z_low')) / (coarsen_size - 1)).alias('v_z_grad'),
        ((col('w_x_high') - col('w_x_low')) / (coarsen_size - 1)).alias('w_x_grad'),
        ((col('w_y_high') - col('w_y_low')) / (coarsen_size - 1)).alias('w_y_grad'),
        ((col('w_z_high') - col('w_z_low')) / (coarsen_size - 1)).alias('w_z_grad')
    ])

    df_uvw = df_uvw.with_columns([
        (0.25 * ((col('u_y_grad') + col('v_x_grad')) ** 2)).alias('s_12'),
        (0.25 * ((col('v_z_grad') + col('w_y_grad')) ** 2)).alias('s_23'),
        (0.25 * ((col('w_x_grad') + col('u_z_grad')) ** 2)).alias('s_31')
    ])

    df_uvw = df_uvw.with_columns(
        (col('u_x_grad') ** 2 + col('v_y_grad') ** 2 + col('w_z_grad') ** 2 + 2 * col('s_12') + 2 * col('s_23') + 2 * col('s_31')).alias('s2_row')
    )

    df_uvw = df_uvw.with_columns(
        (col('s2_row') / col('s2_row').mean()).alias('s2')
    )

    print(df_uvw.head())

    lf_eps = lf.group_by(["i", "j", "k"]).agg(col("eps").mean()).sort(["i", "j", "k"])
    lf_eps.collect(engine='streaming').write_csv("coarsened_data.csv")

    elapse = time.perf_counter() - start
    print(f"Elapsed time: {elapse}")

if __name__ == "__main__":
    main()