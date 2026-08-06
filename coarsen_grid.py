import glob
import json
import polars as pl
from polars import col
import numpy as np


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
    rr = 2 * np.pi * (coarsen_size - 1) / 4096

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
    lf_uvw = lf.filter(c1 | c2 | c3 | c4 | c5 | c6).with_columns([
        c1.alias("face_x_min"),
        c2.alias("face_y_min"),
        c3.alias("face_z_min"),
        c4.alias("face_x_max"),
        c5.alias("face_y_max"),
        c6.alias("face_z_max")
    ])

    lf_uvw = lf_uvw.group_by(["i", "j", "k"]).agg([
        col('u').filter(col('face_x_min') == True).first().alias('u_x_low'),
        col('u').filter(col('face_x_max') == True).first().alias('u_x_high'),
        col('u').filter(col('face_y_min') == True).first().alias('u_y_low'),
        col('u').filter(col('face_y_max') == True).first().alias('u_y_high'),
        col('u').filter(col('face_z_min') == True).first().alias('u_z_low'),
        col('u').filter(col('face_z_max') == True).first().alias('u_z_high'),
        col('v').filter(col('face_x_min') == True).first().alias('v_x_low'),
        col('v').filter(col('face_x_max') == True).first().alias('v_x_high'),
        col('v').filter(col('face_y_min') == True).first().alias('v_y_low'),
        col('v').filter(col('face_y_max') == True).first().alias('v_y_high'),
        col('v').filter(col('face_z_min') == True).first().alias('v_z_low'),
        col('v').filter(col('face_z_max') == True).first().alias('v_z_high'),
        col('w').filter(col('face_x_min') == True).first().alias('w_x_low'),
        col('w').filter(col('face_x_max') == True).first().alias('w_x_high'),
        col('w').filter(col('face_y_min') == True).first().alias('w_y_low'),
        col('w').filter(col('face_y_max') == True).first().alias('w_y_high'),
        col('w').filter(col('face_z_min') == True).first().alias('w_z_low'),
        col('w').filter(col('face_z_max') == True).first().alias('w_z_high')
    ])

    lf_uvw = lf_uvw.with_columns([
        ((col('u_x_high') - col('u_x_low')) / rr).alias('u_x_grad'),
        ((col('u_y_high') - col('u_y_low')) / rr).alias('u_y_grad'),
        ((col('u_z_high') - col('u_z_low')) / rr).alias('u_z_grad'),
        ((col('v_x_high') - col('v_x_low')) / rr).alias('v_x_grad'),
        ((col('v_y_high') - col('v_y_low')) / rr).alias('v_y_grad'),
        ((col('v_z_high') - col('v_z_low')) / rr).alias('v_z_grad'),
        ((col('w_x_high') - col('w_x_low')) / rr).alias('w_x_grad'),
        ((col('w_y_high') - col('w_y_low')) / rr).alias('w_y_grad'),
        ((col('w_z_high') - col('w_z_low')) / rr).alias('w_z_grad')
    ])

    lf_uvw = lf_uvw.with_columns([
        (0.5 * (col('u_y_grad') + col('v_x_grad'))).alias('s_12'),
        (0.5 * (col('v_z_grad') + col('w_y_grad'))).alias('s_23'),
        (0.5 * (col('w_x_grad') + col('u_z_grad'))).alias('s_31'),
        (col('w_y_grad') - col('v_z_grad')).alias('vorticity_x'),
        (col('u_z_grad') - col('w_x_grad')).alias('vorticity_y'),
        (col('v_x_grad') - col('u_y_grad')).alias('vorticity_z')
    ])

    lf_uvw = lf_uvw.with_columns(
        (col('u_x_grad').pow(2) + col('v_y_grad').pow(2) + col('w_z_grad').pow(2) + 2 * col('s_12').pow(2) + 2 * col('s_23').pow(2) + 2 * col('s_31').pow(2)).alias('s2_row'),
        (col('vorticity_x').pow(2) + col('vorticity_y').pow(2) + col('vorticity_z').pow(2)).alias('vorticity_magnitude'),
        (col('s2_row') * col('vorticity_magnitude')).sqrt().alias('s2_vorticity'),
    )

    lf_uvw = lf_uvw.with_columns(
        (col('s2_row') / col('s2_row').mean()).alias('s2'),
        (col('vorticity_magnitude') / col('vorticity_magnitude').mean()).alias('o2'),
        (col('s2_vorticity') / col('s2_vorticity').mean()).alias('so')
    ).sort(["i", "j", "k"])

    #lf_eps = lf.group_by(["i", "j", "k"]).agg(col("eps").mean())

    print("Executing queries")

    df_uvw = lf_uvw.collect(engine="streaming")
    df_uvw.write_csv("coarsened_uvw_data.csv")

    #df_eps = df_eps.sort(["i", "j", "k"])
    #f_eps.write_csv("coarsened_data.csv")

    elapse = time.perf_counter() - start
    print(f"Elapsed time: {elapse}")

if __name__ == "__main__":
    main()