import argparse
import json

import numpy as np
from polars import col, scan_ipc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--size", type=int, required=True, help="Grid size")
    parser.add_argument(
        "--engine",
        choices=("auto", "in-memory", "streaming", "gpu"),
        default="auto",
        help="Engine used to process the query"
    )
    args = parser.parse_args()

    with open("dataset_options.json", "r") as f:
        dataset_options = json.load(f)

    dataset = dataset_options[args.dataset]
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"

    size = int(args.size)
    coarsened_size = dataset['data_size'] // size

    lf = scan_ipc(data_pattern)

    i = (col("gnx") - 1) // size + 1
    j = (col("gny") - 1) // size + 1
    k = (col("nn") - 1) // size + 1
    local_x = (col("gnx") - 1) % size + 1
    local_y = (col("gny") - 1) % size + 1
    local_z = (col("nn") - 1) % size + 1

    lf = lf.with_columns([
        i.alias("i"),
        j.alias("j"),
        k.alias("k"),
    ])
    

    rr = 2 * np.pi / (size - 1) / 4096
    # center_grid = size // 2

    # a center point of surface
    # x0_flag = (local_x == 1) & (local_y == center_grid) & (local_z == center_grid)
    # y0_flag = (local_x == center_grid) & (local_y == 1) & (local_z == center_grid)
    # z0_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == 1)
    # x1_flag = (local_x == size) & (local_y == center_grid) & (local_z == center_grid)
    # y1_flag = (local_x == center_grid) & (local_y == size) & (local_z == center_grid)
    # z1_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == size)
    
    # points of surface
    x0_flag = local_x == 1
    y0_flag = local_y == 1
    z0_flag = local_z == 1
    x1_flag = local_x == size
    y1_flag = local_y == size
    z1_flag = local_z == size

    lf = lf.group_by(["i", "j", "k"]).agg([
        # surface of x0
        col("u").filter(x0_flag).mean().alias("u_x0"),
        col("v").filter(x0_flag).mean().alias("v_x0"),
        col("w").filter(x0_flag).mean().alias("w_x0"),
        # surface of y0
        col("u").filter(y0_flag).mean().alias("u_y0"),
        col("v").filter(y0_flag).mean().alias("v_y0"),
        col("w").filter(y0_flag).mean().alias("w_y0"),
        # surface of z0
        col("u").filter(z0_flag).mean().alias("u_z0"),
        col("v").filter(z0_flag).mean().alias("v_z0"),
        col("w").filter(z0_flag).mean().alias("w_z0"),
        #surface of x1
        col("u").filter(x1_flag).mean().alias("u_x1"),
        col("v").filter(x1_flag).mean().alias("v_x1"),
        col("w").filter(x1_flag).mean().alias("w_x1"),
        # surface of y1
        col("u").filter(y1_flag).mean().alias("u_y1"),
        col("v").filter(y1_flag).mean().alias("v_y1"),
        col("w").filter(y1_flag).mean().alias("w_y1"),
        # surface of z1
        col("u").filter(z1_flag).mean().alias("u_z1"),
        col("v").filter(z1_flag).mean().alias("v_z1"),
        col("w").filter(z1_flag).mean().alias("w_z1"),
        # eps mean
        col("eps").mean().alias("eps")
    ])

    du_dx = (col("u_x1") - col("u_x0")) / rr
    du_dy = (col("u_y1") - col("u_y0")) / rr
    du_dz = (col("u_z1") - col("u_z0")) / rr

    dv_dx = (col("v_x1") - col("v_x0")) / rr
    dv_dy = (col("v_y1") - col("v_y0")) / rr
    dv_dz = (col("v_z1") - col("v_z0")) / rr

    dw_dx = (col("w_x1") - col("w_x0")) / rr
    dw_dy = (col("w_y1") - col("w_y0")) / rr
    dw_dz = (col("w_z1") - col("w_z0")) / rr

    # Vorticity Tensor
    w_01 = 0.5 * (du_dy - dv_dx)
    w_02 = 0.5 * (du_dz - dw_dx)
    w_12 = 0.5 * (dv_dz - dw_dy)

    # Strain Tensor
    s_00 = du_dx
    s_11 = dv_dy
    s_22 = dw_dz
    s_01 = 0.5 * (du_dy + dv_dx)
    s_02 = 0.5 * (du_dz + dw_dx)
    s_12 = 0.5 * (dv_dz + dw_dy)

    # Sum of Squares
    # Strain Tensor Norm
    s_norm = s_00**2 + s_11**2 + s_22**2 + 2*(s_01**2 + s_02**2 + s_12**2)

    # Sum of Squares
    # Vorticity Tensor Norm
    w_norm = 2 * (w_01**2 + w_02**2 + w_12**2)

    # Determinant of Gradient Velocity Tensor
    det_grad_u = du_dx * (dv_dy * dw_dz - dv_dz * dw_dy) - du_dy * (dv_dx * dw_dz - dv_dz * dw_dx) + du_dz * (dv_dx * dw_dy - dv_dy * dw_dx)

    div_grad_u = du_dx + dv_dy + dw_dz

    q = 0.5 * (w_norm - s_norm)

    df = lf.with_columns([
        s_norm.alias("strain_magnitude"),
        w_norm.alias("vorticity_magnitude"),
        det_grad_u.alias("det_gradient_velocity"),
        div_grad_u.alias("div_grad_velocity"),
        q.alias("q_value")
    ]).select([
        "i", "j", "k",
        "strain_magnitude", "vorticity_magnitude",
        "det_gradient_velocity", "div_grad_velocity",
        "q_value", "eps"
    ]).sort(['i', 'j', 'k']).collect(engine=args.engine)

    df.write_csv(f"coarsened_{args.dataset}_{coarsened_size}.csv")
    df.write_ipc(f"coarsened_{args.dataset}_{coarsened_size}.arrow")


if __name__ == "__main__":
    main()