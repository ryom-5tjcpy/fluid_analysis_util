import argparse
import numpy as np
import polars as pl

from polars import col

def fetch_face_mean(df: pl.DataFrame, candidate: pl.Expr) -> np.ndarray:
    return np.array(df.filter(candidate).select(["u", "v", "w"]).mean().row(0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input file")
    parser.add_argument("x_grid_corner", type=int, help="X coordinate of the grid corner")
    parser.add_argument("y_grid_corner", type=int, help="Y coordinate of the grid corner")
    parser.add_argument("z_grid_corner", type=int, help="Z coordinate of the grid corner")
    parser.add_argument("grid_size", type=int, help="Grid size")
    args = parser.parse_args()


    lf = pl.scan_csv(args.file)

    x, y, z = args.x_grid_corner, args.y_grid_corner, args.z_grid_corner
    center_grid = args.grid_size // 2

    rr = 2 * np.pi * (args.grid_size - 1) / 4096

    c_x_0: pl.Expr = (col("gnx") == x) & (col("gny") == y + center_grid) & (col("nn") == z + center_grid)
    c_y_0: pl.Expr = (col("gnx") == x + center_grid) & (col("gny") == y) & (col("nn") == z + center_grid)
    c_z_0: pl.Expr = (col("gnx") == x + center_grid) & (col("gny") == y + center_grid) & (col("nn") == z)
    c_x_1: pl.Expr = (col("gnx") == x + args.grid_size - 1) & (col("gny") == y + center_grid) & (col("nn") == z + center_grid)
    c_y_1: pl.Expr = (col("gnx") == x + center_grid) & (col("gny") == y + args.grid_size - 1) & (col("nn") == z + center_grid)
    c_z_1: pl.Expr = (col("gnx") == x + center_grid) & (col("gny") == y + center_grid) & (col("nn") == z + args.grid_size - 1)

    df = lf.filter(c_x_0 | c_y_0 | c_z_0 | c_x_1 | c_y_1 | c_z_1).with_columns([
        c_x_0.alias("face_x_0"),
        c_y_0.alias("face_y_0"),
        c_z_0.alias("face_z_0"),
        c_x_1.alias("face_x_1"),
        c_y_1.alias("face_y_1"),
        c_z_1.alias("face_z_1")
    ]).collect()

    u_x_0, v_x_0, w_x_0 = fetch_face_mean(df, col("face_x_0"))
    u_y_0, v_y_0, w_y_0 = fetch_face_mean(df, col("face_y_0"))
    u_z_0, v_z_0, w_z_0 = fetch_face_mean(df, col("face_z_0"))
    u_x_1, v_x_1, w_x_1 = fetch_face_mean(df, col("face_x_1"))
    u_y_1, v_y_1, w_y_1 = fetch_face_mean(df, col("face_y_1"))
    u_z_1, v_z_1, w_z_1 = fetch_face_mean(df, col("face_z_1"))

    grad_0_0 = (u_x_1 - u_x_0) / rr
    grad_0_1 = (u_y_1 - u_y_0) / rr
    grad_0_2 = (u_z_1 - u_z_0) / rr
    grad_1_0 = (v_x_1 - v_x_0) / rr
    grad_1_1 = (v_y_1 - v_y_0) / rr
    grad_1_2 = (v_z_1 - v_z_0) / rr
    grad_2_0 = (w_x_1 - w_x_0) / rr
    grad_2_1 = (w_y_1 - w_y_0) / rr
    grad_2_2 = (w_z_1 - w_z_0) / rr

    tensor = np.zeros((3, 3))

    tensor[0, 0] = grad_0_0
    tensor[0, 1] = 0.5 * (grad_0_1 + grad_1_0)
    tensor[0, 2] = 0.5 * (grad_0_2 + grad_2_0)
    tensor[1, 0] = 0.5 * (grad_1_0 + grad_0_1)
    tensor[1, 1] = grad_1_1
    tensor[1, 2] = 0.5 * (grad_1_2 + grad_2_1)
    tensor[2, 0] = 0.5 * (grad_2_0 + grad_0_2)
    tensor[2, 1] = 0.5 * (grad_2_1 + grad_1_2)
    tensor[2, 2] = grad_2_2

    s2 = tensor[0, 0] ** 2 + tensor[1, 1] ** 2 + tensor[2, 2] ** 2 + 2 * (tensor[0, 1] ** 2 + tensor[0, 2] ** 2 + tensor[1, 2] ** 2)
    vorticity_magnitude = (grad_2_1 - grad_1_2) ** 2 + (grad_0_2 - grad_2_0) ** 2 + (grad_1_0 - grad_0_1) ** 2

    eps_mean = lf.select(col("eps")).mean()

    print("s2, vorticity_magnitude, eps_mean")
    print(s2, vorticity_magnitude, eps_mean.collect().row(0)[0])

if __name__ == "__main__":
    main()