from sim_data_loader import SimulationDataLoader

import argparse
import json
import numpy as np

from numpy.linalg import det
from polars import col, DataFrame, Expr

def fetch_face_mean(df: DataFrame, face_flag: Expr) -> np.ndarray:
    """
    Fetch the mean velocity components for a given face flag.

    ### parameter
    * df: DataFrame - The input DataFrame
    * face_flag: Expr - The flag indicating the face of interest
    """
    return np.array(df.filter(face_flag).select(["u", "v", "w"]).mean().row(0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--size", type=int, required=True, help="Grid size")
    args = parser.parse_args()

    with open("dataset_options.json", "r") as f:
        dataset_options = json.load(f)

    dataset = dataset_options[args.dataset]
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"
    index_file = f"{dataset['data_type']}_{dataset['data_size']}_index.json"

    loader = SimulationDataLoader(data_pattern, index_file)

    size = int(args.size)
    n_iter = size ** 3
    i_list = np.repeat(np.arange(1, size + 1), size * size)
    j_list = np.repeat(np.tile(np.arange(1, size + 1), size), size)
    k_list = np.tile(np.arange(1, size + 1), size * size)

    rr = 2 * np.pi / (size - 1) / 4096
    center_grid = size // 2

    local_x = (col("gnx") - 1) % size + 1
    local_y = (col("gny") - 1) % size + 1
    local_z = (col("nn") - 1) % size + 1

    # center point of surface
    x0_flag = (local_x == 1) & (local_y == center_grid) & (local_z == center_grid)
    y0_flag = (local_x == center_grid) & (local_y == 1) & (local_z == center_grid)
    z0_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == 1)
    x1_flag = (local_x == size) & (local_y == center_grid) & (local_z == center_grid)
    y1_flag = (local_x == center_grid) & (local_y == size) & (local_z == center_grid)
    z1_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == size)

    # points of surface
    # x0_flag = local_x == 1
    # y0_flag = local_y == 1
    # z0_flag = local_z == 1
    # x1_flag = local_x == size
    # y1_flag = local_y == size
    # z1_flag = local_z == size

    strain_magnitude_array = np.zeros(n_iter)
    vorticity_magnitude_array = np.zeros(n_iter)
    q_array = np.zeros(n_iter)
    div_velocity_array = np.zeros(n_iter)
    det_grad_velocity_array = np.zeros(n_iter)
    eps_array = np.zeros(n_iter)

    for i in range(n_iter):
        x = size * (i_list[i] - 1) + 1
        y = size * (j_list[i] - 1) + 1
        z = size * (k_list[i] - 1) + 1

        boxcell = loader.fetch_boxcell(x, y, z, size, ["gnx", "gny", "nn", "u", "v", "w", "eps"])

        boxcell_filtered = boxcell.filter(x0_flag | y0_flag | z0_flag | x1_flag | y1_flag | z1_flag).with_columns([
            x0_flag.alias("x_0_flag"),
            y0_flag.alias("y_0_flag"),
            z0_flag.alias("z_0_flag"),
            x1_flag.alias("x_1_flag"),
            y1_flag.alias("y_1_flag"),
            z1_flag.alias("z_1_flag")
        ])

        u_x0, v_x0, w_x0 = fetch_face_mean(boxcell_filtered, x0_flag)
        u_y0, v_y0, w_y0 = fetch_face_mean(boxcell_filtered, y0_flag)
        u_z0, v_z0, w_z0 = fetch_face_mean(boxcell_filtered, z0_flag)
        u_x1, v_x1, w_x1 = fetch_face_mean(boxcell_filtered, x1_flag)
        u_y1, v_y1, w_y1 = fetch_face_mean(boxcell_filtered, y1_flag)
        u_z1, v_z1, w_z1 = fetch_face_mean(boxcell_filtered, z1_flag)

        velocity_grad_tensor = np.array([
            [u_x1 - u_x0, u_y1 - u_y0, u_z1 - u_z0],
            [v_x1 - v_x0, v_y1 - v_y0, v_z1 - v_z0],
            [w_x1 - w_x0, w_y1 - w_y0, w_z1 - w_z0]
        ]) / rr

        vorticity_tensor = np.zeros((3, 3))
        vorticity_tensor[0, 1] = 0.5 * (velocity_grad_tensor[0, 1] - velocity_grad_tensor[1, 0])
        vorticity_tensor[1, 2] = 0.5 * (velocity_grad_tensor[1, 2] - velocity_grad_tensor[2, 1])
        vorticity_tensor[2, 0] = 0.5 * (velocity_grad_tensor[2, 0] - velocity_grad_tensor[0, 2])
        vorticity_tensor[1, 0] = -vorticity_tensor[0, 1]
        vorticity_tensor[2, 1] = -vorticity_tensor[1, 2]
        vorticity_tensor[0, 2] = -vorticity_tensor[2, 0]

        strain_tensor = velocity_grad_tensor - vorticity_tensor

        strain_magnitude_array[i] = np.sum(strain_tensor ** 2)
        vorticity_magnitude_array[i] = np.sum(vorticity_tensor ** 2)
        q_array[i] = 0.5 * (vorticity_magnitude_array[i] - strain_magnitude_array[i])
        div_velocity_array[i] = np.trace(velocity_grad_tensor)
        det_grad_velocity_array[i] = det(velocity_grad_tensor)

        eps_array[i] = boxcell_filtered.select(col("eps")).mean().row(0)[0]


    df = DataFrame({
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "strain_magnitude": strain_magnitude_array,
        "vorticity_magnitude": vorticity_magnitude_array,
        "q": q_array,
        "div_velocity": div_velocity_array,
        "det_grad_velocity": det_grad_velocity_array,
        "eps": eps_array
    })

    df.write_csv(f"coarsened_{size}.csv")

    df_normalized = df.select([
        col("i"),
        col("j"),
        col("k"),
        (col("strain_magnitude") / col("strain_magnitude").mean()).alias("s2"),
        (col("vorticity_magnitude") / col("vorticity_magnitude").mean()).alias("o2"),
        (col("eps") / col("eps").mean()).alias("eps")
    ]).with_columns((col("o2") - col("s2")).alias("q"))

    df_normalized.write_csv(f"coarsened_normalized_{size}.csv")


if __name__ == "__main__":
    main()