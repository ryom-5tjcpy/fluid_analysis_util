from sim_data_loader import SimulationDataLoader

import json

from polars import col

def main():
    with open("dataset_options.json", "r") as f:
        dataset_options = json.load(f)

    dataset = dataset_options["keta2_4096"]
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"
    index_file = f"{dataset['data_type']}_{dataset['data_size']}_index.json"

    loader = SimulationDataLoader(data_pattern, index_file)

    i_list = [i for i in range(64) for _ in range(64) for _ in range(64)]
    j_list = [j for _ in range(64) for j in range(64) for _ in range(64)]
    k_list = [k for _ in range(64) for _ in range(64) for k in range(64)]

    size = 64

    from numpy import pi

    rr = 2 * pi / (size - 1) / 4096
    center_grid = size // 2

    local_x = (col("gnx") - 1) % size + 1
    local_y = (col("gny") - 1) % size + 1
    local_z = (col("nn") - 1) % size + 1

    x_0_flag = (local_x == 1) & (local_y == center_grid) & (local_z == center_grid)
    y_0_flag = (local_x == center_grid) & (local_y == 1) & (local_z == center_grid)
    z_0_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == 1)
    x_1_flag = (local_x == size) & (local_y == center_grid) & (local_z == center_grid)
    y_1_flag = (local_x == center_grid) & (local_y == size) & (local_z == center_grid)
    z_1_flag = (local_x == center_grid) & (local_y == center_grid) & (local_z == size)

    for i in range(5):
        x = size * i_list[i] + 1
        y = size * j_list[i] + 1
        z = size * k_list[i] + 1

        boxcell = loader.fetch_boxcell(x, y, z, size, ["gnx", "gny", "nn", "u", "v", "w", "eps"])

        boxcell = boxcell.filter(x_0_flag | y_0_flag | z_0_flag | x_1_flag | y_1_flag | z_1_flag).with_columns([
            x_0_flag.alias("x_0_flag"),
            y_0_flag.alias("y_0_flag"),
            z_0_flag.alias("z_0_flag"),
            x_1_flag.alias("x_1_flag"),
            y_1_flag.alias("y_1_flag"),
            z_1_flag.alias("z_1_flag")
        ])

        print(boxcell)

if __name__ == "__main__":
    main()