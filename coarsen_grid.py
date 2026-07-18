from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import polars as pl

from sim_data_loader import SimulationDataLoader

def init_loader(dataset: str) -> SimulationDataLoader:
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"

    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"
    
    index_file = f"{dataset['data_type']}_{dataset['data_size']}_index.json"

    return SimulationDataLoader(data_pattern, index_file)

def main():
    dataset_key = "keta2_4096"

    with open("dataset_options.json", "r") as f:
        options = json.load(f)
    
    dataset = options[dataset_key]

    loader = init_loader(dataset)

    size = dataset['data_size']

    coarsen_size = 64

    num_blocks = size // coarsen_size

    block_coords = np.linspace(1, size, num_blocks)

    I, J, K = np.meshgrid(block_coords, block_coords, block_coords, indexing='ij')
    I = np.reshape(I, (-1,))
    J = np.reshape(J, (-1,))
    K = np.reshape(K, (-1,))

    columns = ["gnx", "gny", "nn", "u", "v", "w", "eps"]

    def process_block(coords):
        i, j, k = coords
        gnx = (i - 1) * coarsen_size + 1
        gny = (j - 1) * coarsen_size + 1
        nn = (k - 1) * coarsen_size + 1

        df_fetched = loader.fetch_boxcell(gnx, gny, nn, coarsen_size, columns)
        df_eps = df_fetched.select("eps")
        return df_eps.mean().item()
    
    coords_list = list(zip(I, J, K))

    with ThreadPoolExecutor(max_workers=100) as executor:
        eps = list(executor.map(process_block, coords_list))

    data = np.vstack([I, J, K, eps]).T

    df = pl.LazyFrame(data, schema=["i", "j", "k", "eps"])

    df.collect().write_csv("coarsened_grid.csv")

if __name__ == "__main__":
    main()