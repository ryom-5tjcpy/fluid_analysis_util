import json
import numpy as np
import polars as pl

from sim_data_loader import SimulationDataLoader

def init_loader(dataset: str) -> SimulationDataLoader:
    base_path = f"data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['data_step']}"

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

    I = np.tile(np.linspace(1, coarsen_size, coarsen_size), (size // coarsen_size) ** 2)
    J = np.repeat(np.tile(np.linspace(1, coarsen_size, coarsen_size), size // coarsen_size), size // coarsen_size)
    K = np.repeat(np.linspace(1, coarsen_size, coarsen_size), (size // coarsen_size) ** 2)

    columns = ['gnx', 'gny', 'nn', 'u', 'v', 'w', 'eps']

    eps = np.zeros_like(I, dtype=float)

    for i in I:
        gnx = (i - 1) * coarsen_size + 1
        for j in J:
            gny = (j - 1) * coarsen_size + 1
            for k in K:
                nn = (k - 1) * coarsen_size + 1

                df_fetched = loader.fetch_boxcell(gnx, gny, nn, coarsen_size, columns)

                df_eps = df_fetched.select("eps")
                eps[k * coarsen_size + j * coarsen_size + i] = df_eps.mean().item()
    data = np.vstack([I, J, K, eps]).T

    df = pl.LazyFrame(data, schema=["i", "j", "k", "eps"])

    df.collect().write_csv("coarsened_grid.csv")
