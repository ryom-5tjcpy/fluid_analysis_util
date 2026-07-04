import argparse
import json

from sim_data_loader import SimulationDataLoader
import vtk_writer as vw


def init_loader(dataset_key: str) -> SimulationDataLoader:
    with open("dataset_options.json", "r") as f:
        options = json.load(f)

    dataset = options[dataset_key]
    base_path = f"/fast/jh240062/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    
    data_pattern = f"{base_path}/{dataset['data_type']}_{dataset['data_size']}_*.arrow"
    
    return SimulationDataLoader(data_pattern)

def main():
    parser = argparse.ArgumentParser(
        description='Fluid Analysis Utility - Convert simulation data to VTK format'
    )

    parser.add_argument('--dataset', type=str, required=True, help='Type of the dataset')
    parser.add_argument('--output', type=str, required=True, help='Output VTK file path')
    parser.add_argument('x', type=int, help='X coordinate origin')
    parser.add_argument('y', type=int,help='Y coordinate origin')
    parser.add_argument('z', type=int,help='Z coordinate origin')
    parser.add_argument('size', type=int, help='Size of the region to fetch')
    parser.add_argument('columns', type=str, nargs='*', help='Column names to fetch')
    parser.add_argument('--velocity', action='store_true', help='Include velocity fields')

    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Output file: {args.output}")
    print(f"Origin: ({args.x}, {args.y}, {args.z})")
    print(f"Size: {args.size}")
    print(f"Columns to fetch: {args.columns}")

    if args.velocity and ['u', 'v', 'w'] not in args.columns:
        print("Error: Velocity fields requested but not specified in columns.")
        exit(1)
    
    if not str.endswith(args.output, ".vtk"):
        print("Error: Output file must have a .vtk extension.")
        exit(1)

    loader = init_loader(args.dataset)

    boxcell = loader.fetch_boxcell(args.x, args.y, args.z, args.size, args.columns, sort_order=['nn', 'gny', 'gnx'])

    if args.velocity:
        scalar_fields = [x for x in args.columns if x not in ['u', 'v', 'w']]
        vw.save(args.output, boxcell, args.size, args.x, args.y, args.z, scalar_fields, {"velocity": ['u', 'v', 'w']})
    else:
        vw.save(args.output, boxcell, args.size, args.x, args.y, args.z, args.columns)

if __name__ == "__main__":
    main()
