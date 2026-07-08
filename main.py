import argparse
import json

from sim_data_loader import SimulationDataLoader
import vtk_writer as vw


def init_loader(dataset_key: str) -> SimulationDataLoader:
    with open("dataset_options.json", "r") as f:
        options = json.load(f)

    dataset = options[dataset_key]
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"

    inde_file = f"{dataset['data_type']}_{dataset['data_size']}_index.json"
    
    return SimulationDataLoader(data_pattern, inde_file)

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

    if args.velocity:
        required_velocity_columns = {"u", "v", "w"}
        provided_columns = set(args.columns)
        missing_columns = required_velocity_columns - provided_columns

        if missing_columns:
            print(
                "Error: --velocity requires the following columns: u, v, w. "
                f"Missing: {', '.join(sorted(missing_columns))}."
            )
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
