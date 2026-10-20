import argparse
import json

from sim_data_loader import SimulationDataLoader
import vtk_writer as vw


def load_dataset_config(dataset_key: str) -> dict:
    with open("dataset_options.json", "r") as f:
        options = json.load(f)

    return options[dataset_key]


def init_loader(dataset_key: str, dataset: dict | None = None) -> SimulationDataLoader:
    dataset = dataset or load_dataset_config(dataset_key)
    base_path = f"/data/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"

    index_file = f"{dataset['data_type']}_{dataset['data_size']}_index.json"
    
    return SimulationDataLoader(data_pattern, index_file)

def main():
    parser = argparse.ArgumentParser(
        description='Fluid Analysis Utility - Export simulation data'
    )

    parser.add_argument('--dataset', type=str, required=True, help='Type of the dataset')
    parser.add_argument('--output', type=str, required=True, help='Output VTK or CSV file path')
    parser.add_argument(
        '--format',
        dest='output_format',
        choices=('csv', 'vtk'),
        help='Output format (defaults to the format inferred from the output filename)',
    )
    parser.add_argument('x', type=int, help='X coordinate origin')
    parser.add_argument('y', type=int,help='Y coordinate origin')
    parser.add_argument('z', type=int,help='Z coordinate origin')
    parser.add_argument('size', type=int, help='Size of the region to fetch')
    parser.add_argument('columns', type=str, nargs='*', help='Column names to fetch')
    parser.add_argument('velocity', action='store_true', help='Include velocity fields')

    args = parser.parse_args()

    output_format = args.output_format
    if output_format is None:
        if args.output.lower().endswith('.csv'):
            output_format = 'csv'
        elif args.output.lower().endswith('.vtk'):
            output_format = 'vtk'
        else:
            parser.error('Output file must have a .csv or .vtk extension.')
    if not args.output.lower().endswith(f'.{output_format}'):
        parser.error(f'Output file must have a .{output_format} extension.')
    
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

    loader = init_loader(args.dataset)

    columns = args.columns
    if args.velocity_gradient:
        columns = sorted(set(columns) | {"gnx", "gny", "nn", "u", "v", "w"})
    boxcell = loader.fetch_boxcell(
        args.x,
        args.y,
        args.z,
        args.size,
        columns,
        sort_order=['nn', 'gny', 'gnx'],
        streaming=True,
    )

    if output_format == 'csv':
        boxcell.write_csv(args.output)
    elif args.velocity:
        scalar_fields = [x for x in args.columns if x not in ['u', 'v', 'w']]
        vw.save(args.output, boxcell, args.size, args.x, args.y, args.z, scalar_fields, {"velocity": ['u', 'v', 'w']})
    else:
        vw.save(args.output, boxcell, args.size, args.x, args.y, args.z, args.columns)

if __name__ == "__main__":
    main()
