#!/usr/bin/env python3
"""
Benchmark script to compare ThreadPoolExecutor vs ProcessPoolExecutor for create_index.
Usage: python benchmark_parallel.py --dataset <dataset_key>
"""

import argparse
import json
import time
from sim_data_loader import SimulationDataLoader


def benchmark_create_index(dataset_key: str, use_processes: bool, max_workers: int = None):
    """Benchmark create_index with specified executor type."""
    with open("dataset_options.json", "r") as f:
        options = json.load(f)

    dataset = options[dataset_key]
    base_path = f"/fast/jh240062/arrow_files/{dataset['data_type']}_{dataset['data_size']}_{dataset['step']}"
    data_pattern = f"{base_path}/data/{dataset['data_type']}_{dataset['data_size']}_*.arrow"

    loader = SimulationDataLoader.__new__(SimulationDataLoader)
    loader.pattern = data_pattern
    loader.index_file = "file_index_benchmark.json"
    loader.files = sorted(__import__('glob').glob(data_pattern))

    if not loader.files:
        print(f"No files found for pattern: {data_pattern}")
        return None

    print(f"\nBenchmarking with {len(loader.files)} files...")
    print(f"Executor: {'ProcessPool' if use_processes else 'ThreadPool'}")
    print(f"Max workers: {max_workers if max_workers else 'auto'}")

    start = time.time()
    loader.create_index(max_workers=max_workers, use_processes=use_processes)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Files indexed: {len(loader.file_index)}")
    print(f"Throughput: {len(loader.files) / elapsed:.2f} files/sec")
    
    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Benchmark create_index parallel execution')
    parser.add_argument('--dataset', type=str, required=True, help='Type of the dataset')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (None=auto)')

    args = parser.parse_args()

    print("=" * 60)
    print("Parallel Executor Benchmark")
    print("=" * 60)

    # Benchmark ThreadPoolExecutor
    print("\n[1/2] ThreadPoolExecutor (I/O-bound)")
    time_thread = benchmark_create_index(args.dataset, use_processes=False, max_workers=args.workers)

    # Benchmark ProcessPoolExecutor
    print("\n[2/2] ProcessPoolExecutor (CPU-bound)")
    time_process = benchmark_create_index(args.dataset, use_processes=True, max_workers=args.workers)

    # Summary
    if time_thread and time_process:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        speedup = time_thread / time_process
        faster = "ProcessPool" if speedup > 1 else "ThreadPool"
        print(f"ThreadPool: {time_thread:.2f}s")
        print(f"ProcessPool: {time_process:.2f}s")
        print(f"Faster: {faster} ({abs(speedup):.2f}x)")
        print("=" * 60)


if __name__ == "__main__":
    main()
