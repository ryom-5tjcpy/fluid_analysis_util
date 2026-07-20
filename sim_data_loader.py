import json
import glob
import polars as pl
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from polars import col
import os
from collections import defaultdict


class SimulationDataLoader:
    """
    Load and query simulation data stored as Arrow IPC files.
    Uses spatial indexing for fast range queries on file coverage.
    """
    
    # Grid cell size for spatial index (tunable parameter)
    GRID_CELL_SIZE = 100
    
    def __init__(self, pattern: str, index_file: str = "file_index.json"):
        """
        Initialize the SimulationDataLoader with a glob pattern.

        Parameters:
            pattern (str): Glob pattern matching one or more Arrow IPC files.
            index_file (str): Path to the file index JSON file.
        """
        self.pattern = pattern
        self.index_file = index_file
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise FileNotFoundError(
                f"No IPC files found for pattern: {pattern}. "
                "Please verify the dataset key and base path."
            )
        
        index_path = Path(index_file)

        try:
            if index_path.exists():
                with open(index_path, 'r') as f:
                    self.file_index = json.load(f)

                need_reindex = len(self.file_index) != len(self.files)
            else:
                need_reindex = True
        except Exception as e:
            print(f"Error occurred while loading index: {e}")
            need_reindex = True

        if need_reindex:
            self.create_index(100)
        
        # Build spatial grid index for fast range queries
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build a grid-based spatial index for fast file lookups."""
        self.spatial_grid = defaultdict(list)  # (grid_x, grid_y) -> [file_indices]
        
        for file_idx, file_info in enumerate(self.file_index):
            gnx_min, gnx_max = file_info['gnx_range']
            gny_min, gny_max = file_info['gny_range']
            
            # Compute grid cells covered by this file
            grid_x_min = gnx_min // self.GRID_CELL_SIZE
            grid_x_max = gnx_max // self.GRID_CELL_SIZE
            grid_y_min = gny_min // self.GRID_CELL_SIZE
            grid_y_max = gny_max // self.GRID_CELL_SIZE
            
            # Insert file index into all covered grid cells
            for gx in range(grid_x_min, grid_x_max + 1):
                for gy in range(grid_y_min, grid_y_max + 1):
                    self.spatial_grid[(gx, gy)].append(file_idx)

    def create_index(self, max_workers: int = None, use_processes: bool = False):
        """Create an index of files and their spatial ranges.
        
        Parameters:
            max_workers (int | None): Max number of parallel workers. If None, auto-tune based on CPU count.
            use_processes (bool): If True, use ProcessPoolExecutor (better for CPU-bound); else ThreadPoolExecutor (I/O-bound).
        """
        if max_workers is None:
            # Auto-tune: use CPU count for processes, 2x for threads (I/O wait tolerance)
            cpu_count = os.cpu_count() or 4
            max_workers = cpu_count if use_processes else min(cpu_count * 2, 16)
        
        def process_single_file(file_path):
            try:
                # Only scan gnx, gny columns needed for indexing (skip others)
                df = pl.scan_ipc(file_path).select(['gnx', 'gny'])

                stats = df.select([
                    col('gnx').min().alias('min_gnx'),
                    col('gnx').max().alias('max_gnx'),
                    col('gny').min().alias('min_gny'),
                    col('gny').max().alias('max_gny'),
                    pl.len().alias('row_count')
                ]).collect(streaming=True)

                row = stats.row(0, named=True)

                return {
                    'file': file_path,
                    'gnx_range': (row['min_gnx'], row['max_gnx']),
                    'gny_range': (row['min_gny'], row['max_gny']),
                    'row_count': row['row_count']
                }
            except Exception as e:
                print(f"Error occurred while scanning {file_path}: {e}")
                return None
            
        self.file_index = []

        # Choose executor based on use_processes flag
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, file_path): file_path for file_path in self.files}
            for future in futures:
                result = future.result()
                if result:
                    self.file_index.append(result)

        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.file_index, f)
        except Exception as e:
            print(f"Error occurred while saving index: {e}")
        
        # Rebuild spatial index after creating/updating file_index
        self._build_spatial_index()

    def find_files(self, gnx_range, gny_range):
        """
        Find files that intersect with the given ranges using spatial grid index.
        
        Uses grid-based acceleration to avoid O(n) linear scan of file_index.

        Parameters:
            gnx_range (tuple[int, int]): Range for gnx column.
            gny_range (tuple[int, int]): Range for gny column.

        Returns:
            list[str]: List of file paths that intersect with the given ranges.
        """
        # Compute grid cells that overlap with query range
        grid_x_min = int(gnx_range[0] // self.GRID_CELL_SIZE)
        grid_x_max = int((gnx_range[1] - 1) // self.GRID_CELL_SIZE)
        grid_y_min = int(gny_range[0] // self.GRID_CELL_SIZE)
        grid_y_max = int((gny_range[1] - 1) // self.GRID_CELL_SIZE)

        # Collect candidate file indices from grid cells
        candidate_file_indices = set()
        for gx in range(grid_x_min, grid_x_max + 1):
            for gy in range(grid_y_min, grid_y_max + 1):
                candidate_file_indices.update(self.spatial_grid.get((gx, gy), []))
        
        # Verify candidates actually intersect (range overlap check)
        target_files = []
        for file_idx in candidate_file_indices:
            file_info = self.file_index[file_idx]
            gnx_min, gnx_max = file_info['gnx_range']
            gny_min, gny_max = file_info['gny_range']

            if (gnx_min < gnx_range[1] and gnx_max > gnx_range[0] and
                gny_min < gny_range[1] and gny_max > gny_range[0]):
                target_files.append(file_info['file'])

        return target_files

    def query(self, target_files=None, columns=None, conditions=None, sort_order=None, descending=False, limit=None, streaming=False) -> pl.DataFrame:
        """
        Query the simulation data with specified options.

        Parameters:
            target_files (list[str] | None): List of file paths to query.
            columns (list[str] | None): List of column names to select. Recommended to specify to avoid reading unnecessary columns.
            conditions (list[pl.Expr] | None): List of Polars boolean expressions for filtering.
            sort_order (str | list[str] | None): Column name(s) to sort by.
            descending (bool): Whether to sort in descending order.
            limit (int | None): Maximum number of rows to return.
            streaming (bool): Use streaming execution for better memory efficiency on large datasets.

        Returns:
            pl.DataFrame: Collected result of the query.
        """
        target_files = target_files or self.files

        df = pl.scan_ipc(target_files)

        if conditions:
            df = df.filter(pl.all_horizontal(conditions))
            
        if columns:
            df = df.select(columns)

        if sort_order:
            df = df.sort(sort_order, descending=descending)

        if limit is not None:
            df = df.head(limit)

        return df.collect(streaming=streaming)
    
    def fetch_boxcell(self, x, y, z, size, columns=None, sort_order=None, descending=False, limit=None, streaming=False) -> pl.DataFrame:
        """
        Fetch a boxcell of simulation data based on its position and size.

        Parameters:
            x (int): X-coordinate of the boxcell's origin.
            y (int): Y-coordinate of the boxcell's origin.
            z (int): Z-coordinate of the boxcell's origin.
            size (int): Size of the boxcell in each dimension.
            columns (list[str] | None): List of column names to select.
            sort_order (str | list[str] | None): Column name(s) to sort by.
            descending (bool): Whether to sort in descending order.
            limit (int | None): Maximum number of rows to return.
            streaming (bool): Use streaming execution for better memory efficiency.

        Returns:
            pl.DataFrame: The fetched boxcell of simulation data.
        """
        target_files = self.find_files(
            gnx_range=(x, x + size),
            gny_range=(y, y + size),
        )

        conditions = [
            (col("gnx") >= x) & (col("gnx") < x + size),
            (col("gny") >= y) & (col("gny") < y + size),
            (col("nn") >= z) & (col("nn") < z + size)
        ]
        return self.query(target_files, columns, conditions, sort_order, descending, limit, streaming=streaming)
