import json
import glob
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from polars import col


class SimulationDataLoader:
    """
    Load and query simulation data stored as Arrow IPC files.
    """
    def __init__(self, pattern: str, index_file: str = "file_index.json"):
        """
        Initialize the SimulationDataLoader with a glob pattern.

        Parameters:
            pattern (str): Glob pattern matching one or more Arrow IPC files.
            index_file (str): Path to the file index JSON file.
        """
        self.pattern = pattern
        self.index_file = index_file
        self.files = glob.glob(pattern)

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
        except Exception as e:
            print(f"Error occurred while loading index: {e}")
            need_reindex = True

        if need_reindex:
            self.create_index(50)

    def create_index(self, max_workers: int = 4):
        def process_single_file(file_path):
            try:
                df = pl.scan_ipc(file_path)

                stats = df.select([
                    col('gnx').min().alias('min_gnx'),
                    col('gnx').max().alias('max_gnx'),
                    col('gny').min().alias('min_gny'),
                    col('gny').max().alias('max_gny'),
                    col('nn').min().alias('min_nn'),
                    col('nn').max().alias('max_nn'),
                    pl.len().alias('row_count')
                ]).collect()

                row = stats.row(0, named=True)

                return {
                    'file': file_path,
                    'gnx_range': (row['min_gnx'], row['max_gnx']),
                    'gny_range': (row['min_gny'], row['max_gny']),
                    'nn_range': (row['min_nn'], row['max_nn']),
                    'row_count': row['row_count']
                }
            except Exception as e:
                print(f"Error occurred while scanning {file_path}: {e}")
                return None
            
        self.file_index = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    def query(self, columns=None, conditions=None, sort_order=None, descending=False, limit=None) -> pl.DataFrame:
        """
        Query the simulation data with specified options.

        Parameters:
            columns (list[str] | None): List of column names to select.
            conditions (list[pl.Expr] | None): List of Polars boolean expressions for filtering.
            sort_order (str | list[str] | None): Column name(s) to sort by.
            descending (bool): Whether to sort in descending order.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pl.DataFrame: Collected result of the query.
        """
        target_files = self.files

        df = pl.scan_ipc(target_files)

        if columns:
            df = df.select(columns)

        if conditions:
            df = df.filter(pl.all_horizontal(conditions))

        if sort_order:
            df = df.sort(sort_order, descending=descending)

        if limit is not None:
            df = df.head(limit)

        return df.collect()
    
    def fetch_boxcell(self, x, y, z, size, columns=None, sort_order=None, descending=False, limit=None) -> pl.DataFrame:
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

        Returns:
            pl.DataFrame: The fetched boxcell of simulation data.
        """
        conditions = [
            (col("gnx") >= x) & (col("gnx") < x + size),
            (col("gny") >= y) & (col("gny") < y + size),
            (col("nn") >= z) & (col("nn") < z + size)
        ]
        return self.query(columns=columns, conditions=conditions, sort_order=sort_order, descending=descending, limit=limit)
