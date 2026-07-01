import glob
import polars as pl

class SimulationDataLoader:
    """
    Load and query simulation data stored as Arrow IPC files.
    """
    def __init__(self, pattern: str):
        """
        Initialize the SimulationDataLoader with a glob pattern.

        Parameters:
            pattern (str): Glob pattern matching one or more Arrow IPC files.
        """
        self.files = glob.glob(pattern)

    def query(self, columns=None, conditions=None, sort_order=None, descending=False, limit=None):
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
    
    def fetch_cube(self, x, y, z, size, columns=None, sort_order=None, descending=False, limit=None):
        """
        Fetch a cube of simulation data based on its position and size.

        Parameters:
            x (int): X-coordinate of the cube's origin.
            y (int): Y-coordinate of the cube's origin.
            z (int): Z-coordinate of the cube's origin.
            size (int): Size of the cube in each dimension.
            columns (list[str] | None): List of column names to select.
            sort_order (str | list[str] | None): Column name(s) to sort by.
            descending (bool): Whether to sort in descending order.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pl.DataFrame: The fetched cube of simulation data.
        """
        conditions = [
            (pl.col("x") >= x) & (pl.col("x") < x + size),
            (pl.col("y") >= y) & (pl.col("y") < y + size),
            (pl.col("z") >= z) & (pl.col("z") < z + size)
        ]
        return self.query(columns=columns, conditions=conditions, sort_order=sort_order, descending=descending, limit=limit)
