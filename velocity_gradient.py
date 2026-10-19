import math

import polars as pl


def compute_boxcell_velocity_gradient(
    boxcell: pl.DataFrame,
    size: int,
    grid_size: int,
) -> pl.DataFrame:
    """Compute one velocity-gradient tensor from the six face-center samples."""
    if size < 4 or size % 2 != 0:
        raise ValueError("size must be an even integer greater than or equal to 4")
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than 0")
    if boxcell.is_empty():
        raise ValueError("boxcell must contain data")

    required_columns = {"gnx", "gny", "nn", "u", "v", "w"}
    missing_columns = required_columns - set(boxcell.columns)
    if missing_columns:
        raise ValueError(
            f"boxcell is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    x_min = boxcell["gnx"].min()
    y_min = boxcell["gny"].min()
    z_min = boxcell["nn"].min()
    x_max = boxcell["gnx"].max()
    y_max = boxcell["gny"].max()
    z_max = boxcell["nn"].max()
    if (
        x_max - x_min != size - 1
        or y_max - y_min != size - 1
        or z_max - z_min != size - 1
    ):
        raise ValueError("boxcell coordinate extents do not match size")

    middle = size // 2 - 1
    x_mid = x_min + middle
    y_mid = y_min + middle
    z_mid = z_min + middle
    sample_coordinates = {
        (x_min, y_mid, z_mid),
        (x_max, y_mid, z_mid),
        (x_mid, y_min, z_mid),
        (x_mid, y_max, z_mid),
        (x_mid, y_mid, z_min),
        (x_mid, y_mid, z_max),
    }

    sample_filter = pl.any_horizontal(
        [
            (pl.col("gnx") == x) & (pl.col("gny") == y) & (pl.col("nn") == z)
            for x, y, z in sample_coordinates
        ]
    )
    samples_df = boxcell.filter(sample_filter).select("gnx", "gny", "nn", "u", "v", "w")

    samples: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for row in samples_df.iter_rows(named=True):
        coordinate = (row["gnx"], row["gny"], row["nn"])
        if coordinate in samples:
            raise ValueError(f"duplicate face-center sample at {coordinate}")
        values = (row["u"], row["v"], row["w"])
        if any(value is None for value in values):
            raise ValueError(f"null velocity value at face-center sample {coordinate}")
        samples[coordinate] = tuple(float(value) for value in values)

    if samples.keys() != sample_coordinates:
        missing_samples = sample_coordinates - samples.keys()
        raise ValueError(f"boxcell is missing face-center samples: {sorted(missing_samples)}")

    spacing = 2 * math.pi * (size - 1) / grid_size
    axis_points = {
        "x": ((x_min, y_mid, z_mid), (x_max, y_mid, z_mid)),
        "y": ((x_mid, y_min, z_mid), (x_mid, y_max, z_mid)),
        "z": ((x_mid, y_mid, z_min), (x_mid, y_mid, z_max)),
    }
    result: dict[str, int | float] = {
        "gnx": x_min,
        "gny": y_min,
        "nn": z_min,
    }
    for component_index, velocity in enumerate(("u", "v", "w")):
        for axis, (low_coordinate, high_coordinate) in axis_points.items():
            low_value = samples[low_coordinate][component_index]
            high_value = samples[high_coordinate][component_index]
            result[f"{velocity}_{axis}_grad"] = (high_value - low_value) / spacing

    gradient = {
        (velocity, axis): result[f"{velocity}_{axis}_grad"]
        for velocity in ("u", "v", "w")
        for axis in ("x", "y", "z")
    }
    components = ("x", "y", "z")
    velocity_components = ("u", "v", "w")
    for row_index, row_axis in enumerate(components):
        for column_index, column_axis in enumerate(components):
            forward = gradient[(velocity_components[row_index], column_axis)]
            reverse = gradient[(velocity_components[column_index], row_axis)]
            result[f"s_{row_axis}{column_axis}"] = 0.5 * (forward + reverse)
            result[f"omega_{row_axis}{column_axis}"] = 0.5 * (forward - reverse)

    result["vorticity_x"] = gradient[("w", "y")] - gradient[("v", "z")]
    result["vorticity_y"] = gradient[("u", "z")] - gradient[("w", "x")]
    result["vorticity_z"] = gradient[("v", "x")] - gradient[("u", "y")]

    return pl.DataFrame([result])
