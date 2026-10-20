import math

import polars as pl
import pytest

from velocity_gradient import compute_boxcell_velocity_gradient


def make_linear_boxcell(size: int = 4) -> pl.DataFrame:
    coordinates = [
        (x, y, z)
        for z in range(10, 10 + size)
        for y in range(20, 20 + size)
        for x in range(30, 30 + size)
    ]
    return pl.DataFrame(
        {
            "gnx": [x for x, _, _ in coordinates],
            "gny": [y for _, y, _ in coordinates],
            "nn": [z for _, _, z in coordinates],
            "u": [2 * x + 3 * y - z for x, y, z in coordinates],
            "v": [-x + 4 * y + 2 * z for x, y, z in coordinates],
            "w": [5 * x - 2 * y + 3 * z for x, y, z in coordinates],
        }
    )


def test_computes_each_velocity_derivative_from_face_centers() -> None:
    size = 4
    grid_size = 128
    spacing = 2 * math.pi * (size - 1) / grid_size

    result = compute_boxcell_velocity_gradient(
        make_linear_boxcell(size),
        size=size,
        grid_size=grid_size,
    )

    assert result.height == 1
    assert result.select("gnx", "gny", "nn").row(0) == (30, 20, 10)
    actual = result.select(
        "u_x_grad", "u_y_grad", "u_z_grad",
        "v_x_grad", "v_y_grad", "v_z_grad",
        "w_x_grad", "w_y_grad", "w_z_grad",
    ).row(0)
    assert actual == pytest.approx(
        tuple(value / spacing for value in (2, 3, -1, -1, 4, 2, 5, -2, 3)),
        abs=1e-12,
    )

    tensor_values = result.select(
        "s_xx", "s_xy", "s_xz",
        "s_yx", "s_yy", "s_yz",
        "s_zx", "s_zy", "s_zz",
        "omega_xx", "omega_xy", "omega_xz",
        "omega_yx", "omega_yy", "omega_yz",
        "omega_zx", "omega_zy", "omega_zz",
        "vorticity_x", "vorticity_y", "vorticity_z",
    ).row(0)
    assert tensor_values == pytest.approx(
        tuple(value / spacing for value in (
            2, 1, 2,
            1, 4, 0,
            2, 0, 3,
            0, 2, -3,
            -2, 0, 2,
            3, -2, 0,
            -4, -6, -4,
        )),
        abs=1e-12,
    )


def test_rejects_boxcell_without_a_required_face_sample() -> None:
    boxcell = make_linear_boxcell().filter(
        ~((pl.col("gnx") == 30) & (pl.col("gny") == 21) & (pl.col("nn") == 21))
    )

    with pytest.raises(ValueError, match="missing face-center samples"):
        compute_boxcell_velocity_gradient(boxcell, size=4, grid_size=128)


def test_requires_even_size() -> None:
    with pytest.raises(ValueError, match="even integer"):
        compute_boxcell_velocity_gradient(make_linear_boxcell(), size=3, grid_size=128)
