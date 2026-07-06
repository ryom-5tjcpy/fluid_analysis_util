from typing import BinaryIO

import numpy as np
from polars import DataFrame


def write_header(buffer: BinaryIO, size: int, x: int, y: int, z: int, length: int):
    buffer.write(b"# vtk DataFile Version 3.0\n")
    buffer.write(b"Fluid Simulation Data\n")
    buffer.write(b"BINARY\n")
    buffer.write(b"DATASET STRUCTURED_POINTS\n")
    buffer.write(f"DIMENSIONS {size} {size} {size}\n".encode("ascii"))
    buffer.write(f"ORIGIN {x} {y} {z}\n".encode("ascii"))
    buffer.write(b"SPACING 1 1 1\n")
    buffer.write(f"POINT_DATA {length}\n".encode("ascii"))


def write_scalar_data(buffer: BinaryIO, values: np.ndarray, name: str):
    """
    Write scalar data to a VTK buffer.

    Parameters:
        buffer (io.BytesIO): The buffer to write the data to.
        values (np.ndarray): The scalar values to write.
        name (str): The name of the scalar field.
    """
    buffer.write(f"SCALARS {name} float\n".encode("ascii"))
    buffer.write(f"LOOKUP_TABLE default\n".encode("ascii"))

    arr = np.asarray(values)
    arr = arr.ravel()
    # convert to big-endian float32 for VTK binary; allow numpy to avoid copy if possible
    be_arr = arr.astype('>f4', copy=False)
    buffer.write(be_arr.tobytes())

def write_vector_data(buffer: BinaryIO, values: np.ndarray, name: str):
    """
    Write vector data to a VTK buffer.

    Parameters:
        buffer (io.BytesIO): The buffer to write the data to.
        values (np.ndarray): The vector values to write.
        name (str): The name of the vector field.
    """
    buffer.write(f"VECTORS {name} float\n".encode("ascii"))

    arr = np.asarray(values)
    arr = arr.reshape(-1, arr.shape[-1])
    be_arr = arr.astype('>f4', copy=False)
    buffer.write(be_arr.ravel().tobytes())

def save(filename: str, df: DataFrame, size: int, x: int, y: int, z: int, scalar_fields:list[str]=None, vector_fields:dict[str, list[str]]=None):
    # Open file and write incrementally to avoid building entire output in memory
    length = len(df)
    with open(filename, "wb") as f:
        write_header(f, size, x, y, z, length)

        if scalar_fields:
            for field in scalar_fields:
                # df.select(field).to_numpy() returns shape (n,1) typically
                values = df.select(field).to_numpy()
                write_scalar_data(f, values, field)

        if vector_fields:
            for field_name, field_list in vector_fields.items():
                values = df.select(field_list).to_numpy()
                write_vector_data(f, values, field_name)