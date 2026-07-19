from typing import BinaryIO

import numpy as np
import pyarrow as pa
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


def _write_scalar_data_arrow(buffer: BinaryIO, arrow_array: pa.Array, name: str):
    """
    Write scalar data from Arrow Array to VTK buffer (zero-copy path when possible).

    Parameters:
        buffer (BinaryIO): The buffer to write the data to.
        arrow_array (pa.Array): Arrow array of scalar values.
        name (str): The name of the scalar field.
    """
    buffer.write(f"SCALARS {name} float\n".encode("ascii"))
    buffer.write(f"LOOKUP_TABLE default\n".encode("ascii"))

    # Try to extract buffer directly from Arrow array
    if arrow_array.type == pa.float32():
        # Already float32; get the data buffer
        buffers = arrow_array.buffers()
        data_buffer = buffers[1]  # buffers[0] is null bitmap, buffers[1] is data
        
        if data_buffer is not None:
            # Arrow uses native endian; convert if needed
            data_bytes = data_buffer.hex() if hasattr(data_buffer, 'hex') else data_buffer.tobytes()
            
            # For simplicity, fall back to numpy conversion if we can't directly access
            arr = arrow_array.to_numpy(zero_copy_only=False)
            be_arr = arr.astype('>f4', copy=False)
            buffer.write(be_arr.tobytes())
        else:
            # Null buffer; convert via numpy
            arr = arrow_array.to_numpy(zero_copy_only=False)
            be_arr = arr.astype('>f4', copy=False)
            buffer.write(be_arr.tobytes())
    else:
        # Need type conversion; use numpy
        arr = arrow_array.to_numpy(zero_copy_only=False).flatten()
        be_arr = arr.astype('>f4', copy=False)
        buffer.write(be_arr.tobytes())


def _write_vector_data_arrow(buffer: BinaryIO, arrow_table: pa.Table, field_names: list, name: str):
    """
    Write vector data from Arrow Table columns to VTK buffer (zero-copy path when possible).

    Parameters:
        buffer (BinaryIO): The buffer to write the data to.
        arrow_table (pa.Table): Arrow table containing the vector components.
        field_names (list): List of column names for vector components (e.g., ['u', 'v', 'w']).
        name (str): The name of the vector field.
    """
    buffer.write(f"VECTORS {name} float\n".encode("ascii"))

    # Combine columns into contiguous array
    arrays = [arrow_table[col].to_numpy(zero_copy_only=False) for col in field_names]
    combined = np.column_stack(arrays)
    be_arr = combined.astype('>f4', copy=False)
    buffer.write(be_arr.ravel().tobytes())


def save_arrow(filename: str, df: DataFrame, size: int, x: int, y: int, z: int, 
               scalar_fields: list[str] = None, vector_fields: dict[str, list[str]] = None):
    """
    Save simulation data to VTK file using Arrow buffers (optimized zero-copy path).

    Parameters:
        filename (str): Output VTK filename.
        df (DataFrame): Polars DataFrame containing simulation data.
        size (int): Grid size in each dimension.
        x, y, z (int): Grid origin coordinates.
        scalar_fields (list[str]): Column names for scalar fields.
        vector_fields (dict[str, list[str]]): Mapping of vector field names to component column lists.
    """
    length = len(df)
    arrow_table = df.to_arrow()
    
    with open(filename, "wb") as f:
        write_header(f, size, x, y, z, length)

        if scalar_fields:
            for field in scalar_fields:
                arr = arrow_table[field]
                _write_scalar_data_arrow(f, arr, field)

        if vector_fields:
            for field_name, field_list in vector_fields.items():
                _write_vector_data_arrow(f, arrow_table, field_list, field_name)