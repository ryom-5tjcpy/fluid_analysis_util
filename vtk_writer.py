import io

import numpy as np
from polars import DataFrame

def write_header(buffer: io.BytesIO, size: int, x: int, y: int, z: int, length: int):
    buffer.write(b"# vtk DataFile Version 3.0\n")
    buffer.write(b"Fluid Simulation Data\n")
    buffer.write(b"BINARY\n")
    buffer.write(b"DATASET STRUCTURED_POINTS\n")
    buffer.write(b"DIMENSIONS {size} {size} {size}\n".format(size=size))
    buffer.write(b"ORIGIN {x} {y} {z}\n".format(x=x, y=y, z=z))
    buffer.write(b"SPACING 1 1 1\n")
    buffer.write(b"POINT_DATA {length}\n".format(length=length))

def write_scalar_data(buffer: io.BytesIO, values: np.ndarray, name: str):
    """
    Write scalar data to a VTK buffer.

    Parameters:
        buffer (io.BytesIO): The buffer to write the data to.
        values (np.ndarray): The scalar values to write.
        name (str): The name of the scalar field.
    """
    buffer.write(f"SCALARS {name} float\n".encode("ascii"))
    buffer.write(f"LOOKUP_TABLE default\n".encode("ascii"))
    f_array = np.array(values, dtype='>f4').flatten()
    buffer.write(f_array.tobytes())

def write_vector_data(buffer: io.BytesIO, values: np.ndarray, name: str):
    """
    Write vector data to a VTK buffer.

    Parameters:
        buffer (io.BytesIO): The buffer to write the data to.
        values (np.ndarray): The vector values to write.
        name (str): The name of the vector field.
    """
    buffer.write(f"VECTORS {name} float\n".encode("ascii"))
    f_array = np.array(values, dtype='>f4').flatten()
    buffer.write(f_array.tobytes())

def save(filename: str, df: DataFrame, size: int, x: int, y: int, z: int, scalar_fields:list[str]=None, vector_fields:dict[str, list[str]]=None):
    buffer = io.BytesIO()
    write_header(buffer, size, x, y, z, len(df))

    if scalar_fields:
        for field in scalar_fields:
            write_scalar_data(buffer, df.select(field).to_numpy(), field)

    if vector_fields:
        for field_name, field_list in vector_fields.items():
            # Assuming field_list contains the column names for the vector components
            vector_values = np.column_stack([df.select(f).to_numpy().flatten() for f in field_list])
            write_vector_data(buffer, vector_values, field_name)

    with open(filename, "wb") as f:
        f.write(buffer.getvalue())