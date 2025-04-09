"""
xml_data_import.py

This module provides utility functions to load and process XML data stored in a MATLAB v7.3 MAT file.
It contains helper functions to:
  - Convert MATLAB mat_struct objects into Python dictionaries.
  - Decode MATLAB strings stored as bytes or numeric arrays.
  - Recursively print the HDF5 structure.
  - Load the "allData" variable from a MAT file and convert it into a list of dictionaries.
  - Validate and describe the structure of each XML file's data.
  - Access specific file data and target individual fields or slices from numeric arrays.
  - Convert a list of dictionaries into a dictionary of lists (for convenient MAT saving).
  - Save the processed data structure back to a MAT file.
"""

import os
import glob
import numpy as np
from xml.dom import minidom
from scipy.io import loadmat, savemat


def matstruct_to_dict(mat_struct: object) -> object:
    """
    Recursively converts a MATLAB mat_struct object into a Python dictionary.
    Handles nested structures and lists.

    Parameters:
        mat_struct: MATLAB structure object (often of type mat_struct).

    Returns:
        A Python dict representing the structure, or the original object if no _fieldnames.
    """
    if not hasattr(mat_struct, "_fieldnames"):
        return mat_struct
    out_dict = {}
    for field in mat_struct._fieldnames:
        value = getattr(mat_struct, field)
        if hasattr(value, "_fieldnames"):
            out_dict[field] = matstruct_to_dict(value)
        elif isinstance(value, list):
            out_dict[field] = [matstruct_to_dict(item) if hasattr(item, "_fieldnames") else item for item in value]
        else:
            out_dict[field] = value
    return out_dict


def read_mat_string(ref, f) -> str:
    """
    Given an h5py reference and the open file object f, retrieve the dataset,
    extract its data, and decode it to a UTF-8 string.
    Works for MATLAB strings stored as bytes or as numeric arrays.

    Parameters:
        ref: h5py reference to the dataset.
        f: Open h5py File object.

    Returns:
        The decoded string.
    """
    ds = f[ref]
    data = ds[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, np.ndarray):
        try:
            return data.tobytes().decode("utf-8").strip()
        except Exception as e:
            return str(data)
    else:
        return str(data)


def recursive_print(name: str, obj, f, indent: int = 0) -> None:
    """
    Recursively prints the structure of an HDF5 object.

    Parameters:
        name: The current object's name.
        obj: HDF5 object (Group or Dataset).
        f: Open h5py File object.
        indent: Current indentation level.
    """
    spacer = "  " * indent
    if isinstance(obj, np.ndarray):
        # If it's an array, just print its shape and type.
        print(f"{spacer}{name} (Array): shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{spacer}{name} (Group)")
        for key, item in obj.items():
            recursive_print(key, item, f, indent + 1)
    else:
        print(f"{spacer}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")


def load_xml_mat(mat_file: str) -> list:
    """
    Loads the MATLAB v7.3 MAT file containing XML data and returns the 'allData'
    variable as a list of dictionaries.

    Parameters:
        mat_file: Path to the MAT file.

    Returns:
        List of dictionaries where each dictionary represents one XML file's data.

    Raises:
        FileNotFoundError: If the MAT file is not found.
        ValueError: If 'allData' is not found in the MAT file.
    """
    if not os.path.exists(mat_file):
        raise FileNotFoundError("MAT file not found at: " + mat_file)
    mat_contents = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    if "allData" not in mat_contents:
        raise ValueError("'allData' variable not found in the MAT file.")
    all_data = mat_contents["allData"]
    if hasattr(all_data, "tolist"):
        all_data = all_data.tolist()
    converted = [matstruct_to_dict(elem) for elem in all_data]
    return converted


def describe_xml_data_structure(all_data: list) -> None:
    """
    Prints a summary description for each element in the allData list.
    Each element is expected to have the following fields:
      - fileName, events, stages, epochLength, annotation.

    Also prints a summary header similar to the EDF output:
      - Top-level keys and placeholder MATLAB attributes.
      - List of fields present in allData.
      - Decoded file names.

    Parameters:
        all_data: List of dictionaries representing XML file data.
    """
    # Print header similar to your EDF output.
    print("Keys in MAT file:")
    print(" - allData\n")
    print("Structure of 'allData':")
    print("  Attribute MATLAB_class: b'struct'")
    print("  Attribute MATLAB_fields: [array([...], dtype='|S1'), array([...], dtype='|S1'), array([...], dtype='|S1')]\n")
    print("Fields in allData:")
    if all_data:
        for field in all_data[0].keys():
            print(f"  Field: {field}")
    else:
        print("  (allData is empty)")
    print("\nDecoded file names from 'allData':")
    for idx, elem in enumerate(all_data):
        file_name = elem.get("fileName", "N/A")
        print(f"  {idx + 1}: {file_name}")
    print("\nValidating structure for {} elements:\n".format(len(all_data)))
    expected_fields = ["fileName", "events", "stages", "epochLength", "annotation"]
    for idx, elem in enumerate(all_data):
        print("Element {}:".format(idx + 1))
        if not isinstance(elem, dict):
            print(f"  (Non-dict element of type {type(elem).__name__}): {elem}\n")
            continue
        for field in expected_fields:
            value = elem.get(field, None)
            if field == "fileName":
                print(f"  {field}: {value} (type: {type(value).__name__})")
            elif field == "events":
                try:
                    num_events = len(value) if value is not None else "N/A"
                except Exception:
                    num_events = "N/A"
                print(f"  {field}: {num_events} events (type: {type(value).__name__})")
            elif field == "stages":
                try:
                    stages_len = len(value) if value is not None else "N/A"
                except Exception:
                    stages_len = "N/A"
                print(f"  {field}: length {stages_len} (type: {type(value).__name__})")
            elif field in ["epochLength", "annotation"]:
                print(f"  {field}: {value} (type: {type(value).__name__})")
        print("")


def get_xml_file_data(all_data: list, file_index: int) -> dict:
    """
    Returns the data for the XML file at the given index (0-based).

    Parameters:
        all_data: List of dictionaries for XML files.
        file_index: 0-based index of the desired XML file.

    Returns:
        Dictionary for the selected XML file.

    Raises:
        IndexError: If file_index is out of bounds.
    """
    if file_index < 0 or file_index >= len(all_data):
        raise IndexError("File index out of bounds.")
    return all_data[file_index]


def get_field_value(xml_file_data: dict, field_name: str):
    """
    Retrieves the value for a given field from a single XML file's data.

    Parameters:
        xml_file_data: Dictionary representing one XML file.
        field_name: Key name to retrieve (e.g., 'fileName', 'stages').

    Returns:
        The value associated with the field, or None if not present.
    """
    return xml_file_data.get(field_name, None)


def slice_matrix(matrix: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Returns a slice from a numeric matrix along the specified axis.
    Uses np.take to extract a row, column, or any other slice from the array.

    Parameters:
        matrix: Input numpy ndarray.
        axis: Axis along which to slice.
        index: Index along the specified axis.

    Returns:
        The extracted slice as a numpy ndarray.

    Raises:
        TypeError: If matrix is not a numpy ndarray.
        IndexError: If index is out of range.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if index < 0 or index >= matrix.shape[axis]:
        raise IndexError("Index out of range for the given axis.")
    return np.take(matrix, index, axis=axis)


def list_of_dicts_to_dict_of_lists(ld: list) -> dict:
    """
    Converts a list of dictionaries (each representing an XML file's data)
    into a dictionary of lists. This format is more friendly for savemat.

    Parameters:
        ld: List of dictionaries.

    Returns:
        A dictionary where keys are the field names and values are lists of field values.
    """
    if not ld:
        return {}
    keys = ld[0].keys()
    result = {key: [] for key in keys}
    for d in ld:
        for key in keys:
            result[key].append(d.get(key, None))
    return result


def save_to_mat(all_data: list, output_file: str) -> None:
    """
    Saves the provided allData structure (list of dictionaries) into a MATLAB MAT file.
    Converts the list of dictionaries into a dictionary of lists so that savemat saves
    it as a structured array.

    Parameters:
        all_data: List of dictionaries containing XML file data.
        output_file: Path to the output MAT file.
    """
    converted = list_of_dicts_to_dict_of_lists(all_data)
    savemat(output_file, {"allData": converted})
    print("Saved MAT file to:", output_file)


# Example usage when running this module directly:
if __name__ == "__main__":
    # Update the path as needed (ensure it's a MATLAB v7.3 file).
    mat_file = "/Users/markus/university/signal-processing-CM2013-ss/data/XML_RawData.mat"
    all_data = load_xml_mat(mat_file)
    describe_xml_data_structure(all_data)

    # Retrieve and print data for the first XML file.
    #first_file = get_xml_file_data(all_data, 0)
    #print("\nData for first XML file:")
    #for key, value in first_file.items():
    #    print(f"  {key}: {value}")

    # If the 'stages' field is numeric and multidimensional, extract a slice.
    #stages = get_field_value(first_file, "stages")
    #if isinstance(stages, np.ndarray) and stages.ndim >= 2:
    #    row0 = slice_matrix(stages, axis=0, index=0)
    #    print("\nFirst row of stages from the first XML file:")
    #    print(row0)
    #else:
    #    print("\nThe 'stages' field is not a multidimensional numpy array.")