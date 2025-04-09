import h5py
import numpy as np


def read_mat_string(ref, f):
    """
    Given an h5py reference and the open file f,
    retrieve the dataset, extract its data, and decode it to a UTF-8 string.
    This works for MATLAB strings stored as arrays of bytes or as numeric arrays.
    """
    ds = f[ref]
    data = ds[()]
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, np.ndarray):
        # Convert numeric array (e.g., uint16) to bytes then decode.
        try:
            return data.tobytes().decode('utf-8').strip()
        except Exception as e:
            return str(data)
    else:
        return str(data)


def recursive_print(name, obj, f, indent=0):
    """
    Recursively print the structure of an HDF5 object.
    """
    spacer = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{spacer}{name} (Group)")
        for key, item in obj.items():
            recursive_print(key, item, f, indent + 1)
    else:
        print(f"{spacer}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")


def describe_edf_mat_structure(mat_file):
    """
    Loads a MATLAB v7.3 MAT file containing EDF raw data and prints a description of its structure.

    Expected contents:
      - 'allData': A structure array with fields:
            • fileName : EDF file name (string)
            • hdr      : Header information from edfread (structure with metadata)
            • record   : Raw EDF data (numeric matrix)
      - 'bigMatrix' (optional): A 3D matrix with dimensions [channels x samples x files].

    This function prints:
      - Top-level keys in the MAT file.
      - For the 'allData' group: its attributes and fields.
      - Decoded file names from the 'fileName' dataset.
    """
    with h5py.File(mat_file, 'r') as f:
        print("Keys in MAT file:")
        for key in f.keys():
            print(" -", key)

        if 'allData' in f:
            print("\nStructure of 'allData':")
            allData = f['allData']
            # Print attributes of the allData group.
            for attr_key, attr_val in allData.attrs.items():
                print(f"  Attribute {attr_key}: {attr_val}")
            # List the fields in allData.
            for field in allData.keys():
                print(f"  Field: {field}")

            # Process the fileName field.
            if 'fileName' in allData:
                file_names_ds = allData['fileName']
                file_names_list = []
                # file_names_ds is expected to be of shape (N,1) with each element a reference.
                for i in range(file_names_ds.shape[0]):
                    ref = file_names_ds[i, 0]
                    name_str = read_mat_string(ref, f)
                    file_names_list.append(name_str)
                print("\nDecoded file names from 'allData':")
                for idx, name in enumerate(file_names_list):
                    print(f"  {idx + 1}: {name}")
            else:
                print("'fileName' field not found in 'allData'.")
        else:
            print("'allData' not found in the MAT file.")

        if 'bigMatrix' in f:
            print("\nStructure of 'bigMatrix':")
            bigMatrix = f['bigMatrix']
            print(f"bigMatrix: shape={bigMatrix.shape}, dtype={bigMatrix.dtype}")
        else:
            pass
            #print("'bigMatrix' not found in the MAT file.")


if __name__ == "__main__":
    # Update the path to your MAT file if needed.
    mat_file = "EDF_RawData.mat"
    describe_edf_mat_structure(mat_file)