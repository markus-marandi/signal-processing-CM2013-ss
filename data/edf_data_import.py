# edf_data_import.py

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

            # Process the fileName field if available.
            if 'fileName' in allData:
                file_names_ds = allData['fileName']
                file_names_list = []
                # file_names_ds is expected to be of shape (N, 1) with each element a reference.
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
            print("'bigMatrix' not found in the MAT file.")


def extract_eeg_signal(mat_file, edf_index=0, channel=0):
    """
    Extracts the EEG signal from the specified EDF file within the MAT file.

    This updated function assumes that:
      - The MAT file contains a group 'allData'.
      - All EDF records are stored in the dataset 'record' within the 'allData' group.
      - 'allData/record' is a dataset of references, one per EDF file.
      - Each reference points to a subgroup (or dataset) that holds a 2D array (channels x samples).

    Parameters:
      mat_file : path to the MAT file.
      edf_index: index of the EDF file to use (default is the first one, index 0).
      channel  : channel index to extract (for example, 0 for the first EEG channel).

    Returns:
      signal   : the extracted EEG signal (numpy array).
    """
    with h5py.File(mat_file, 'r') as f:
        # Check that 'allData' group exists
        if 'allData' not in f:
            raise KeyError("The MAT file does not contain an 'allData' group.")
        allData = f['allData']

        # Get the 'record' dataset that holds references to each EDF's record.
        if 'record' not in allData:
            raise KeyError("The 'allData' group does not contain a 'record' dataset.")
        record_ds = allData['record']

        # Check that the dataset has enough entries for the requested index.
        if edf_index >= record_ds.shape[0]:
            raise IndexError("edf_index is out of bounds.")

        # Get the reference for the selected EDF file's record.
        # Note: The record dataset is stored as a 2D array (nEDF x 1)
        ref = record_ds[edf_index, 0]
        # Dereference to get the actual data.
        rec = f[ref][()]

        # Verify that the record is 2D (channels x samples)
        if rec.ndim != 2:
            raise ValueError("Expected 'record' dataset to be 2D (channels x samples).")

        # Ensure the requested channel index is valid.
        if channel >= rec.shape[0]:
            raise IndexError("Channel index out of bounds for this EDF file.")

        # Extract and return the channel.
        signal = rec[channel, :]
    return signal


def compute_signal_features(signal):
    """
    Given a 1D numpy array signal, compute various signal features:
      - Mean
      - Variance
      - Root Mean Square (RMS)
      - Peak-to-Peak (Max - Min)
      - Zero-crossings (approximate count)
      - Hjorth parameters: Activity, Mobility, Complexity
    Returns:
      A dictionary with the calculated features.
    """
    features = {}
    # Basic statistics
    features['mean'] = np.mean(signal)
    features['variance'] = np.var(signal)
    features['rms'] = np.sqrt(np.mean(signal ** 2))
    features['peak_to_peak'] = np.ptp(signal)

    # Zero crossings: Count the number of times the signal changes sign.
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossings'] = len(zero_crossings)

    # Hjorth parameters
    # Activity is simply the variance of the signal.
    activity = features['variance']

    # First derivative (approximate)
    diff_signal = np.diff(signal)
    var_diff = np.var(diff_signal)
    # Mobility: standard deviation of the derivative / standard deviation of the signal.
    mobility = np.sqrt(var_diff / features['variance']) if features['variance'] > 0 else 0

    # Second derivative (approximate)
    diff2_signal = np.diff(diff_signal)
    var_diff2 = np.var(diff2_signal)
    mobility_diff = np.sqrt(var_diff2 / var_diff) if var_diff > 0 else 0

    # Complexity: ratio of the mobility of the derivative to the mobility of the signal.
    complexity = mobility_diff / mobility if mobility > 0 else 0

    features['hjorth_activity'] = activity
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity

    return features


def extract_all_eeg_signals(mat_file, channel=0):
    """
    Extracts the EEG signal (a specific channel) from all EDF files in the MAT file.

    Parameters:
      mat_file: Path to the MATLAB MAT file.
      channel : The channel index to extract from each EDF file (default is 0 for the first EEG channel).

    Returns:
      signals: A list of numpy arrays, one for each EDF file.
    """
    signals = []
    with h5py.File(mat_file, 'r') as f:
        # Confirm 'allData' group exists.
        if 'allData' not in f:
            raise KeyError("The MAT file does not contain an 'allData' group.")
        allData = f['allData']

        # Access the 'record' dataset which stores references for each EDF record.
        if 'record' not in allData:
            raise KeyError("The 'allData' group does not contain a 'record' dataset.")
        record_ds = allData['record']

        n_edf = record_ds.shape[0]
        for edf_index in range(n_edf):
            # Get the reference for this EDF record.
            ref = record_ds[edf_index, 0]
            rec = f[ref][()]
            # Make sure 'rec' is a 2D array (channels x samples).
            if rec.ndim != 2:
                raise ValueError("Expected 'record' dataset to be 2D (channels x samples).")
            # Check that the requested channel index exists for this EDF record.
            if channel >= rec.shape[0]:
                raise IndexError(f"Channel index {channel} is out of bounds for EDF file at index {edf_index}.")
            # Extract the specified channel.
            signal = rec[channel, :]
            signals.append(signal)
    return signals


def compute_psd(signal, fs=125, nperseg=None):
    """
    Computes the power spectral density (PSD) of a given 1D signal using Welch’s method.

    Parameters:
      signal : 1D numpy array of EEG data.
      fs     : Sampling frequency (default is 125 Hz).
      nperseg: Length of each segment for Welch's method. If None, defaults to fs*4.

    Returns:
      f      : Frequency bins.
      Pxx    : Power spectral density.
    """
    from scipy.signal import welch
    if nperseg is None:
        nperseg = fs * 4  # Using 4-second segments by default
    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), nperseg))
    return f, Pxx

# If the module is run as a script, show the MAT file structure.
if __name__ == "__main__":
    # Update the path to your MAT file if needed.
    mat_file = "EDF_RawData.mat"
    describe_edf_mat_structure(mat_file)