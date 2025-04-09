import numpy as np
import pandas as pd
from utils.eog.eog_epoch_feature_extraction import extract_epoch_features, segment_signal_into_epochs
from data.edf_data_import import extract_eeg_signal  # Your EDF module already provides extraction routines


# --- Example: A dummy implementation of get_edf_header_and_record ---
# In your actual pipeline, you should have a proper implementation.
def get_edf_header_and_record(mat_file, edf_index):
    """
    Dummy implementation to return a header and record from the EDF MAT file.
    In practice, your EDF import module should provide this.
    """
    # Here we simulate by using extract_eeg_signal for each channel.
    # In reality, you would read the 'allData' group directly.
    import h5py
    with h5py.File(mat_file, 'r') as f:
        allData = f['allData']
        record_ds = allData['record']
        if edf_index >= record_ds.shape[0]:
            raise IndexError("edf_index out of bounds.")
        ref = record_ds[edf_index, 0]
        rec = f[ref][()]
        if rec.ndim != 2:
            raise ValueError("Expected record shape (samples x channels).")
    # Construct a dummy header dictionary
    header = {
        'fileName': f"EDF_file_{edf_index}.edf",
        'samples': [125] * rec.shape[1],  # assume 125 Hz sampling for each channel
        'label': [f"Channel_{i}" for i in range(rec.shape[1])]
    }
    return header, rec


def process_eog_channel_with_xml(edf_mat_path, xml_data, file_index, eog_channel_idx, epoch_length_sec=30):
    """
    For the given EDF file (by index) and corresponding XML data,
    extracts the specified EOG channel, splits it into epochs,
    computes EOG features for each epoch, and attaches sleep stage labels
    from the XML (assuming the XML order matches EDF order).

    Parameters:
       edf_mat_path    : Path to the EDF_RawData.mat file.
       xml_data        : List of XML dictionaries loaded from XML_RawData.mat.
       file_index      : Index of the EDF file to process.
       eog_channel_idx : Index of the EOG channel to process.
       epoch_length_sec: Length of each epoch in seconds (default 30).

    Returns:
       A pandas DataFrame with one row per epoch, containing:
         - fileName, EOGChannel, epoch, blinkRatePerMin, numBlinks,
         - movementDensityMean, slowPower, rapidPower, powerRatio,
         - sleepStage.
    """
    # Retrieve header and record for the given EDF file.
    hdr, record = get_edf_header_and_record(edf_mat_path, file_index)
    edf_filename = hdr['fileName']

    # Extract the EOG signal from the EDF matrix.
    # (Assumes record shape is (samples, channels))
    eog_signal = record[:, eog_channel_idx]
    fs_eog = hdr['samples'][eog_channel_idx]  # Sampling frequency

    # Split the signal into epochs.
    epochs = segment_signal_into_epochs(eog_signal, fs_eog, epoch_length_sec)

    # Compute features for each epoch.
    features_list = [extract_epoch_features(epoch, fs_eog) for epoch in epochs]

    # Retrieve the corresponding XML sleep stage labels.
    xml_file_data = xml_data[file_index]
    sleep_stages = xml_file_data["stages"]

    # If there are fewer sleep stage entries than epochs, truncate.
    n_epochs = len(epochs)
    if len(sleep_stages) < n_epochs:
        print("Warning: fewer sleep stage entries than epochs; truncating epochs.")
        n_epochs = len(sleep_stages)
        features_list = features_list[:n_epochs]

    # Assemble results into a DataFrame.
    df_rows = []
    for i in range(n_epochs):
        row = {
            'fileName': edf_filename,
            'EOGChannel': hdr['label'][eog_channel_idx],
            'epoch': i,
            'blinkRatePerMin': features_list[i]['blinkRatePerMin'],
            'numBlinks': features_list[i]['numBlinks'],
            'movementDensityMean': features_list[i]['movementDensityMean'],
            'slowPower': features_list[i]['slowPower'],
            'rapidPower': features_list[i]['rapidPower'],
            'powerRatio': features_list[i]['powerRatio'],
            'sleepStage': int(sleep_stages[i])  # convert NumPy int64 to Python int
        }
        df_rows.append(row)
    return pd.DataFrame(df_rows)