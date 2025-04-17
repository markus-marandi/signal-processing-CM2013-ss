import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pickle
import preprocess_library as prepro_lib
from scipy import signal
import pywt
import pandas as pd
from scipy.stats import skew
from scipy.integrate import simpson



if __name__ == "__main__":
    
    # Create a look up table for the channels
    channels = {
        2: 'EEG2',  # 0-based index for position 3
        3: 'ECG',   # 0-based index for position 4
        4: 'EMG',   # 0-based index for position 5
        5: 'EOGl',  # 0-based index for position 6
        6: 'EOGr',  # 0-based index for position 7
        7: 'EEG'    # 0-based index for position 8
    }

    # Create a dict for sampling rate
    sampling_rate = {
        2: 125,  # 0-based index for position 3
        3: 125,   # 0-based index for position 4
        4: 125,   # 0-based index for position 5
        5: 50,  # 0-based index for position 6
        6: 50,  # 0-based index for position 7
        7: 125   # 0-based index for position 8
    }

    # File paths
    input_dir = os.path.join(os.getcwd(), 'Input')
    output_dir = os.path.join(os.getcwd(), 'Output')
    mat_file = os.path.join(input_dir, 'EDF_RawData.mat')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_data = prepro_lib.import_edf_data(input_dir, output_dir, mat_file, sampling_rate)

    plt.plot(raw_data[0][channels[7]][0:1000])
    plt.show()

    plt.plot(raw_data[0][channels[6]][0:1000])
    plt.show()

    filtered_data = prepro_lib.filter_data(raw_data, channels, sampling_rate, output_dir, normalize=True)

    # Apply ICA for artifact removal
    ica_cleaned_data = prepro_lib.apply_ica(filtered_data, channels, sampling_rate, output_dir)

    # Create epochs from the ICA-cleaned data
    processed_data = prepro_lib.create_epochs(ica_cleaned_data, channels, sampling_rate, output_dir)
    print(f"\nExample epoch shape: {processed_data[1][channels[5]][0]}")

    # Perform wavelet decomposition
    wavelet_data = prepro_lib.wavelet_decomposition(processed_data, channels, sampling_rate, output_dir)

    # Example: Print the shape of Delta band epochs for EEG channel of the first subject
    if wavelet_data and channels[7] in wavelet_data[0]:
        print(f"Shape of Delta band epochs for {channels[7]} (Subject 1): {wavelet_data[0][channels[7]]['Delta'].shape}")

    # Perform spindle detection
    spindle_results = prepro_lib.spindle_detection(wavelet_data, channels, sampling_rate, output_dir)

    # Example: Print the number of spindles detected in the first epoch of EEG channel for the first subject
    if spindle_results and channels[7] in spindle_results[0]:
        first_epoch_mask = spindle_results[0][channels[7]][0]
        # Find distinct spindle events (changes from False to True)
        spindle_starts = np.where(np.diff(first_epoch_mask.astype(int)) == 1)[0]
        num_spindles = len(spindle_starts)
        # Handle case where spindle starts at the very beginning
        if first_epoch_mask[0]:
             num_spindles += 1
        print(f"Number of spindles detected in first epoch for {channels[7]} (Subject 1): {num_spindles}")

    # Extract features
    features_dataframe = prepro_lib.extract_features(wavelet_data, spindle_results, channels, sampling_rate, output_dir)

    # Display first few rows of the features dataframe
    print("\nFeature DataFrame Head:")
    print(features_dataframe.head())

