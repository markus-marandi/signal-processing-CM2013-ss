import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pickle
import preprocess_library as prepro_lib
from scipy import signal



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

    processed_data = prepro_lib.create_epochs(filtered_data, channels, sampling_rate, output_dir)
    print(f"\nExample epoch shape: {processed_data[1][channels[5]][0]}")
