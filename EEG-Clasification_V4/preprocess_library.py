import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pickle
from scipy import signal

def import_edf_data(input_dir, output_dir, mat_file, sampling_rate):
    
    # Create a look up table for the channels
    channels = {
        'EEG2': 2,  # 0-based index for position 3
        'ECG': 3,   # 0-based index for position 4
        'EMG': 4,   # 0-based index for position 5
        'EOGl': 5,  # 0-based index for position 6
        'EOGr': 6,  # 0-based index for position 7
        'EEG': 7    # 0-based index for position 8
    }
    # Create the input directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Create Output directory for saving processed data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if processed data file already exists
    processed_data_file = os.path.join(output_dir, 'all_patients_data.pkl')
    if os.path.exists(processed_data_file):
        print(f"\nProcessed data file already exists at: {processed_data_file}")
        print("Loading existing data instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            all_patients_data = pickle.load(f)
        return all_patients_data

    print(f"Looking for file at: {mat_file}")
    if not os.path.exists(mat_file):
        print(f"Warning: {mat_file} does not exist. Please make sure the file is in the correct location.")

    # Initialize list to store all patients' data
    all_patients_data = []

    # Load the MATLAB file using h5py
    with h5py.File(mat_file, 'r') as f:
        # Print available top-level keys
        print("Top-level keys in the file:", list(f.keys()))
        
        # Locate all datasets with shape (4065000, 14) or similar
        record_datasets = []
        patient_info = []
        
        def find_record_datasets(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape[0] > 3000000 and obj.shape[1] == 14:
                record_datasets.append((name, obj))
                # Try to find patient information based on naming patterns
                parts = name.split('/')
                if len(parts) > 1:
                    patient_id = parts[-2]  # Assuming pattern /patient_id/record
                    patient_info.append(patient_id)
        
        f.visititems(find_record_datasets)
        
        print(f"Found {len(record_datasets)} datasets with appropriate shape")
        for i, (name, _) in enumerate(record_datasets):
            print(f"Dataset {i+1}: {name}")
        
        # Process each record dataset
        for i, (dataset_name, dataset) in enumerate(record_datasets):
            print(f"\nProcessing dataset {i+1}: {dataset_name}")
            
            # Load the data and transpose it to get (14, n)
            record_data = dataset[:].T  # Transpose to get channels as rows
            
            # Extract the channels of interest
            selected_channels = {}
            for channel_name, channel_idx in channels.items():
                selected_channels[channel_name] = record_data[channel_idx, :]
            
            # Add to all patients data list
            all_patients_data.append(selected_channels)
            
            # Print information about the channels
            print(f"Patient ID: {patient_info[i] if i < len(patient_info) else 'Unknown'}")
            print("Selected channels shape:", {k: v.shape for k, v in selected_channels.items()})
            
            # Optional: Plot a small sample of each channel for visualization
            plt.figure(figsize=(15, 10))
            for idx, (name, data) in enumerate(selected_channels.items()):
                # Plot only the first 1000 points to keep the visualization manageable
                plt.subplot(len(channels), 1, idx+1)
                plt.plot(data[:1000])
                plt.title(f"Channel: {name}")
                plt.ylabel("Amplitude")
            
            plt.tight_layout()
            # Create Figures directory if it doesn't exist
            figures_dir = os.path.join(os.path.dirname(input_dir), 'Figures')
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            
            # Save the figure in the Figures folder
            plt.savefig(os.path.join(figures_dir, f"patient_{i+1}_channels.png"))
            plt.close()

    # Save all patients' data in a single file for easy access
    # Format: all_patients_data[patient_index][channel_name]
    with open(processed_data_file, 'wb') as f:
        pickle.dump(all_patients_data, f)

    print("Processing complete! All patients' data saved to: " + processed_data_file)
    print("You can access the data using the format: all_patients_data[patient_index][channel_name]")
    print("See 'how_to_access_data.py' for examples on how to access the data.")
    
    return all_patients_data


def create_epochs(raw_data, channels, sampling_rate, output_dir):
    """
    Process the raw data by separating each channel's data into 30-second epochs
    based on the channel's sampling rate.
    
    Args:
        raw_data: List of dictionaries containing patient data
        channels: Dictionary mapping channel indices to channel names
        sampling_rate: Dictionary mapping channel indices to sampling rates
    
    Returns:
        patients_epoch: List of dictionaries containing patient data organized by epochs
    """
    # Check if processed data file already exists
    processed_data_file = os.path.join(output_dir, 'patients_epoch.pkl')
    if os.path.exists(processed_data_file):
        print(f"\nProcessed epochs file already exists at: {processed_data_file}")
        print("Loading existing epochs instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            patients_epoch = pickle.load(f)
        return patients_epoch
    
    patients_epoch = []
    
    # Process each patient's data
    for patient_idx, patient_data in enumerate(raw_data):
        print(f"Processing patient {patient_idx + 1} into epochs...")
        
        # Create a dictionary to store this patient's epochs for each channel
        patient_epochs = {}
        
        # Process each channel
        for channel_idx, channel_name in channels.items():
            # Get the channel data for this patient
            channel_data = patient_data[channel_name]
            
            # Get the sampling rate for this channel
            sr = sampling_rate[channel_idx]
            
            # Calculate the number of samples in a 30-second epoch
            samples_per_epoch = sr * 30
            
            # Calculate the total number of complete epochs
            total_epochs = len(channel_data) // samples_per_epoch
            
            # Create a list to store all epochs for this channel
            channel_epochs = []
            
            # Split the data into epochs
            for epoch_idx in range(total_epochs):
                start_idx = epoch_idx * samples_per_epoch
                end_idx = start_idx + samples_per_epoch
                epoch_data = channel_data[start_idx:end_idx]
                channel_epochs.append(epoch_data)
            
            # Store all epochs for this channel in the patient's dictionary
            patient_epochs[channel_name] = channel_epochs
            
            print(f"  - Channel {channel_name}: Created {len(channel_epochs)} epochs of length {samples_per_epoch}")
        
        # Add this patient's epochs to the list
        patients_epoch.append(patient_epochs)
    
    # Save the processed data
    with open(processed_data_file, 'wb') as f:
        pickle.dump(patients_epoch, f)
    
    print("Epochs data saved to Output/patients_epoch.pkl")
    print("Access example: patients_epoch[0]['EEG2'][100] for the 100th epoch of EEG2 of patient 1")
    
    return patients_epoch


def filter_data(raw_data, channels, sampling_rate, output_dir, normalize=True):
    """
    Clean the raw data by applying appropriate band-pass filters for each channel type.
    Optionally normalize the data.
    
    Typical filtering ranges:
    - EEG: 0.5-30 Hz
    - ECG: 0.5-100 Hz
    - EMG: 20-500 Hz (we'll use 20-125 Hz since our Nyquist frequency is 125/2 = 62.5 Hz)
    - EOG: 0.1-10 Hz
    
    Args:
        raw_data: List of dictionaries containing patient data
        channels: Dictionary mapping channel indices to channel names
        sampling_rate: Dictionary mapping channel indices to sampling rates
        output_dir: Directory to save the cleaned data
        normalize: Boolean indicating whether to normalize the data (default: True)
    
    Returns:
        filtered_data_list: List of dictionaries containing filtered patient data
    """
    # Check if processed data file already exists
    file_suffix = '_normalized' if normalize else ''
    processed_data_file = os.path.join(output_dir, f'filtered_data{file_suffix}.pkl')
    if os.path.exists(processed_data_file):
        print(f"\nFiltered data file already exists at: {processed_data_file}")
        print("Loading existing data instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            filtered_data_list = pickle.load(f)
        return filtered_data_list
    
    # Define filter parameters for each channel type
    filter_params = {
        'EEG': {'lowcut': 0.5, 'highcut': 30},
        'EEG2': {'lowcut': 0.5, 'highcut': 30},
        'ECG': {'lowcut': 0.5, 'highcut': 100},
        'EMG': {'lowcut': 20, 'highcut': 60},  # Limited by Nyquist frequency
        'EOGl': {'lowcut': 0.1, 'highcut': 10},
        'EOGr': {'lowcut': 0.1, 'highcut': 10}
    }
    
    # Create a copy to store filtered data
    filtered_data_list = []
    
    # Process each patient's data
    for patient_idx, patient_data in enumerate(raw_data):
        print(f"Filtering data for patient {patient_idx + 1}...")
        
        # Create a dictionary to store this patient's filtered channels
        filtered_channels = {}
        
        # Process each channel
        for channel_idx, channel_name in channels.items():
            # Get the channel data for this patient
            channel_data = patient_data[channel_name]
            
            # Get the sampling rate for this channel
            sr = sampling_rate[channel_idx]
            
            # Get filter parameters for this channel type
            params = filter_params[channel_name]
            lowcut = params['lowcut']
            highcut = min(params['highcut'], sr/2 - 1)  # Ensure below Nyquist frequency
            
            print(f"  - Filtering {channel_name} with band-pass {lowcut}-{highcut} Hz (sampling rate: {sr} Hz)")
            
            # Design the bandpass filter
            nyquist = 0.5 * sr
            low = lowcut / nyquist
            high = highcut / nyquist
            
            # Use a higher order for sharper filter response
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply the filter (forward and backward to avoid phase shift)
            channel_filtered = signal.filtfilt(b, a, channel_data)
            
            # Normalize the data if requested
            if normalize:
                print(f"  - Normalizing {channel_name} data")
                # Z-score normalization (mean=0, std=1)
                mean_val = np.mean(channel_filtered)
                std_val = np.std(channel_filtered)
                if std_val > 0:  # Avoid division by zero
                    normalized_data = (channel_filtered - mean_val) / std_val
                    channel_filtered = normalized_data
                else:
                    print(f"  - Warning: Standard deviation is zero for {channel_name}, skipping normalization")
            
            # Store the filtered (and possibly normalized) data
            filtered_channels[channel_name] = channel_filtered
            
            # Create a plot to visualize the filtering effect (original vs filtered)
            if patient_idx == 0:  # Only for the first patient
                plt_samples = 30 * sampling_rate[channel_idx]  # Plot 30 seconds for visualization
                plt.figure(figsize=(15, 5))
                plt.plot(channel_data[:plt_samples], 'b-', label='Original', alpha=0.5)
                plt.plot(channel_filtered[:plt_samples], 'r-', label='Filtered' + (' + Normalized' if normalize else ''))
                plt.title(f'Original vs Processed {channel_name} (band-pass {lowcut}-{highcut} Hz' + (', Normalized' if normalize else '') + ')')
                plt.legend()
                
                # Create folder for filter visualization
                filter_dir = os.path.join(output_dir, 'filter_visualization')
                if not os.path.exists(filter_dir):
                    os.makedirs(filter_dir)
                    
                plt.savefig(os.path.join(filter_dir, f'{channel_name}_filter_effect' + ('_normalized' if normalize else '') + '.png'))
                plt.close()
        
        # Add this patient's filtered channels to the list
        filtered_data_list.append(filtered_channels)
    
    # Save the filtered data
    with open(processed_data_file, 'wb') as f:
        pickle.dump(filtered_data_list, f)
    
    print(f"Filtered data saved to {processed_data_file}")
    
    return filtered_data_list