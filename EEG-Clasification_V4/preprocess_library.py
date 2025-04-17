import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pickle
from scipy import signal
from scipy.signal import resample
from sklearn.decomposition import FastICA
import pywt
import pandas as pd
from scipy.stats import skew
from scipy.integrate import simpson

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


def apply_ica(filtered_data, channels, sampling_rate, output_dir, eeg_channels=['EEG', 'EEG2'], artifact_channels=['EOGl', 'EOGr', 'EMG', 'ECG']):
    """
    Apply Independent Component Analysis (ICA) to remove artifacts from EEG signals.

    Args:
        filtered_data: List of dictionaries containing filtered patient data.
        channels: Dictionary mapping channel indices to channel names.
        sampling_rate: Dictionary mapping channel indices to sampling rates.
        output_dir: Directory to save the cleaned data.
        eeg_channels: List of names for the EEG channels to be cleaned.
        artifact_channels: List of names for channels considered as artifact sources.

    Returns:
        ica_cleaned_data: List of dictionaries containing patient data with cleaned EEG signals.
    """
    # Determine if normalization was applied based on input data structure (assuming presence of mean/std implies normalization happened)
    # This is a heuristic; ideally, the normalization status would be passed explicitly or stored with the data.
    # For now, we'll assume the input `filtered_data` corresponds to the normalized version if it exists.
    was_normalized = '_normalized' in next((f for f in os.listdir(output_dir) if f.startswith('filtered_data')), '')

    file_suffix = '_normalized' if was_normalized else ''
    processed_data_file = os.path.join(output_dir, f'ica_cleaned_data{file_suffix}.pkl')
    
    if os.path.exists(processed_data_file):
        print(f"\nICA cleaned data file already exists at: {processed_data_file}")
        print("Loading existing data instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            ica_cleaned_data = pickle.load(f)
        return ica_cleaned_data

    print("\nApplying ICA for artifact removal...")
    ica_cleaned_data_list = []
    target_sr = sampling_rate[next(iter(key for key, name in channels.items() if name in eeg_channels))] # Get SR from the first EEG channel

    # Process each patient
    for patient_idx, patient_data in enumerate(filtered_data):
        print(f"Processing patient {patient_idx + 1} with ICA...")
        
        cleaned_patient_data = {}
        ica_input_data = []
        original_lengths = {}
        resampled_artifact_data = {} # Store resampled artifact signals for correlation

        # Prepare data for ICA: Collect EEG and resample/collect artifact channels
        print("  - Preparing data for ICA...")
        all_channel_names = eeg_channels + artifact_channels
        
        channel_map_inv = {v: k for k, v in channels.items()} # Map name back to index if needed

        for name in all_channel_names:
            channel_idx = channel_map_inv[name]
            sr = sampling_rate[channel_idx]
            data = patient_data[name]
            original_lengths[name] = len(data)

            if name in eeg_channels:
                if sr != target_sr:
                     # This case should ideally not happen if all EEG channels have the same SR
                    print(f"    Warning: EEG channel {name} has unexpected sampling rate {sr} Hz. Expected {target_sr} Hz. Skipping resampling for now.")
                    ica_input_data.append(data)
                else:
                    print(f"    Using EEG channel: {name} (Length: {len(data)}, SR: {sr} Hz)")
                    ica_input_data.append(data)
            
            elif name in artifact_channels:
                if sr != target_sr:
                    # Resample artifact channel to target SR
                    target_len = int(len(data) * target_sr / sr)
                    print(f"    Resampling artifact channel: {name} from {sr} Hz (Len: {len(data)}) to {target_sr} Hz (Target Len: {target_len})")
                    resampled_sig = resample(data, target_len)
                    ica_input_data.append(resampled_sig)
                    resampled_artifact_data[name] = resampled_sig
                else:
                    print(f"    Using artifact channel: {name} (Length: {len(data)}, SR: {sr} Hz)")
                    ica_input_data.append(data)
                    resampled_artifact_data[name] = data # Store non-resampled too

        # Ensure all data streams for ICA have the same length (due to potential rounding in resampling)
        min_len = min(len(d) for d in ica_input_data)
        ica_input_matrix = np.array([d[:min_len] for d in ica_input_data]) # Shape: (n_channels, n_samples)
        
        # --- Define component_idx_map here ---
        component_idx_map = {name: i for i, name in enumerate(all_channel_names)} # Map name to row index in ica_input_matrix
        # --- Store original EEG for plotting ---
        original_eeg_signals_for_plot = {name: ica_input_matrix[component_idx_map[name], :] for name in eeg_channels} # Store the version fed into ICA

        print(f"  - Running ICA on matrix of shape: {ica_input_matrix.shape}")
        n_components = ica_input_matrix.shape[0] # Use number of channels as components
        ica = FastICA(n_components=n_components, random_state=0, whiten='unit-variance', max_iter=1000, tol=0.01)
        
        try:
            components = ica.fit_transform(ica_input_matrix.T).T # Input should be (samples, features); components shape (n_components, n_samples)
        except Exception as e:
            print(f"    Error during ICA fitting for patient {patient_idx + 1}: {e}")
            print(f"    Skipping ICA for this patient and using original EEG data.")
            # Copy original data if ICA fails
            for name in patient_data:
                 cleaned_patient_data[name] = patient_data[name]
            ica_cleaned_data_list.append(cleaned_patient_data)
            continue # Skip to next patient

        mixing_matrix = ica.mixing_ # Shape: (n_features, n_components)
        unmixing_matrix = ica.components_ # Shape: (n_components, n_features)

        print(f"  - Identifying artifact components...")
        artifact_indices = []
        correlation_threshold = 0.6 # Adjust threshold as needed

        # Correlate components with (resampled) artifact channels
        # component_idx_map = {name: i for i, name in enumerate(all_channel_names)} # Map name to row index in ica_input_matrix -- MOVED EARLIER

        for i in range(n_components):
            component = components[i, :]
            for art_name in artifact_channels:
                if art_name in resampled_artifact_data:
                    artifact_signal = resampled_artifact_data[art_name][:min_len] # Ensure same length
                    correlation = np.abs(np.corrcoef(component, artifact_signal)[0, 1])
                    print(f"    Component {i} vs {art_name} correlation: {correlation:.3f}")
                    if correlation > correlation_threshold:
                        print(f"    -> Found artifact component {i} strongly correlated with {art_name}.")
                        if i not in artifact_indices:
                             artifact_indices.append(i)
                        # Don't break here, a component might correlate with multiple artifacts

        # Reconstruct EEG signals, removing artifact components
        print(f"  - Reconstructing cleaned EEG signals (removing components: {artifact_indices})...")
        
        # Zero out columns in mixing matrix corresponding to artifacts
        cleaned_mixing_matrix = mixing_matrix.copy()
        cleaned_mixing_matrix[:, artifact_indices] = 0

        # Reconstruct the signals: X_clean = S @ A_clean = (X @ W) @ A_clean 
        # Note: S = components.T, A = mixing_matrix
        reconstructed_signals = components.T @ cleaned_mixing_matrix.T # Shape: (n_samples, n_features/channels)
        reconstructed_signals = reconstructed_signals.T # Back to (n_channels, n_samples)

        # --- Add Plotting logic here for the first patient ---
        if patient_idx == 0:
            print("  - Generating comparison plots for Patient 1...")
            figures_dir = os.path.join(output_dir, 'Figures', 'ICA_Comparison') # Save in Figures/ICA_Comparison
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)

            plot_seconds = 30 # How many seconds to plot
            plot_samples = int(plot_seconds * target_sr)
            
            for eeg_name in eeg_channels:
                try:
                    original_signal_plot = original_eeg_signals_for_plot[eeg_name][:plot_samples]
                    
                    # Find the corresponding cleaned signal from reconstructed_signals
                    recon_idx = component_idx_map[eeg_name]
                    cleaned_signal_plot = reconstructed_signals[recon_idx, :plot_samples]

                    time_axis = np.arange(len(original_signal_plot)) / target_sr

                    plt.figure(figsize=(15, 6))
                    plt.plot(time_axis, original_signal_plot, label=f'Filtered {eeg_name}', alpha=0.8)
                    plt.plot(time_axis, cleaned_signal_plot, label=f'ICA Cleaned {eeg_name}', alpha=0.8)
                    plt.title(f'Patient 1: {eeg_name} - Filtered vs ICA Cleaned ({plot_seconds} seconds)')
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude" + (" (Normalized)" if was_normalized else ""))
                    plt.legend()
                    plt.tight_layout()
                    plot_filename = os.path.join(figures_dir, f"patient_1_{eeg_name}_ICA_comparison{file_suffix}.png")
                    plt.savefig(plot_filename)
                    plt.close()
                    print(f"    Saved comparison plot: {plot_filename}")
                except Exception as plot_e:
                     print(f"    Could not generate plot for {eeg_name}: {plot_e}")
        # --- End Plotting logic ---


        # Store cleaned EEG and original artifact channels
        for name in patient_data: # Iterate through original channel names
             channel_idx = channel_map_inv[name]
             sr = sampling_rate[channel_idx]
             original_len = original_lengths[name]

             if name in eeg_channels:
                 # Find the corresponding row in the reconstructed matrix
                 recon_idx = component_idx_map[name]
                 cleaned_signal = reconstructed_signals[recon_idx, :]
                 
                 # Resample back to original length if needed (e.g., if min_len was shorter)
                 if len(cleaned_signal) != original_len:
                     print(f"    Resampling cleaned {name} back to original length {original_len}")
                     # Use signal.resample (or resample_poly for potentially better quality)
                     cleaned_signal = resample(cleaned_signal, original_len)
                 
                 cleaned_patient_data[name] = cleaned_signal
                 print(f"    Stored cleaned EEG channel: {name}")
             elif name in artifact_channels:
                 # Keep original artifact channels (not the resampled ones used for correlation)
                 cleaned_patient_data[name] = patient_data[name]
                 print(f"    Stored original artifact channel: {name}")
             else:
                 # Keep any other channels unmodified
                 cleaned_patient_data[name] = patient_data[name]
                 print(f"    Stored other channel: {name}")


        ica_cleaned_data_list.append(cleaned_patient_data)

    # Save the ICA cleaned data
    print(f"\nSaving ICA cleaned data to {processed_data_file}")
    with open(processed_data_file, 'wb') as f:
        pickle.dump(ica_cleaned_data_list, f)

    return ica_cleaned_data_list


# Wavelet Decomposition
def wavelet_decomposition(processed_data, channels, sampling_rate, output_dir):
    """Performs wavelet decomposition on EEG signals for each epoch.

    Args:
        processed_data (list): List of dictionaries, where each dict holds
                                 epoched data for one subject.
        channels (dict): Dictionary mapping channel index to channel name.
        sampling_rate (dict): Dictionary mapping channel index to sampling rate.
        output_dir (str): Path to the output directory.

    Returns:
        list: List of dictionaries containing wavelet features for each subject.
    """
    output_file = os.path.join(output_dir, 'wavelet_features.pkl')

    if os.path.exists(output_file):
        print(f"Loading wavelet features from {output_file}...")
        with open(output_file, 'rb') as f:
            wavelet_features = pickle.load(f)
        return wavelet_features

    print("Calculating wavelet features...")
    wavelet_features = []
    wavelet_name = 'db4'  # Daubechies 4 wavelet
    level = 5  # Decomposition level

    # Define band reconstructions based on DWT levels (adjust if needed based on wavelet/level)
    # Fs = 125 Hz, Level 5 -> Frequencies (approx):
    # D1: 31.25 - 62.5 Hz (Gamma)
    # D2: 15.6 - 31.25 Hz (Beta)
    # D3: 7.8 - 15.6 Hz (Alpha + Sigma)
    # D4: 3.9 - 7.8 Hz (Theta)
    # D5: 1.95 - 3.9 Hz (Delta)
    # A5: 0 - 1.95 Hz (Delta)
    band_mapping = {
        'Gamma': [1], # D1
        'Beta': [2],  # D2
        'Sigma': [3], # D3 - Approximation for Sigma (often 12-16 Hz)
        'Alpha': [3], # D3 - Approximation for Alpha (often 8-12 Hz)
        'Theta': [4], # D4
        'Delta': [0, 5] # A5, D5
    }

    eeg_channel_keys = [k for k, v in channels.items() if 'EEG' in v]

    for subject_idx, subject_data in enumerate(processed_data):
        print(f" Processing subject {subject_idx + 1}/{len(processed_data)}")
        subject_features = {}
        for ch_key in eeg_channel_keys:
            ch_name = channels[ch_key]
            fs = sampling_rate[ch_key]
            if fs != 125:
                 print(f"Warning: Unexpected sampling rate {fs} for {ch_name}. Wavelet band mapping assumes 125 Hz.")

            ch_epochs = subject_data[ch_name]
            channel_bands = {'Original': ch_epochs}

            for band_name, coeff_indices in band_mapping.items():
                band_epochs = []
                for epoch in ch_epochs:
                    # Perform DWT
                    coeffs = pywt.wavedec(epoch, wavelet_name, level=level)
                    
                    # Create a template for reconstruction with zeros
                    recon_coeffs = [np.zeros_like(c) for c in coeffs]
                    
                    # Place the coefficients for the current band
                    for idx in coeff_indices:
                        if idx == 0: # Approximation coefficient
                            recon_coeffs[0] = coeffs[0]
                        elif idx <= level:
                             # Detail coefficients are indexed from 1 (cD1) to level (cDlevel)
                             # List coeffs is [cA_level, cD_level, cD_level-1, ..., cD1]
                             # So cD1 is coeffs[-1], cD2 is coeffs[-2], ..., cD_level is coeffs[1]
                             recon_coeffs[level - idx + 1] = coeffs[level - idx + 1]
                        else:
                             print(f"Warning: Coefficient index {idx} out of range for level {level}.")

                    # Reconstruct the signal for the band
                    reconstructed_signal = pywt.waverec(recon_coeffs, wavelet_name)
                    
                    # Ensure reconstructed signal has the same length as the original epoch
                    if len(reconstructed_signal) > len(epoch):
                        reconstructed_signal = reconstructed_signal[:len(epoch)]
                    elif len(reconstructed_signal) < len(epoch):
                        # Pad if necessary (less likely with waverec)
                        padding = len(epoch) - len(reconstructed_signal)
                        reconstructed_signal = np.pad(reconstructed_signal, (0, padding), 'constant')

                    band_epochs.append(reconstructed_signal)
                channel_bands[band_name] = np.array(band_epochs)
            subject_features[ch_name] = channel_bands
        wavelet_features.append(subject_features)

    print(f"Saving wavelet features to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(wavelet_features, f)

    return wavelet_features


def spindle_detection(wavelet_data, channels, sampling_rate, output_dir,
                      rms_thresh_multiplier=1.5, min_duration_s=0.5, max_duration_s=2.0):
    """Detects sleep spindles in the sigma band of EEG signals for each epoch.

    Args:
        wavelet_data (list): List of dictionaries containing wavelet features (including Sigma band)
                             for each subject.
        channels (dict): Dictionary mapping channel index to channel name.
        sampling_rate (dict): Dictionary mapping channel index to sampling rate.
        output_dir (str): Path to the output directory.
        rms_thresh_multiplier (float): Multiplier for RMS standard deviation threshold.
        min_duration_s (float): Minimum duration of a spindle in seconds.
        max_duration_s (float): Maximum duration of a spindle in seconds.

    Returns:
        list: List of dictionaries containing spindle detection results for each subject.
              Each channel's entry will have a boolean array indicating spindle presence
              for each sample within each epoch.
    """
    output_file = os.path.join(output_dir, 'spindle_results.pkl')

    if os.path.exists(output_file):
        print(f"Loading spindle results from {output_file}...")
        with open(output_file, 'rb') as f:
            spindle_results = pickle.load(f)
        return spindle_results

    print("Detecting sleep spindles...")
    spindle_results = []
    eeg_channel_keys = [k for k, v in channels.items() if 'EEG' in v]

    for subject_idx, subject_wavelet_data in enumerate(wavelet_data):
        print(f" Processing subject {subject_idx + 1}/{len(wavelet_data)} for spindles")
        subject_spindles = {}
        for ch_key in eeg_channel_keys:
            ch_name = channels[ch_key]
            fs = sampling_rate[ch_key]
            min_duration_samples = int(min_duration_s * fs)
            max_duration_samples = int(max_duration_s * fs)
            moving_rms_window_size = int(0.1 * fs) # 100 ms window for RMS

            if ch_name not in subject_wavelet_data or 'Sigma' not in subject_wavelet_data[ch_name]:
                print(f"Warning: Sigma band data not found for {ch_name} in subject {subject_idx + 1}. Skipping.")
                continue

            sigma_epochs = subject_wavelet_data[ch_name]['Sigma']
            detected_spindles_all_epochs = []

            for epoch in sigma_epochs:
                # Calculate moving RMS
                # Using pandas rolling RMS for convenience
                rms_series = pd.Series(epoch).rolling(window=moving_rms_window_size, center=True).std() # Using std as proxy for RMS variability
                # Handle NaNs at the edges produced by rolling window
                rms_series = rms_series.fillna(method='bfill').fillna(method='ffill')
                epoch_rms = rms_series.to_numpy()

                # Calculate threshold
                threshold = np.mean(epoch_rms) + rms_thresh_multiplier * np.std(epoch_rms)

                # Find supra-threshold segments
                above_threshold = epoch_rms > threshold
                
                # Find start and end points of potential spindles
                diff = np.diff(above_threshold.astype(int))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1

                # Handle cases where segment starts/ends at the epoch boundary
                if above_threshold[0]:
                    starts = np.insert(starts, 0, 0)
                if above_threshold[-1]:
                    ends = np.append(ends, len(epoch))

                # Filter by duration
                epoch_spindle_mask = np.zeros_like(epoch, dtype=bool)
                for start, end in zip(starts, ends):
                    duration = end - start
                    if min_duration_samples <= duration <= max_duration_samples:
                        epoch_spindle_mask[start:end] = True
                
                detected_spindles_all_epochs.append(epoch_spindle_mask)

            subject_spindles[ch_name] = np.array(detected_spindles_all_epochs)
        spindle_results.append(subject_spindles)

    print(f"Saving spindle results to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(spindle_results, f)

    return spindle_results

# --- Helper Functions for Feature Extraction ---

def calculate_band_power(freqs, psd, band):
    """Calculates power in a frequency band using Simpson's rule."""
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    # Use Simpson's rule for integration under the PSD curve
    bp = simpson(psd[idx_band], freqs[idx_band])
    return bp

def calculate_spectral_features(freqs, psd, total_power, fs):
    """Calculates Spectral Edge Frequency, Roll-off, Centroid, and Flux."""
    features = {}
    
    # Normalize PSD for cumulative sum
    psd_norm = psd / total_power if total_power > 0 else psd
    cum_power = np.cumsum(psd_norm) * (freqs[1] - freqs[0]) # Approximate integration

    # Spectral Edge Frequency (SEF) - 95%
    # Frequency below which 95% of the total power is contained.
    # Often related to the higher frequency content and cognitive state.
    try:
        sef_95_idx = np.where(cum_power >= 0.95)[0][0]
        features['SEF95'] = freqs[sef_95_idx]
    except IndexError:
        features['SEF95'] = freqs[-1] # Use max freq if threshold not reached

    # Spectral Roll-off - 85%
    # Frequency below which 85% of the total power resides.
    # Similar to SEF, reflects the skewness towards lower or higher frequencies.
    try:
        rolloff_85_idx = np.where(cum_power >= 0.85)[0][0]
        features['SpectralRollOff85'] = freqs[rolloff_85_idx]
    except IndexError:
        features['SpectralRollOff85'] = freqs[-1] # Use max freq if threshold not reached

    # Spectral Centroid
    # The center of mass of the spectrum, indicating the dominant frequency area.
    # Higher centroid means more high-frequency content.
    features['SpectralCentroid'] = np.sum(freqs * psd_norm) * (freqs[1] - freqs[0]) if total_power > 0 else 0

    # Spectral Flux
    # Measures the rate of change of the power spectrum.
    # Calculated as the sum of squared differences between adjacent PSD bins.
    # High flux indicates rapid spectral changes.
    psd_norm_diff = np.diff(psd_norm)
    features['SpectralFlux'] = np.sum(psd_norm_diff**2)

    return features

def calculate_hjorth_parameters(epoch):
    """Calculates Hjorth parameters (Activity, Mobility, Complexity)."""
    features = {}
    # Activity: Variance of the signal
    # Represents the signal power.
    activity = np.var(epoch)
    features['HjorthActivity'] = activity

    if activity == 0: # Avoid division by zero if signal is flat
        features['HjorthMobility'] = 0
        features['HjorthComplexity'] = 0
        return features

    # Mobility: sqrt(var(diff(epoch)) / var(epoch))
    # Represents the mean frequency or proportion of standard deviation.
    diff_epoch = np.diff(epoch)
    mobility = np.sqrt(np.var(diff_epoch) / activity)
    features['HjorthMobility'] = mobility

    # Complexity: mobility(diff(epoch)) / mobility(epoch)
    # Represents the change in frequency, or deviation from a pure sine wave.
    diff2_epoch = np.diff(diff_epoch)
    mobility_diff = np.sqrt(np.var(diff2_epoch) / np.var(diff_epoch)) if np.var(diff_epoch) > 0 else 0
    complexity = mobility_diff / mobility if mobility > 0 else 0
    features['HjorthComplexity'] = complexity

    return features

def count_spindles(spindle_mask):
    """Counts the number of distinct spindle events in a boolean mask."""
    if not np.any(spindle_mask): # No spindles
        return 0
    # Find changes from False to True
    diff = np.diff(spindle_mask.astype(int))
    starts = np.where(diff == 1)[0]
    num_spindles = len(starts)
    # Add 1 if the epoch starts with a spindle
    if spindle_mask[0]:
        num_spindles += 1
    return num_spindles

# Feature Extraction Function
def extract_features(wavelet_data, spindle_results, channels, sampling_rate, output_dir):
    """Extracts features from EEG epochs and saves them to a CSV file.

    Features include: Absolute/Relative Power, Spectral characteristics,
    Hjorth parameters, Basic Statistics, Spindle Density, Wavelet band stats.

    Args:
        wavelet_data (list): Data structure containing original epochs and reconstructed bands.
        spindle_results (list): Data structure containing spindle detection masks.
        channels (dict): Channel mapping.
        sampling_rate (dict): Sampling rate mapping.
        output_dir (str): Directory to save the output CSV.

    Returns:
        pd.DataFrame: DataFrame containing all extracted features.
    """
    output_file = os.path.join(output_dir, 'features_dataset.csv')

    if os.path.exists(output_file):
        print(f"Loading features from {output_file}...")
        features_df = pd.read_csv(output_file)
        return features_df

    print("Extracting features...")
    all_features = []
    eeg_channel_keys = [k for k, v in channels.items() if 'EEG' in v]
    
    # Define frequency bands for power calculation
    freq_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Sigma': (11, 16), # Note: Overlaps Alpha/Beta slightly
        'Beta': (13, 30),
        'Gamma': (30, 50) # Up to 50Hz based on typical EEG interest
    }
    total_power_range = (0.5, 50)
    welch_window_sec = 2 # Use 2-second window for Welch PSD

    for subject_idx, subject_data in enumerate(wavelet_data):
        print(f" Extracting features for subject {subject_idx + 1}/{len(wavelet_data)}")
        for ch_key in eeg_channel_keys:
            ch_name = channels[ch_key]
            fs = sampling_rate[ch_key]
            nperseg = int(welch_window_sec * fs)

            if ch_name not in subject_data or 'Original' not in subject_data[ch_name]:
                print(f"Warning: Original epoch data not found for {ch_name} in subject {subject_idx + 1}. Skipping.")
                continue
                
            original_epochs = subject_data[ch_name]['Original']
            if not original_epochs: # Check if list is empty
                print(f"Warning: No original epochs found for {ch_name} in subject {subject_idx + 1}. Skipping channel.")
                continue
            num_epochs = len(original_epochs)
            epoch_len_samples = original_epochs[0].shape[0] # Get shape from the first epoch
            epoch_duration_s = epoch_len_samples / fs

            # Get corresponding spindle mask for the channel
            subject_spindle_masks = spindle_results[subject_idx].get(ch_name, None)
            if subject_spindle_masks is None:
                 print(f"Warning: Spindle results not found for {ch_name} in subject {subject_idx + 1}. Spindle density will be NaN.")

            for epoch_idx in range(num_epochs):
                epoch_signal = original_epochs[epoch_idx]
                features = {
                    'Subject': subject_idx + 1,
                    'Epoch': epoch_idx + 1,
                    'Channel': ch_name
                }

                # --- Basic Statistics ---
                features['Mean'] = np.mean(epoch_signal)
                features['Median'] = np.median(epoch_signal)
                features['Variance'] = np.var(epoch_signal) # Same as Hjorth Activity
                features['Skewness'] = skew(epoch_signal)
                features['MeanAbsVal'] = np.mean(np.abs(epoch_signal))

                # --- Spectral Features (using Welch's method) ---
                freqs, psd = signal.welch(epoch_signal, fs=fs, nperseg=min(nperseg, epoch_len_samples), noverlap=nperseg // 2)
                
                # Absolute Power
                abs_power = {}
                for band_name, band_limits in freq_bands.items():
                    abs_power[band_name] = calculate_band_power(freqs, psd, band_limits)
                    features[f'{band_name}_AbsPwr'] = abs_power[band_name]
                
                # Total Power
                total_power = calculate_band_power(freqs, psd, total_power_range)
                features['Total_AbsPwr'] = total_power

                # Relative Power
                for band_name, band_abs_power in abs_power.items():
                    features[f'{band_name}_RelPwr'] = band_abs_power / total_power if total_power > 0 else 0
                
                # Other Spectral Features
                spectral_calcs = calculate_spectral_features(freqs, psd, total_power, fs)
                features.update(spectral_calcs)

                # --- Hjorth Parameters ---
                hjorth_params = calculate_hjorth_parameters(epoch_signal)
                features.update(hjorth_params)

                # --- Spindle Density ---
                if subject_spindle_masks is not None:
                    spindle_mask_epoch = subject_spindle_masks[epoch_idx]
                    num_spindles = count_spindles(spindle_mask_epoch)
                    features['SpindleDensity'] = num_spindles / epoch_duration_s
                else:
                    features['SpindleDensity'] = np.nan

                # --- Wavelet Band Statistics ---
                for band_name in freq_bands.keys(): # Use same band names
                    if band_name in subject_data[ch_name]:
                        band_signal_epoch = subject_data[ch_name][band_name][epoch_idx]
                        features[f'{band_name}_Mean'] = np.mean(band_signal_epoch)
                        features[f'{band_name}_Std'] = np.std(band_signal_epoch)
                        features[f'{band_name}_Var'] = np.var(band_signal_epoch)
                    else:
                        # Handle case where a band might be missing from wavelet_data (unlikely)
                        features[f'{band_name}_Mean'] = np.nan
                        features[f'{band_name}_Std'] = np.nan
                        features[f'{band_name}_Var'] = np.nan

                all_features.append(features)

    features_df = pd.DataFrame(all_features)
    
    print(f"Saving features to {output_file}...")
    features_df.to_csv(output_file, index=False)
    
    return features_df
