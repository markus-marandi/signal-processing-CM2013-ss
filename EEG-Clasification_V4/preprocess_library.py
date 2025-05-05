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
from scipy.stats import skew, mode as scipy_mode
from scipy.integrate import simpson
import xml.etree.ElementTree as ET
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib # For saving model and encoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

def load_patient_data(num_patients, base_input_dir, channels_dict, output_dir):
    """
    Loads EDF and XML data for a specified number of patients.
    Checks for cached data first and saves data if not cached.

    Args:
        num_patients (int): The number of patients to process (e.g., 10 for R1 to R10).
        base_input_dir (str): The path to the directory containing patient files (e.g., 'Input').
        channels_dict (dict): Dictionary mapping channel indices (int) to channel names (str).
        output_dir (str): The path to the directory where processed data and cache files are saved.

    Returns:
        tuple: A tuple containing:
            - list: List of dictionaries, where each dictionary holds the raw
                    channel data for one patient (keys are channel names, values are numpy arrays).
            - list: List of dictionaries, where each dictionary holds the XML
                    data (events, stages, etc.) for one patient.
    """
    # Define cache file paths
    raw_data_cache_path = os.path.join(output_dir, 'raw_data_list.pkl')
    xml_data_cache_path = os.path.join(output_dir, 'xml_data_list.pkl')

    # Check if cached data exists
    if os.path.exists(raw_data_cache_path) and os.path.exists(xml_data_cache_path):
        print(f"Loading raw and XML data from cache files in {output_dir}...")
        # Load directly, assuming cache files are valid if they exist
        with open(raw_data_cache_path, 'rb') as f:
            raw_data_list = pickle.load(f)
        with open(xml_data_cache_path, 'rb') as f:
            xml_data_list = pickle.load(f)
        print("Successfully loaded data from cache.")
        # Perform a basic sanity check (e.g., check length)
        if len(raw_data_list) == num_patients and len(xml_data_list) == num_patients:
             print(f"Cache contains data for {len(raw_data_list)} patients.")
             return raw_data_list, xml_data_list
        else:
             print(f"Warning: Cache data length mismatch (expected {num_patients}, found {len(raw_data_list)}). Reloading data.")
             # Reset lists to proceed with loading
             raw_data_list = []
             xml_data_list = []
    else:
        print("Cache files not found. Loading data from source EDF and XML files...")
        # Initialize lists if cache doesn't exist or loading failed
        raw_data_list = []
        xml_data_list = []


    # --- If cache wasn't used, proceed with loading ---
    if not raw_data_list and not xml_data_list: # Check if lists are still empty
        # Let's Loop Over all patients files
        for k in range(1, num_patients + 1):
            patient_index = k - 1 # 0-based index for list access
            print(f"--- Processing Patient R{k} (Index: {patient_index}) ---")
            #
            ### --- 1.LOAD DATA : Load EDF & xml file for patient R{k} ---
            #
            ## Construct file paths using base_input_dir
            edf_path = os.path.join(base_input_dir, f"R{k}.edf")
            xml_path = os.path.join(base_input_dir, f"R{k}.xml")

            raw = None # Initialize raw to None for cleanup
            raw_picked = None # Initialize for cleanup

            # Load EDF data using MNE
            print(f"Loading EDF: {edf_path}")
            # Reduce MNE verbosity during loading
            # Import mne here if not imported globally or pass it as an argument
            import mne
            # Assume file exists and is loadable
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')

            # Load XML data using the library function
            print(f"Loading XML: {xml_path}")
            # Assume file exists and is readable by read_xml
            events, stages, epoch_length, has_annotations = read_xml(xml_path)

            # --- Store EDF channel data ---
            patient_raw_data = {}
            # Use the channel names from the 'channels_dict' argument
            channel_names_to_extract = list(channels_dict.values())

            # Verify which requested channels are actually available in the EDF file
            available_channels_in_edf = [ch for ch in channel_names_to_extract if ch in raw.ch_names]
            missing_channels = [ch for ch in channel_names_to_extract if ch not in raw.ch_names]

            if missing_channels:
                print(f"Warning: Requested channels {missing_channels} not found in {edf_path}. Available channels: {raw.ch_names}")

            # Extract data for the available channels among the requested ones
            if available_channels_in_edf:
                # Assume extraction works
                # Pick only the desired channels from the raw object
                raw_picked = raw.copy().pick(picks=available_channels_in_edf, verbose='WARNING')
                # Get data returns a numpy array of shape (n_channels, n_times)
                extracted_data = raw_picked.get_data()
                # Store each channel's data in the dictionary
                for i, name in enumerate(raw_picked.ch_names):
                    patient_raw_data[name] = extracted_data[i]
                print(f"  Extracted {len(available_channels_in_edf)} channels.")
                # Example shape print (optional, uncomment if needed)
                # if patient_raw_data:
                #    first_ch_name = list(patient_raw_data.keys())[0]
                #    print(f"  Example shape for '{first_ch_name}': {patient_raw_data[first_ch_name].shape}")
            else:
                print(f"Warning: None of the requested channels {channel_names_to_extract} were found in {edf_path}.")
                # patient_raw_data remains {}

            # Append the dictionary for this patient to the main list
            raw_data_list.append(patient_raw_data)

            # --- Store XML data ---
            patient_xml_data = {
                'events': events,
                'stages': stages,
                'epoch_length': epoch_length,
                'has_annotations': has_annotations
            }
            # Append the dictionary for this patient to the main list
            xml_data_list.append(patient_xml_data)
            print(f"  Stored XML data (Events: {len(events)}, Stages: {len(stages)}, Epoch Length: {epoch_length})")

            print(f"Finished processing Patient R{k}.")
            # Clean up MNE objects to free memory
            if raw_picked: del raw_picked
            if raw: del raw

        # After the loop summary
        print("\n--- Data Loading Loop Complete ---")
        print(f"Total patients attempted: {num_patients}")
        # Count how many patients had both EDF and XML successfully loaded and processed
        # Note: With error handling removed, this count should always equal num_patients if the loop completes.
        successful_loads = sum(1 for i in range(len(raw_data_list)) if raw_data_list[i] and xml_data_list[i])
        print(f"Successfully loaded data sets (EDF+XML pairs): {successful_loads}")

        # --- Save the loaded data to cache ---
        if successful_loads > 0: # Only save if some data was actually loaded
            print(f"\nSaving loaded raw and XML data to cache files in {output_dir}...")
            # Assume saving works
            # Ensure output directory exists (it should, but double-check)
            os.makedirs(output_dir, exist_ok=True)
            with open(raw_data_cache_path, 'wb') as f:
                pickle.dump(raw_data_list, f)
            with open(xml_data_cache_path, 'wb') as f:
                pickle.dump(xml_data_list, f)
            print("Data successfully saved to cache.")
        else:
            # This case is less likely without error handling, but kept for consistency
            print("\nNo data successfully loaded, skipping cache saving.")

    return raw_data_list, xml_data_list

def read_xml(xml_filename):
    """ Parses the XML file and extracts events, sleep stages, epoch length, and annotations. """
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_filename)
        root = tree.getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to read XML file {xml_filename}: {e}")

    # Extract epoch length
    epoch_length = int(root.find("EpochLength").text)

    # Define sleep stage mapping
    stage_mapping = {
        "SDO:NonRapidEyeMovementSleep-N1": 4,
        "SDO:NonRapidEyeMovementSleep-N2": 3,
        "SDO:NonRapidEyeMovementSleep-N3": 2,
        "SDO:NonRapidEyeMovementSleep-N4": 1,
        "SDO:RapidEyeMovementSleep": 0,  # REM sleep
        "SDO:WakeState": 5               # Wake state
    }

    # Initialize storage
    events = []
    stages = []
    
    # Parse events
    for scored_event in root.find("ScoredEvents"):
        event_concept = scored_event.find("EventConcept").text
        start = float(scored_event.find("Start").text)
        duration = float(scored_event.find("Duration").text)

        # Check if event is a sleep stage
        if event_concept in stage_mapping:
            stages.extend([stage_mapping[event_concept]] * int(duration))
        else:
            # Extract additional attributes if available
            spo2_nadir = scored_event.find("SpO2Nadir")
            spo2_baseline = scored_event.find("SpO2Baseline")
            desaturation = scored_event.find("Desaturation")

            event_data = {
                "EventConcept": event_concept,
                "Start": start,
                "Duration": duration,
                "SpO2Nadir": float(spo2_nadir.text) if spo2_nadir is not None else None,
                "SpO2Baseline": float(spo2_baseline.text) if spo2_baseline is not None else None,
                "Desaturation": float(desaturation.text) if desaturation is not None else None
            }
            events.append(event_data)

    return events, stages, epoch_length, bool(events)


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

# ==== FILTER HELPERS ====

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='high')
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq - 1  # adjust to safe margin
    b, a = signal.butter(order, cutoff / nyq, btype='low')
    return b, a

def notch_filter(signal, fs, freq=50.0, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, signal)


def filter_data(raw_data, channels, sampling_rate, output_dir, normalize=True):
    """
    Clean the raw data by applying appropriate band-pass filters for each channel type.
    Optionally normalize the data.
    
    Typical filtering ranges:
    - EEG: 0.5-30 Hz
    - ECG: 0.5-60 Hz
    - EMG: 20-60 Hz
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
        'EEG(sec)': {'lowcut': 0.5, 'highcut': 30},
        'ECG': {'lowcut': 0.5, 'highcut': 60},
        'EMG': {'lowcut': 20, 'highcut': 60},  # Limited by Nyquist frequency
        'EOG(L)': {'lowcut': 0.1, 'highcut': 10},
        'EOG(R)': {'lowcut': 0.1, 'highcut': 10}
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


def apply_ica_mne(raw_data, channels, sampling_rate, output_dir):
    """
    Apply ICA using MNE-Python to remove artifacts from EEG signals.

    Args:
        raw_data: List of dictionaries containing patient data
        channels: Dictionary mapping channel indices to channel names
        sampling_rate: Dictionary mapping channel indices to sampling rates
        output_dir: Directory to save the cleaned data

    Returns:
        ica_cleaned_data: List of dictionaries containing patient data with cleaned EEG signals
    """
    # Check if processed data file already exists
    processed_data_file = os.path.join(output_dir, 'ica_cleaned_data.pkl')
    if os.path.exists(processed_data_file):
        print(f"\nICA cleaned data file already exists at: {processed_data_file}")
        print("Loading existing data instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            ica_cleaned_data = pickle.load(f)
        return ica_cleaned_data

    print("\nApplying ICA for artifact removal...")
    ica_cleaned_data_list = []
    
    ica = mne.preprocessing.ICA(n_components=2, method='infomax',max_iter='auto')
    ica.fit(raw_data)
    ica

    return ica


def apply_ica(filtered_data, channels, sampling_rate, output_dir, eeg_channels=['EEG', 'EEG(sec)'], artifact_channels=['EOG(L)', 'EOG(R)', 'EMG', 'ECG']):
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
    # Determine if normalization was applied based on input data structure
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
                print(f"    Using EEG channel: {name} (Length: {len(data)}, SR: {sr} Hz)")
                ica_input_data.append(data)
            
            elif name in artifact_channels:
                print(f"    Using artifact channel: {name} (Length: {len(data)}, SR: {sr} Hz)")
                ica_input_data.append(data)
                resampled_artifact_data[name] = data

        # Ensure all data streams for ICA have the same length
        min_len = min(len(d) for d in ica_input_data)
        ica_input_matrix = np.array([d[:min_len] for d in ica_input_data]) # Shape: (n_channels, n_samples)
        
        # --- Define component_idx_map here ---
        component_idx_map = {name: i for i, name in enumerate(all_channel_names)} # Map name to row index in ica_input_matrix
        # --- Store original EEG for plotting ---
        original_eeg_signals_for_plot = {name: ica_input_matrix[component_idx_map[name], :] for name in eeg_channels} # Store the version fed into ICA

        print(f"  - Running ICA on matrix of shape: {ica_input_matrix.shape}")
        n_components = ica_input_matrix.shape[0] # Use number of channels as components
        ica = FastICA(n_components=n_components, random_state=90, whiten='unit-variance', max_iter=10000, tol=0.01)
        
        components = ica.fit_transform(ica_input_matrix.T).T # Input should be (samples, features); components shape (n_components, n_samples)

        mixing_matrix = ica.mixing_ # Shape: (n_features, n_components)
        unmixing_matrix = ica.components_ # Shape: (n_components, n_features)

        print(f"  - Identifying artifact components...")
        artifact_indices = []
        correlation_threshold = 0.7 # Adjust threshold as needed

        # Correlate components with (resampled) artifact channels
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


def lms_filter(d, x, M=10, mu=0.01):
    """Applies the LMS adaptive filter.

    Args:
        d (np.ndarray): The desired signal (e.g., noisy EEG).
        x (np.ndarray): The reference noise signal (e.g., EOG).
        M (int): Filter order (number of weights).
        mu (float): Learning rate (step size).

    Returns:
        np.ndarray: The filtered signal (error signal e).
    """
    N = len(d)
    if N != len(x):
        raise ValueError("Desired signal and reference noise must have the same length.")
    
    # Initialize weights and signals
    w = np.zeros(M)
    e = np.zeros(N) # Error signal (cleaned signal)
    y = np.zeros(N) # Filter output (estimated noise in d)

    # Pad x at the beginning to handle initial filter states
    x_padded = np.pad(x, (M-1, 0), 'constant')

    # Iterate through samples
    for n in range(N):
        x_vec = x_padded[n : n+M][::-1] # Input vector (past M samples including current)
        y[n] = np.dot(w, x_vec)         # Filter output
        e[n] = d[n] - y[n]             # Error signal (desired - output)
        
        # Update weights: w = w + mu * e * x
        # Add a small stability factor related to input power, prevents instability if x_vec is large
        norm_factor = 1 + np.dot(x_vec, x_vec) 
        w = w + (mu / norm_factor) * e[n] * x_vec 

    return e


def adaptive_filter(input_data, channels, sampling_rate, output_dir, 
                      eeg_channels=['EEG', 'EEG(sec)'], 
                      reference_channels=['EOG(L)', 'EOG(R)', 'EMG', 'ECG'],
                      lms_order=10, lms_mu=0.01):
    """
    Applies adaptive filtering (LMS) to remove noise from EEG signals using reference channels.

    Args:
        input_data (list): List of dictionaries containing patient data (NumPy arrays).
                           Expected to be the output of a previous step like filtering or ICA.
        channels (dict): Dictionary mapping channel indices to channel names.
        sampling_rate (dict): Dictionary mapping channel indices to sampling rates.
        output_dir (str): Directory to save the cleaned data.
        eeg_channels (list): List of names for the EEG channels to be cleaned.
        reference_channels (list): List of names for channels used as noise references.
        lms_order (int): Order (number of taps) for the LMS filter.
        lms_mu (float): Learning rate (step size) for the LMS filter.

    Returns:
        list: List of dictionaries containing patient data with adaptively filtered EEG signals.
    """
    # Determine if input data was normalized based on previous steps (heuristic)
    # This is tricky, rely on a consistent naming convention if possible or pass explicitly.
    # For now, assume no specific suffix needed unless derived from input file name.
    # Consider adding a flag if normalization state needs to be strictly tracked.
    processed_data_file = os.path.join(output_dir, f'adaptively_filtered_data_M{lms_order}_mu{lms_mu}.pkl')
    
    if os.path.exists(processed_data_file):
        print(f"\nAdaptively filtered data file already exists at: {processed_data_file}")
        print("Loading existing data instead of reprocessing.")
        with open(processed_data_file, 'rb') as f:
            adaptively_filtered_data = pickle.load(f)
        return adaptively_filtered_data

    print("\nApplying Adaptive Filtering (LMS)...")
    adaptively_filtered_data_list = []

    channel_map_inv = {v: k for k, v in channels.items()} # Map name back to index

    # Process each patient
    for patient_idx, patient_data in enumerate(input_data):
        print(f" Processing patient {patient_idx + 1} with Adaptive Filtering...")
        cleaned_patient_data = {}
        
        # --- Get data length and verify consistency ---
        first_channel_name = next(iter(patient_data))
        if not first_channel_name:
            print(f"  Warning: Patient {patient_idx + 1} has no data. Skipping.")
            adaptively_filtered_data_list.append({})
            continue
            
        data_length = len(patient_data[first_channel_name])
        consistent_length = True
        for name, data in patient_data.items():
            if len(data) != data_length:
                print(f"  Warning: Inconsistent data lengths found for patient {patient_idx + 1}. Channel {name} has length {len(data)}, expected {data_length}. Skipping patient.")
                consistent_length = False
                break
        if not consistent_length:
            adaptively_filtered_data_list.append(patient_data) # Append original data if inconsistent
            continue
        
        # --- Process Channels ---
        # Copy non-EEG channels directly
        for name, data in patient_data.items():
            if name not in eeg_channels:
                cleaned_patient_data[name] = data

        # Apply adaptive filtering to EEG channels
        for eeg_name in eeg_channels:
            if eeg_name not in patient_data:
                print(f"  Warning: EEG channel '{eeg_name}' not found for patient {patient_idx + 1}. Skipping.")
                continue

            print(f"  - Processing EEG channel: {eeg_name}")
            current_eeg_signal = patient_data[eeg_name].copy() # Start with the original EEG

            # Apply LMS sequentially for each reference channel
            for ref_name in reference_channels:
                if ref_name not in patient_data:
                    print(f"    - Reference channel '{ref_name}' not found. Skipping this reference.")
                    continue
                
                print(f"    - Using reference: {ref_name}")
                reference_signal = patient_data[ref_name]

                # Apply LMS: Use current EEG signal as desired, ref signal as noise input
                try:
                    cleaned_signal = lms_filter(current_eeg_signal, reference_signal, M=lms_order, mu=lms_mu)
                    current_eeg_signal = cleaned_signal # Output of this stage is input for the next
                except ValueError as e:
                     print(f"    - Error applying LMS with reference {ref_name}: {e}. Using signal before this reference.")
                     # Keep current_eeg_signal as it was before this failed reference

            # Store the final cleaned EEG signal for this channel
            cleaned_patient_data[eeg_name] = current_eeg_signal
            print(f"  - Finished processing {eeg_name}")
            
        adaptively_filtered_data_list.append(cleaned_patient_data)

    # Save the adaptively filtered data
    print(f"\nSaving adaptively filtered data to {processed_data_file}")
    with open(processed_data_file, 'wb') as f:
        pickle.dump(adaptively_filtered_data_list, f)

    return adaptively_filtered_data_list


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
        # Determine number of epochs from one of the channels (assuming same number across EEG channels)
        num_epochs = 0
        epoch_len_samples = 0
        fs = 0
        first_ch_key = None
        for ch_key in eeg_channel_keys:
            ch_name = channels[ch_key]
            if ch_name in subject_data and 'Original' in subject_data[ch_name] and subject_data[ch_name]['Original']:
                num_epochs = len(subject_data[ch_name]['Original'])
                epoch_len_samples = subject_data[ch_name]['Original'][0].shape[0]
                fs = sampling_rate[ch_key]
                first_ch_key = ch_key # Store the key we used to get epoch info
                break # Found valid channel data, exit loop

        if num_epochs == 0 or fs == 0 or first_ch_key is None:
             print(f"Warning: Could not determine epoch information for subject {subject_idx + 1}. Skipping feature extraction for this subject.")
             continue

        epoch_duration_s = epoch_len_samples / fs

        for epoch_idx in range(num_epochs):
            # We'll compute features averaged across EEG channels for each epoch
            epoch_features_agg = {} # To store aggregated features before adding subject/epoch info
            channel_count = 0

            for ch_key in eeg_channel_keys:
                ch_name = channels[ch_key]
                current_fs = sampling_rate[ch_key] # Use current channel's FS

                # Check if data exists and fs matches the one used for epoch calculation
                if ch_name not in subject_data or 'Original' not in subject_data[ch_name] or \
                   len(subject_data[ch_name]['Original']) <= epoch_idx or current_fs != fs:
                    # print(f"Debug: Skipping channel {ch_name} for epoch {epoch_idx+1} due to missing data or FS mismatch.")
                    continue
                
                epoch_signal = subject_data[ch_name]['Original'][epoch_idx]
                if epoch_signal.shape[0] != epoch_len_samples:
                    # print(f"Debug: Skipping channel {ch_name} for epoch {epoch_idx+1} due to length mismatch.")
                    continue # Skip if length mismatches (shouldn't happen with create_epochs)

                channel_count += 1
                nperseg = int(welch_window_sec * current_fs)

                # --- Basic Statistics ---
                epoch_features_agg['Mean'] = epoch_features_agg.get('Mean', 0) + np.mean(epoch_signal)
                epoch_features_agg['Median'] = epoch_features_agg.get('Median', 0) + np.median(epoch_signal)
                epoch_features_agg['Variance'] = epoch_features_agg.get('Variance', 0) + np.var(epoch_signal)
                epoch_features_agg['Skewness'] = epoch_features_agg.get('Skewness', 0) + skew(epoch_signal)
                epoch_features_agg['MeanAbsVal'] = epoch_features_agg.get('MeanAbsVal', 0) + np.mean(np.abs(epoch_signal))

                # --- Spectral Features (using Welch's method) ---
                freqs, psd = signal.welch(epoch_signal, fs=current_fs, nperseg=min(nperseg, epoch_len_samples), noverlap=nperseg // 2)

                abs_power = {}
                for band_name, band_limits in freq_bands.items():
                    power = calculate_band_power(freqs, psd, band_limits)
                    abs_power[band_name] = power
                    epoch_features_agg[f'{band_name}_AbsPwr'] = epoch_features_agg.get(f'{band_name}_AbsPwr', 0) + power

                total_power = calculate_band_power(freqs, psd, total_power_range)
                epoch_features_agg['Total_AbsPwr'] = epoch_features_agg.get('Total_AbsPwr', 0) + total_power

                for band_name, band_abs_power in abs_power.items():
                     rel_power = band_abs_power / total_power if total_power > 0 else 0
                     epoch_features_agg[f'{band_name}_RelPwr'] = epoch_features_agg.get(f'{band_name}_RelPwr', 0) + rel_power

                spectral_calcs = calculate_spectral_features(freqs, psd, total_power, current_fs)
                for key, value in spectral_calcs.items():
                    epoch_features_agg[key] = epoch_features_agg.get(key, 0) + value

                # --- Hjorth Parameters ---
                hjorth_params = calculate_hjorth_parameters(epoch_signal)
                for key, value in hjorth_params.items():
                    epoch_features_agg[key] = epoch_features_agg.get(key, 0) + value

                # --- Spindle Density ---
                # Spindle density is channel specific, average it too
                subject_spindle_masks = spindle_results[subject_idx].get(ch_name, None)
                if subject_spindle_masks is not None and len(subject_spindle_masks) > epoch_idx:
                    spindle_mask_epoch = subject_spindle_masks[epoch_idx]
                    num_spindles = count_spindles(spindle_mask_epoch)
                    density = num_spindles / epoch_duration_s
                    epoch_features_agg['SpindleDensity'] = epoch_features_agg.get('SpindleDensity', 0) + density
                elif 'SpindleDensity' not in epoch_features_agg: # Initialize if no channel has reported yet
                    epoch_features_agg['SpindleDensity'] = 0


                # --- Wavelet Band Statistics ---
                for band_name in freq_bands.keys():
                    if band_name in subject_data[ch_name] and len(subject_data[ch_name][band_name]) > epoch_idx:
                        band_signal_epoch = subject_data[ch_name][band_name][epoch_idx]
                        epoch_features_agg[f'{band_name}_Mean'] = epoch_features_agg.get(f'{band_name}_Mean', 0) + np.mean(band_signal_epoch)
                        epoch_features_agg[f'{band_name}_Std'] = epoch_features_agg.get(f'{band_name}_Std', 0) + np.std(band_signal_epoch)
                        epoch_features_agg[f'{band_name}_Var'] = epoch_features_agg.get(f'{band_name}_Var', 0) + np.var(band_signal_epoch)
                    else: # Initialize if missing
                        if f'{band_name}_Mean' not in epoch_features_agg: epoch_features_agg[f'{band_name}_Mean'] = 0
                        if f'{band_name}_Std' not in epoch_features_agg: epoch_features_agg[f'{band_name}_Std'] = 0
                        if f'{band_name}_Var' not in epoch_features_agg: epoch_features_agg[f'{band_name}_Var'] = 0


            # Average the aggregated features over the number of channels processed
            if channel_count > 0:
                final_epoch_features = {key: val / channel_count for key, val in epoch_features_agg.items()}
                final_epoch_features['Subject'] = subject_idx + 1
                final_epoch_features['Epoch'] = epoch_idx + 1
                # Add a general 'EEG' channel label since we averaged
                final_epoch_features['Channel'] = 'EEG_Avg'
                all_features.append(final_epoch_features)
            else:
                print(f"Warning: No valid EEG channel data found for subject {subject_idx+1}, epoch {epoch_idx+1}. Skipping epoch.")


    features_df = pd.DataFrame(all_features)
    
    print(f"Saving features to {output_file}...")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    features_df.to_csv(output_file, index=False)
    
    return features_df


def classify_sleep_stages(features_dataframe, xml_data, output_dir, test_size=0.3, random_state=42):
    """
    Trains an XGBoost classifier to predict sleep stages based on extracted features.

    Args:
        features_dataframe (pd.DataFrame): DataFrame containing features per epoch,
                                           must include 'Subject' and 'Epoch' columns.
                                           Assumes features are averaged across EEG channels ('EEG_Avg').
        xml_data (list): List of dictionaries, where each dictionary contains XML data
                         for a patient, including 'stages' (list of per-second stages)
                         and 'epoch_length'. Index corresponds to subject_idx.
        output_dir (str): Directory to save the trained model, label encoder, and results.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed for reproducibility of train/test split.

    Returns:
        tuple: (trained_model, label_encoder, evaluation_results)
               - trained_model: The fitted XGBoost classifier.
               - label_encoder: The fitted LabelEncoder for sleep stages.
               - evaluation_results (dict): Dictionary containing accuracy, classification report,
                                           and confusion matrix.
    """
    print("\\n--- Starting Sleep Stage Classification ---")
    # Create output sub-directory for classification results
    classification_output_dir = os.path.join(output_dir, 'classification')
    os.makedirs(classification_output_dir, exist_ok=True)

    # --- 1. Prepare Target Labels (Sleep Stages per Epoch) ---
    print("Preparing target labels from XML data...")
    epoch_labels = []
    stage_mapping_rev = { # Reverse mapping from read_xml for easier interpretation
        4: 'N1', 3: 'N2', 2: 'N3', 1: 'N4', 0: 'REM', 5: 'Wake'
    }

    for subject_idx, patient_xml in enumerate(xml_data):
        subject_id = subject_idx + 1
        if not patient_xml or 'stages' not in patient_xml or 'epoch_length' not in patient_xml:
            print(f"Warning: Missing XML data for Subject {subject_id}. Skipping.")
            continue

        stages_per_second = patient_xml['stages']
        epoch_length_sec = patient_xml['epoch_length']

        if epoch_length_sec <= 0:
             print(f"Warning: Invalid epoch length ({epoch_length_sec}) for Subject {subject_id}. Skipping.")
             continue

        num_epochs = len(stages_per_second) // epoch_length_sec

        for epoch_idx in range(num_epochs):
            epoch_start_sec = epoch_idx * epoch_length_sec
            epoch_end_sec = epoch_start_sec + epoch_length_sec
            epoch_sec_stages = stages_per_second[epoch_start_sec:epoch_end_sec]

            if not epoch_sec_stages:
                print(f"Warning: No stage data for Subject {subject_id}, Epoch {epoch_idx+1}. Skipping.")
                continue

            # Use the mode (most frequent stage) for the epoch label
            mode_result = scipy_mode(epoch_sec_stages, keepdims=False) # Use keepdims=False for newer scipy
            epoch_stage_numeric = mode_result.mode # Access mode value

            # Map numeric stage back to string label using reverse mapping
            epoch_stage_label = stage_mapping_rev.get(epoch_stage_numeric, f'Unknown_{epoch_stage_numeric}')

            epoch_labels.append({
                'Subject': subject_id,
                'Epoch': epoch_idx + 1,
                'Stage': epoch_stage_label
            })

    if not epoch_labels:
        raise ValueError("No valid epoch labels could be generated from the XML data.")

    labels_df = pd.DataFrame(epoch_labels)
    print(f"Generated {len(labels_df)} epoch labels.")
    print("Stage distribution in labels:\n", labels_df['Stage'].value_counts())

    # --- 2. Merge Features and Labels ---
    print("Merging features with labels...")
    # Ensure Subject/Epoch are the correct type for merging
    features_dataframe['Subject'] = features_dataframe['Subject'].astype(int)
    features_dataframe['Epoch'] = features_dataframe['Epoch'].astype(int)
    labels_df['Subject'] = labels_df['Subject'].astype(int)
    labels_df['Epoch'] = labels_df['Epoch'].astype(int)

    # Filter features to only include the averaged EEG channel if present
    if 'EEG_Avg' in features_dataframe['Channel'].unique():
         print("Filtering features for 'EEG_Avg' channel.")
         features_to_merge = features_dataframe[features_dataframe['Channel'] == 'EEG_Avg'].copy()
    else:
         # If no averaged channel, we might have multiple rows per Subject/Epoch (one per EEG channel)
         # For classification per epoch, we should average features here or select one channel.
         # Averaging seems consistent with the modified feature extraction.
         print("Warning: 'EEG_Avg' channel not found. Averaging features across available EEG channels per epoch.")
         feature_cols = [col for col in features_dataframe.columns if col not in ['Subject', 'Epoch', 'Channel']]
         features_to_merge = features_dataframe.groupby(['Subject', 'Epoch'])[feature_cols].mean().reset_index()
         # Need to re-add a Channel column? Or proceed without it. Let's proceed without for now.

    # Perform the merge
    merged_data = pd.merge(features_to_merge, labels_df, on=['Subject', 'Epoch'], how='inner')

    if merged_data.empty:
        raise ValueError("Merging features and labels resulted in an empty DataFrame. Check Subject/Epoch alignment.")

    print(f"Merged data shape: {merged_data.shape}")
    print("Stage distribution after merge:\n", merged_data['Stage'].value_counts())


    # --- 3. Prepare Data for XGBoost (X, y) ---
    print("Preparing data for XGBoost...")
    # Features (X): All columns except Subject, Epoch, Channel (if exists), and Stage
    feature_columns = [col for col in merged_data.columns if col not in ['Subject', 'Epoch', 'Channel', 'Stage']]
    X = merged_data[feature_columns]
    y_labels = merged_data['Stage']

    # Handle potential NaN/inf values in features (e.g., fill with mean or median)
    if X.isnull().values.any():
        print("Warning: NaN values found in features. Filling with column median.")
        X = X.fillna(X.median())
    if np.isinf(X.values).any():
         print("Warning: Infinite values found in features. Replacing with large finite numbers.")
         X = X.replace([np.inf, -np.inf], np.nan)
         X = X.fillna(X.median()) # Fill NaNs created by replacement

    # Encode target labels (y)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print(f"Encoded labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Save the label encoder
    encoder_path = os.path.join(classification_output_dir, 'label_encoder.joblib')
    joblib.dump(le, encoder_path)
    print(f"Label encoder saved to {encoder_path}")


    # --- 4. Split Data ---
    print(f"Splitting data into training ({1-test_size:.0%}) and validation ({test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Important for potentially imbalanced sleep stages
    )
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")


    # --- 5. Train XGBoost Classifier ---
    print("Training XGBoost classifier...")
    num_classes = len(le.classes_)
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # for multi-class classification
        num_class=num_classes,
        eval_metric='mlogloss',     # evaluation metric
        use_label_encoder=False,    # Recommended for recent XGBoost versions
        random_state=random_state
    )

    model.fit(X_train, y_train)
    print("Training complete.")

    # Save the trained model
    model_path = os.path.join(classification_output_dir, 'xgboost_model.joblib')
    joblib.dump(model, model_path)
    print(f"Trained XGBoost model saved to {model_path}")


    # --- 6. Evaluate Model ---
    print("\\n--- Evaluating Model on Validation Set ---")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    # Pretty print report
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    print("\\nConfusion Matrix:")
    # Print confusion matrix with labels for clarity
    conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
    print(conf_matrix_df)

    # Store results
    evaluation_results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(), # Convert numpy array for saving if needed
        'confusion_matrix_labels': le.classes_.tolist()
    }

    # Save evaluation results to a file
    results_path = os.path.join(classification_output_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    print(f"Evaluation results saved to {results_path}")

    return model, le, evaluation_results

def plot_band_power_with_stages(wavelet_data, xml_data, channels, sampling_rate, output_dir, 
                                subject_index, eeg_channel_name):
    """Generates a plot showing EEG band power per epoch overlaid with the sleep hypnogram.

    Args:
        wavelet_data (list): Output from wavelet_decomposition.
        xml_data (list): List of dictionaries containing patient XML data (stages, epoch_length).
        channels (dict): Channel mapping dictionary.
        sampling_rate (dict): Sampling rate dictionary.
        output_dir (str): Base directory to save the plot.
        subject_index (int): Index of the subject to plot (0-based).
        eeg_channel_name (str): Name of the EEG channel to use for band power.
    """
    print(f"\\nGenerating band power and hypnogram plot for Subject {subject_index + 1}, Channel {eeg_channel_name}...")

    # --- 1. Input Validation and Data Selection ---
    if subject_index >= len(wavelet_data) or subject_index >= len(xml_data):
        print(f"Error: Subject index {subject_index} is out of bounds.")
        return
        
    subject_wavelet_data = wavelet_data[subject_index]
    subject_xml_data = xml_data[subject_index]

    if eeg_channel_name not in subject_wavelet_data:
        print(f"Error: EEG channel '{eeg_channel_name}' not found in wavelet data for Subject {subject_index + 1}.")
        return
        
    if not subject_xml_data or 'stages' not in subject_xml_data or 'epoch_length' not in subject_xml_data:
        print(f"Error: Missing or incomplete XML data (stages/epoch_length) for Subject {subject_index + 1}.")
        return
        
    # Find channel key for sampling rate
    channel_key = None
    for key, name in channels.items():
        if name == eeg_channel_name:
            channel_key = key
            break
    if channel_key is None:
         print(f"Error: Could not find channel key for '{eeg_channel_name}'.")
         return
    fs = sampling_rate[channel_key]

    # --- 2. Calculate Band Power per Epoch --- 
    bands_to_plot = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
    band_powers = {band: [] for band in bands_to_plot}
    num_epochs = 0

    if 'Original' in subject_wavelet_data[eeg_channel_name]:
         num_epochs = len(subject_wavelet_data[eeg_channel_name]['Original'])
    else:
         print(f"Error: 'Original' epoch data missing for channel '{eeg_channel_name}'. Cannot determine number of epochs.")
         return
         
    if num_epochs == 0:
        print(f"Warning: No epochs found for channel '{eeg_channel_name}'. Cannot generate plot.")
        return

    print(f"  Calculating power for {num_epochs} epochs...")
    for epoch_idx in range(num_epochs):
        for band_name in bands_to_plot:
            if band_name in subject_wavelet_data[eeg_channel_name] and len(subject_wavelet_data[eeg_channel_name][band_name]) > epoch_idx:
                band_signal_epoch = subject_wavelet_data[eeg_channel_name][band_name][epoch_idx]
                # Calculate power as variance
                power = np.var(band_signal_epoch)
                band_powers[band_name].append(power)
            else:
                # Append NaN or 0 if band data is missing for this epoch
                band_powers[band_name].append(np.nan)
                print(f"Warning: Missing wavelet data for band '{band_name}', epoch {epoch_idx+1}. Plotting as NaN.")

    # Convert power lists to numpy arrays
    for band_name in bands_to_plot:
        band_powers[band_name] = np.array(band_powers[band_name])

    # --- 3. Determine Sleep Stage per Epoch --- 
    stages_per_second = subject_xml_data['stages']
    epoch_length_sec = subject_xml_data['epoch_length']
    num_epochs_xml = len(stages_per_second) // epoch_length_sec
    
    if num_epochs_xml != num_epochs:
        print(f"Warning: Number of epochs mismatch between wavelet data ({num_epochs}) and XML data ({num_epochs_xml}). Truncating to minimum.")
        num_epochs = min(num_epochs, num_epochs_xml)
        # Trim band power arrays if needed
        for band_name in bands_to_plot:
            band_powers[band_name] = band_powers[band_name][:num_epochs]
            
    epoch_stages_numeric = []
    for epoch_idx in range(num_epochs):
        epoch_start_sec = epoch_idx * epoch_length_sec
        epoch_end_sec = epoch_start_sec + epoch_length_sec
        epoch_sec_stages = stages_per_second[epoch_start_sec:epoch_end_sec]
        if not epoch_sec_stages:
            epoch_stages_numeric.append(np.nan) # Use NaN for missing stages
            continue
        mode_result = scipy_mode(epoch_sec_stages, keepdims=False)
        epoch_stages_numeric.append(mode_result.mode)
        
    epoch_stages_numeric = np.array(epoch_stages_numeric)

    # --- 4. Plotting --- 
    stage_mapping_rev = { # For labeling hypnogram
        5: 'Wake', 4: 'N1', 3: 'N2', 2: 'N3', 1: 'N4', 0: 'REM'
    }
    stage_colors = { # Colors for hypnogram stages
        5: 'orange', 4: 'yellow', 3: 'lightgreen', 2: 'lightblue', 1: 'blue', 0: 'red',
        np.nan: 'grey' # Color for unknown/missing stages
    }

    epochs_time_axis = np.arange(num_epochs)

    fig, ax1 = plt.subplots(figsize=(18, 8))
    fig.suptitle(f'Subject {subject_index + 1} - {eeg_channel_name} - Band Power & Sleep Stages')

    # Plot band powers on primary axis (ax1)
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Log Band Power (Variance)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # Use log scale for power because it varies greatly between bands
    # Add small epsilon to avoid log(0)
    epsilon = 1e-9
    for band_name in bands_to_plot:
        ax1.plot(epochs_time_axis, np.log10(band_powers[band_name] + epsilon), label=f'{band_name} Log Power')
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Create secondary axis for hypnogram (ax2)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sleep Stage', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Define hypnogram levels (reverse order for plotting: Wake highest, N4 lowest)
    hypno_levels = {'Wake': 5, 'REM': 4, 'N1': 3, 'N2': 2, 'N3': 1, 'N4': 0}
    hypno_levels_rev = {v: k for k,v in hypno_levels.items()} # map level back to name
    # Map numeric stages to plotting levels
    stage_plot_levels = np.full(num_epochs, np.nan) # Initialize with NaN
    for num_stage, name in stage_mapping_rev.items():
        level = hypno_levels.get(name, np.nan) # Get plot level, NaN if stage not in hypno_levels (e.g. N4 mapped to N3)
        if name == 'N4': level = hypno_levels['N3'] # Map N4 to N3 level for plotting if N3 exists
        stage_plot_levels[epoch_stages_numeric == num_stage] = level

    # Plot hypnogram using step plot
    ax2.step(epochs_time_axis, stage_plot_levels, where='mid', color='black', linewidth=1.5)
    # Set y-axis limits and ticks for stages
    ax2.set_ylim(-0.5, len(hypno_levels) - 0.5)
    ax2.set_yticks(list(hypno_levels.values()))
    ax2.set_yticklabels(list(hypno_levels.keys())) # Use names for ticks
    # Optional: Fill areas between steps with color
    #for stage_num, stage_name in stage_mapping_rev.items():
    #     level = hypno_levels.get(stage_name, np.nan)
    #     color = stage_colors.get(stage_num, 'grey')
    #     ax2.fill_between(epochs_time_axis, stage_plot_levels, where=(stage_plot_levels==level), 
    #                      step='mid', alpha=0.3, color=color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # --- 5. Save Plot --- 
    plot_output_dir = os.path.join(output_dir, 'Figures', 'BandPower_Hypnograms')
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filename = os.path.join(plot_output_dir, f'subject_{subject_index + 1}_{eeg_channel_name}_bandpower_hypno.png')
    
    try:
        plt.savefig(plot_filename)
        print(f"  Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory


def generate_final_visualizations(features_dataframe, xml_data, model, label_encoder, evaluation_results, output_dir):
    """Generates and saves final summary visualizations for the sleep stage classification.

    Args:
        features_dataframe (pd.DataFrame): DataFrame with extracted features per epoch.
        xml_data (list): List of patient XML data dictionaries.
        model (xgb.XGBClassifier): The trained XGBoost model.
        label_encoder (LabelEncoder): The fitted LabelEncoder for stages.
        evaluation_results (dict): Dictionary containing metrics like 'confusion_matrix'.
        output_dir (str): Base directory to save the figures.
    """
    print("\\n--- Generating Final Visualizations ---")

    # --- 0. Define Output Paths & Ensure Directories Exist ---
    figures_base_dir = os.path.join(output_dir, 'Figures')
    band_power_dir = os.path.join(figures_base_dir, 'BandPowerDistribution')
    feature_imp_dir = os.path.join(figures_base_dir, 'FeatureImportance')
    conf_matrix_dir = os.path.join(figures_base_dir, 'ConfusionMatrix')
    spindle_dir = os.path.join(figures_base_dir, 'SpindleAnalysis')

    os.makedirs(band_power_dir, exist_ok=True)
    os.makedirs(feature_imp_dir, exist_ok=True)
    os.makedirs(conf_matrix_dir, exist_ok=True)
    os.makedirs(spindle_dir, exist_ok=True)

    # --- 1. Prepare Merged Data with Stage Labels --- 
    # (Similar logic to the beginning of classify_sleep_stages)
    print("  Preparing data for visualization (merging features and labels)...")
    epoch_labels = []
    stage_mapping_rev = { # Reverse mapping from read_xml for easier interpretation
        4: 'N1', 3: 'N2', 2: 'N3', 1: 'N4', 0: 'REM', 5: 'Wake'
    }
    for subject_idx, patient_xml in enumerate(xml_data):
        subject_id = subject_idx + 1
        if not patient_xml or 'stages' not in patient_xml or 'epoch_length' not in patient_xml:
            continue # Skip if XML invalid
        stages_per_second = patient_xml['stages']
        epoch_length_sec = patient_xml['epoch_length']
        if epoch_length_sec <= 0: continue
        num_epochs = len(stages_per_second) // epoch_length_sec
        for epoch_idx in range(num_epochs):
            epoch_start_sec = epoch_idx * epoch_length_sec
            epoch_end_sec = epoch_start_sec + epoch_length_sec
            epoch_sec_stages = stages_per_second[epoch_start_sec:epoch_end_sec]
            if not epoch_sec_stages: continue
            mode_result = scipy_mode(epoch_sec_stages, keepdims=False)
            epoch_stage_numeric = mode_result.mode
            epoch_stage_label = stage_mapping_rev.get(epoch_stage_numeric, f'Unknown_{epoch_stage_numeric}')
            epoch_labels.append({'Subject': subject_id, 'Epoch': epoch_idx + 1, 'Stage': epoch_stage_label})
    
    if not epoch_labels:
        print("Error: No valid epoch labels found. Cannot generate visualizations.")
        return
    labels_df = pd.DataFrame(epoch_labels)

    # Merge features (assuming 'EEG_Avg' or averaging was done before classification)
    features_dataframe['Subject'] = features_dataframe['Subject'].astype(int)
    features_dataframe['Epoch'] = features_dataframe['Epoch'].astype(int)
    labels_df['Subject'] = labels_df['Subject'].astype(int)
    labels_df['Epoch'] = labels_df['Epoch'].astype(int)

    # Select the relevant feature rows (e.g., averaged)
    if 'EEG_Avg' in features_dataframe['Channel'].unique():
         features_to_merge = features_dataframe[features_dataframe['Channel'] == 'EEG_Avg'].copy()
    else:
         # Fallback: Average features if EEG_Avg is missing (should match classification input prep)
         print("Warning: 'EEG_Avg' channel not found in features. Averaging for visualization.")
         feature_cols_vis = [col for col in features_dataframe.columns if col not in ['Subject', 'Epoch', 'Channel']]
         features_to_merge = features_dataframe.groupby(['Subject', 'Epoch'])[feature_cols_vis].mean().reset_index()

    vis_data = pd.merge(features_to_merge, labels_df, on=['Subject', 'Epoch'], how='inner')

    if vis_data.empty:
        print("Error: Merged data for visualization is empty. Cannot proceed.")
        return
        
    # Ensure stage order for plotting consistency if possible
    stage_order = [s for s in ['Wake', 'N1', 'N2', 'N3', 'REM'] if s in vis_data['Stage'].unique()]
    # Handle N4 potentially mapped to N3 during labeling
    if 'N4' in stage_mapping_rev.values() and 'N4' not in stage_order and 'N3' in stage_order: 
        pass # Assume N4 was mapped to N3 if N4 exists in mapping but not data
    elif 'N4' in vis_data['Stage'].unique():
        stage_order.insert(stage_order.index('N3')+1 if 'N3' in stage_order else len(stage_order), 'N4')
        
    print(f"  Data prepared for plotting (shape: {vis_data.shape}). Stage order: {stage_order}")
    
    # Add necessary imports if they aren't global
    import matplotlib.pyplot as plt
    import seaborn as sns 

    # --- 2. Band Power Distribution by Stage ---
    print("  Generating Band Power Distribution plot...")
    band_power_cols = [col for col in vis_data.columns if '_AbsPwr' in col and 'Total' not in col]
    if band_power_cols:
        n_bands = len(band_power_cols)
        n_cols_plot = 3
        n_rows_plot = (n_bands + n_cols_plot - 1) // n_cols_plot
        fig_bp, axes_bp = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 5 * n_rows_plot), sharey=True)
        axes_bp = axes_bp.flatten() # Flatten for easy iteration

        for i, bp_col in enumerate(band_power_cols):
            band_name = bp_col.replace('_AbsPwr', '')
            # Log transform power for visualization (handle zeros/negatives)
            log_power = np.log10(vis_data[bp_col].clip(1e-9)) # Clip to avoid log(0)
            sns.boxplot(ax=axes_bp[i], x='Stage', y=log_power, data=vis_data, order=stage_order, showfliers=False)
            axes_bp[i].set_title(f'{band_name} Power Distribution')
            axes_bp[i].set_ylabel('Log10(Absolute Power)')
            axes_bp[i].set_xlabel('Sleep Stage')
            axes_bp[i].tick_params(axis='x', rotation=45)
            
        # Hide unused subplots
        for j in range(i + 1, len(axes_bp)):
             fig_bp.delaxes(axes_bp[j])

        plt.tight_layout()
        plot_path = os.path.join(band_power_dir, 'band_power_distribution_by_stage.png')
        try:
            plt.savefig(plot_path)
            print(f"    Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error saving band power plot: {e}")
        plt.close(fig_bp)
    else:
        print("    Skipping band power plot: No absolute power columns found.")

    # --- 3. Feature Importance ---
    print("  Generating Feature Importance plot...")
    try:
        feature_names = model.get_booster().feature_names
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot top N features (e.g., top 30)
        top_n = min(30, len(importance_df))
        plt.figure(figsize=(10, top_n * 0.3))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Feature Importances (XGBoost)')
        plt.tight_layout()
        plot_path = os.path.join(feature_imp_dir, 'feature_importances.png')
        plt.savefig(plot_path)
        print(f"    Saved plot: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"    Error generating feature importance plot: {e}")

    # --- 4. Confusion Matrix Heatmap ---
    print("  Generating Confusion Matrix heatmap...")
    if 'confusion_matrix' in evaluation_results:
        conf_matrix = np.array(evaluation_results['confusion_matrix'])
        class_names = label_encoder.classes_

        # Normalize for better color comparison if desired
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm) # handle potential division by zero
        
        fig_cm, ax_cm = plt.subplots(1, 2, figsize=(18, 7))
        fig_cm.suptitle('Confusion Matrix')

        # Plot raw counts
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm[0])
        ax_cm[0].set_title('Counts')
        ax_cm[0].set_xlabel('Predicted Label')
        ax_cm[0].set_ylabel('True Label')

        # Plot normalized percentages
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm[1])
        ax_cm[1].set_title('Normalized by True Label')
        ax_cm[1].set_xlabel('Predicted Label')
        ax_cm[1].set_ylabel('True Label')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(conf_matrix_dir, 'confusion_matrix_heatmap.png')
        try:
            plt.savefig(plot_path)
            print(f"    Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error saving confusion matrix plot: {e}")
        plt.close(fig_cm)
    else:
        print("    Skipping confusion matrix plot: 'confusion_matrix' not found in evaluation_results.")

    # --- 5. Spindle Density Distribution by Stage ---
    print("  Generating Spindle Density Distribution plot...")
    if 'SpindleDensity' in vis_data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Stage', y='SpindleDensity', data=vis_data, order=stage_order, showfliers=False)
        # sns.violinplot(x='Stage', y='SpindleDensity', data=vis_data, order=stage_order, inner='quartile') # Alternative
        plt.title('Spindle Density Distribution by Sleep Stage')
        plt.xlabel('Sleep Stage')
        plt.ylabel('Spindle Density (spindles/second)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(spindle_dir, 'spindle_density_distribution_by_stage.png')
        try:
            plt.savefig(plot_path)
            print(f"    Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error saving spindle density plot: {e}")
        plt.close()
    else:
        print("    Skipping spindle density plot: 'SpindleDensity' column not found.")

    print("--- Finished Generating Final Visualizations ---")

def classify_sleep_stages_nn(features_dataframe, xml_data, output_dir, test_size=0.3, random_state=42):
    """
    Trains a Neural Network classifier (TensorFlow/Keras) to predict sleep stages.

    Args:
        features_dataframe (pd.DataFrame): DataFrame containing features per epoch.
        xml_data (list): List of patient XML data dictionaries.
        output_dir (str): Directory to save the model, encoder, scaler, and results.
        test_size (float): Proportion of the dataset for the validation split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (trained_model, label_encoder, scaler, evaluation_results)
               - trained_model: The fitted Keras model.
               - label_encoder: The fitted LabelEncoder for sleep stages.
               - scaler: The fitted StandardScaler for features.
               - evaluation_results (dict): Dictionary with accuracy, report, confusion matrix.
    """
    print("\\n--- Starting Sleep Stage Classification (Neural Network) ---")

    # Create output sub-directory for NN classification results
    classification_output_dir = os.path.join(output_dir, 'classification_nn')
    os.makedirs(classification_output_dir, exist_ok=True)

    # --- 1. Prepare Target Labels (Same as XGBoost version) ---
    print("Preparing target labels from XML data...")
    # (This part is identical to the XGBoost function - reusing logic)
    epoch_labels = []
    stage_mapping_rev = { 4: 'N1', 3: 'N2', 2: 'N3', 1: 'N4', 0: 'REM', 5: 'Wake'}
    for subject_idx, patient_xml in enumerate(xml_data):
        subject_id = subject_idx + 1
        if not patient_xml or 'stages' not in patient_xml or 'epoch_length' not in patient_xml:
            continue
        stages_per_second = patient_xml['stages']
        epoch_length_sec = patient_xml['epoch_length']
        if epoch_length_sec <= 0: continue
        num_epochs = len(stages_per_second) // epoch_length_sec
        for epoch_idx in range(num_epochs):
            epoch_start_sec = epoch_idx * epoch_length_sec
            epoch_end_sec = epoch_start_sec + epoch_length_sec
            epoch_sec_stages = stages_per_second[epoch_start_sec:epoch_end_sec]
            if not epoch_sec_stages: continue
            mode_result = scipy_mode(epoch_sec_stages, keepdims=False) 
            epoch_stage_numeric = mode_result.mode
            epoch_stage_label = stage_mapping_rev.get(epoch_stage_numeric, f'Unknown_{epoch_stage_numeric}')
            epoch_labels.append({'Subject': subject_id, 'Epoch': epoch_idx + 1, 'Stage': epoch_stage_label})
    if not epoch_labels: raise ValueError("No valid epoch labels could be generated.")
    labels_df = pd.DataFrame(epoch_labels)
    print(f"Generated {len(labels_df)} epoch labels.")

    # --- 2. Merge Features and Labels --- 
    print("Merging features with labels...")
    # (Identical merging logic as XGBoost version)
    features_dataframe['Subject'] = features_dataframe['Subject'].astype(int)
    features_dataframe['Epoch'] = features_dataframe['Epoch'].astype(int)
    labels_df['Subject'] = labels_df['Subject'].astype(int)
    labels_df['Epoch'] = labels_df['Epoch'].astype(int)
    if 'EEG_Avg' in features_dataframe['Channel'].unique():
         features_to_merge = features_dataframe[features_dataframe['Channel'] == 'EEG_Avg'].copy()
    else:
         print("Warning: 'EEG_Avg' channel not found. Averaging features across EEG channels.")
         feature_cols = [col for col in features_dataframe.columns if col not in ['Subject', 'Epoch', 'Channel']]
         features_to_merge = features_dataframe.groupby(['Subject', 'Epoch'])[feature_cols].mean().reset_index()
    merged_data = pd.merge(features_to_merge, labels_df, on=['Subject', 'Epoch'], how='inner')
    if merged_data.empty: raise ValueError("Merging features and labels resulted in an empty DataFrame.")
    print(f"Merged data shape: {merged_data.shape}")

    # --- 3. Prepare Data for Neural Network --- 
    print("Preparing data for Neural Network (Scaling, Encoding)...")
    feature_columns = [col for col in merged_data.columns if col not in ['Subject', 'Epoch', 'Channel', 'Stage']]
    X = merged_data[feature_columns]
    y_labels = merged_data['Stage']

    # Handle potential NaN/inf values (important before scaling)
    if X.isnull().values.any():
        print("Warning: NaN values found in features. Filling with column median.")
        X = X.fillna(X.median())
    if np.isinf(X.values).any():
         print("Warning: Infinite values found in features. Replacing with large finite numbers and filling NaNs.")
         X = X.replace([np.inf, -np.inf], np.nan)
         X = X.fillna(X.median()) # Fill NaNs created by replacement or original NaNs

    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Features scaled using StandardScaler. Shape: {X_scaled.shape}")
    
    # Save the scaler
    scaler_path = os.path.join(classification_output_dir, 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Feature scaler saved to {scaler_path}")

    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    num_classes = len(le.classes_)
    print(f"Encoded labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Convert labels to one-hot encoding for Keras
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    print(f"Labels converted to one-hot encoding. Shape: {y_one_hot.shape}")

    # Save the label encoder
    encoder_path = os.path.join(classification_output_dir, 'label_encoder.joblib')
    joblib.dump(le, encoder_path)
    print(f"Label encoder saved to {encoder_path}")

    # --- 4. Split Data --- 
    print(f"Splitting data into training ({1-test_size:.0%}) and validation ({test_size:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_one_hot, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encoded # Stratify based on original encoded labels before one-hot
    )
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")

    # --- 5. Define Keras Model --- 
    print("Defining Keras Sequential model...")
    input_shape = (X_train.shape[1],)
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    # --- 6. Compile Model --- 
    print("Compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # --- 7. Train Model --- 
    print("Training Neural Network model...")
    batch_size = 64
    epochs = 100 # Max epochs, EarlyStopping will likely stop it sooner
    
    # Define callbacks
    model_checkpoint_path = os.path.join(classification_output_dir, 'best_nn_model.keras') # Use .keras format
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=False), # Restore_best_weights handled by loading saved model
        ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        verbose=2 # Set verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    )
    
    print("Training complete.")

    # --- 8. Load Best Model & Evaluate --- 
    print("Loading best model saved during training...")
    try:
        best_model = tf.keras.models.load_model(model_checkpoint_path)
        print(f"Best model loaded from {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading saved model: {e}. Using the model from the end of training.")
        best_model = model # Fallback to the model at the end of training

    print("\\n--- Evaluating Best Model on Validation Set ---")
    # Get predictions (probabilities)
    y_pred_proba = best_model.predict(X_val)
    # Convert probabilities to class indices
    y_pred_indices = np.argmax(y_pred_proba, axis=1)
    # Convert true one-hot labels back to class indices
    y_true_indices = np.argmax(y_val, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    report = classification_report(y_true_indices, y_pred_indices, target_names=le.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_true_indices, y_pred_indices)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    print("\\nConfusion Matrix:")
    conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
    print(conf_matrix_df)

    # Store results
    evaluation_results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'confusion_matrix_labels': le.classes_.tolist()
    }
    results_path = os.path.join(classification_output_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    print(f"Evaluation results saved to {results_path}")

    # --- 9. Return --- 
    # Note: The saved model is separate, we return the loaded best model object
    return best_model, le, scaler, evaluation_results

def generate_final_visualizations_nn(features_dataframe, xml_data, label_encoder, evaluation_results, output_dir):
    """Generates and saves final summary visualizations for Neural Network classification.

    Args:
        features_dataframe (pd.DataFrame): DataFrame with extracted features per epoch.
        xml_data (list): List of patient XML data dictionaries.
        label_encoder (LabelEncoder): The fitted LabelEncoder for stages.
        evaluation_results (dict): Dictionary containing metrics like 'confusion_matrix'.
        output_dir (str): Base directory to save the figures.
    """
    print("\\n--- Generating Final Visualizations (Neural Network) ---")

    # --- 0. Define Output Paths & Ensure Directories Exist ---
    # Save NN plots in a separate subdirectory to avoid overwriting XGBoost ones
    figures_base_dir = os.path.join(output_dir, 'Figures_NN') # Changed base directory
    band_power_dir = os.path.join(figures_base_dir, 'BandPowerDistribution')
    conf_matrix_dir = os.path.join(figures_base_dir, 'ConfusionMatrix')
    spindle_dir = os.path.join(figures_base_dir, 'SpindleAnalysis')

    os.makedirs(figures_base_dir, exist_ok=True)
    os.makedirs(band_power_dir, exist_ok=True)
    os.makedirs(conf_matrix_dir, exist_ok=True)
    os.makedirs(spindle_dir, exist_ok=True)

    # --- 1. Prepare Merged Data with Stage Labels --- 
    # (Identical logic to the previous visualization function)
    print("  Preparing data for visualization (merging features and labels)...")
    epoch_labels = []
    stage_mapping_rev = { 4: 'N1', 3: 'N2', 2: 'N3', 1: 'N4', 0: 'REM', 5: 'Wake'}
    for subject_idx, patient_xml in enumerate(xml_data):
        subject_id = subject_idx + 1
        if not patient_xml or 'stages' not in patient_xml or 'epoch_length' not in patient_xml: continue
        stages_per_second = patient_xml['stages']
        epoch_length_sec = patient_xml['epoch_length']
        if epoch_length_sec <= 0: continue
        num_epochs = len(stages_per_second) // epoch_length_sec
        for epoch_idx in range(num_epochs):
            epoch_start_sec = epoch_idx * epoch_length_sec
            epoch_end_sec = epoch_start_sec + epoch_length_sec
            epoch_sec_stages = stages_per_second[epoch_start_sec:epoch_end_sec]
            if not epoch_sec_stages: continue
            mode_result = scipy_mode(epoch_sec_stages, keepdims=False)
            epoch_stage_numeric = mode_result.mode
            epoch_stage_label = stage_mapping_rev.get(epoch_stage_numeric, f'Unknown_{epoch_stage_numeric}')
            epoch_labels.append({'Subject': subject_id, 'Epoch': epoch_idx + 1, 'Stage': epoch_stage_label})
    if not epoch_labels: 
        print("Error: No valid epoch labels found. Cannot generate visualizations.")
        return
    labels_df = pd.DataFrame(epoch_labels)

    features_dataframe['Subject'] = features_dataframe['Subject'].astype(int)
    features_dataframe['Epoch'] = features_dataframe['Epoch'].astype(int)
    labels_df['Subject'] = labels_df['Subject'].astype(int)
    labels_df['Epoch'] = labels_df['Epoch'].astype(int)

    if 'EEG_Avg' in features_dataframe['Channel'].unique():
         features_to_merge = features_dataframe[features_dataframe['Channel'] == 'EEG_Avg'].copy()
    else:
         print("Warning: 'EEG_Avg' channel not found in features. Averaging for visualization.")
         feature_cols_vis = [col for col in features_dataframe.columns if col not in ['Subject', 'Epoch', 'Channel']]
         features_to_merge = features_dataframe.groupby(['Subject', 'Epoch'])[feature_cols_vis].mean().reset_index()

    vis_data = pd.merge(features_to_merge, labels_df, on=['Subject', 'Epoch'], how='inner')
    if vis_data.empty: 
        print("Error: Merged data for visualization is empty. Cannot proceed.")
        return
        
    stage_order = [s for s in ['Wake', 'N1', 'N2', 'N3', 'REM'] if s in vis_data['Stage'].unique()]
    if 'N4' in stage_mapping_rev.values() and 'N4' not in stage_order and 'N3' in stage_order: pass
    elif 'N4' in vis_data['Stage'].unique():
        stage_order.insert(stage_order.index('N3')+1 if 'N3' in stage_order else len(stage_order), 'N4')
    print(f"  Data prepared for plotting (shape: {vis_data.shape}). Stage order: {stage_order}")
    
    # Add necessary imports if they aren't global
    import matplotlib.pyplot as plt
    import seaborn as sns 
    import numpy as np # Ensure numpy is available

    # --- 2. Band Power Distribution by Stage ---
    # (Identical to previous visualization function)
    print("  Generating Band Power Distribution plot...")
    band_power_cols = [col for col in vis_data.columns if '_AbsPwr' in col and 'Total' not in col]
    if band_power_cols:
        n_bands = len(band_power_cols)
        n_cols_plot = 3
        n_rows_plot = (n_bands + n_cols_plot - 1) // n_cols_plot
        fig_bp, axes_bp = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 5 * n_rows_plot), sharey=True)
        axes_bp = axes_bp.flatten()
        for i, bp_col in enumerate(band_power_cols):
            band_name = bp_col.replace('_AbsPwr', '')
            log_power = np.log10(vis_data[bp_col].clip(1e-9))
            sns.boxplot(ax=axes_bp[i], x='Stage', y=log_power, data=vis_data, order=stage_order, showfliers=False)
            axes_bp[i].set_title(f'{band_name} Power Distribution')
            axes_bp[i].set_ylabel('Log10(Absolute Power)')
            axes_bp[i].set_xlabel('Sleep Stage')
            axes_bp[i].tick_params(axis='x', rotation=45)
        for j in range(i + 1, len(axes_bp)):
             fig_bp.delaxes(axes_bp[j])
        plt.tight_layout()
        plot_path = os.path.join(band_power_dir, 'band_power_distribution_by_stage_nn.png') # Added suffix
        try: plt.savefig(plot_path); print(f"    Saved plot: {plot_path}")
        except Exception as e: print(f"Error saving band power plot: {e}")
        plt.close(fig_bp)
    else:
        print("    Skipping band power plot: No absolute power columns found.")

    # --- 3. Feature Importance (Omitted for NN) ---
    print("  Skipping Feature Importance plot (not directly available for this Keras model).")
    # Consider implementing permutation importance or SHAP if needed, requires extra libraries/computation.

    # --- 4. Confusion Matrix Heatmap ---
    # (Identical to previous visualization function)
    print("  Generating Confusion Matrix heatmap...")
    if 'confusion_matrix' in evaluation_results:
        conf_matrix = np.array(evaluation_results['confusion_matrix'])
        class_names = label_encoder.classes_
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
        fig_cm, ax_cm = plt.subplots(1, 2, figsize=(18, 7))
        fig_cm.suptitle('Confusion Matrix (Neural Network)') # Added NN label
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm[0])
        ax_cm[0].set_title('Counts'); ax_cm[0].set_xlabel('Predicted Label'); ax_cm[0].set_ylabel('True Label')
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm[1])
        ax_cm[1].set_title('Normalized by True Label'); ax_cm[1].set_xlabel('Predicted Label'); ax_cm[1].set_ylabel('True Label')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(conf_matrix_dir, 'confusion_matrix_heatmap_nn.png') # Added suffix
        try: plt.savefig(plot_path); print(f"    Saved plot: {plot_path}")
        except Exception as e: print(f"Error saving confusion matrix plot: {e}")
        plt.close(fig_cm)
    else:
        print("    Skipping confusion matrix plot: 'confusion_matrix' not found.")

    # --- 5. Spindle Density Distribution by Stage ---
    # (Identical to previous visualization function)
    print("  Generating Spindle Density Distribution plot...")
    if 'SpindleDensity' in vis_data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Stage', y='SpindleDensity', data=vis_data, order=stage_order, showfliers=False)
        plt.title('Spindle Density Distribution by Sleep Stage')
        plt.xlabel('Sleep Stage'); plt.ylabel('Spindle Density (spindles/second)')
        plt.xticks(rotation=45); plt.tight_layout()
        plot_path = os.path.join(spindle_dir, 'spindle_density_distribution_by_stage_nn.png') # Added suffix
        try: plt.savefig(plot_path); print(f"    Saved plot: {plot_path}")
        except Exception as e: print(f"Error saving spindle density plot: {e}")
        plt.close()
    else:
        print("    Skipping spindle density plot: 'SpindleDensity' column not found.")

    print("--- Finished Generating Final Visualizations (Neural Network) ---")

def plot_wavelet_decomposition(wavelet_data, channels, sampling_rate, output_dir, 
                              subject_index, eeg_channel_name, epoch_to_plot=None):
    """Generates a plot showing the wavelet decomposition for a single epoch.

    Args:
        wavelet_data (list): Output from wavelet_decomposition.
        channels (dict): Channel mapping dictionary.
        sampling_rate (dict): Sampling rate dictionary.
        output_dir (str): Base directory to save the plot.
        subject_index (int): Index of the subject to plot (0-based).
        eeg_channel_name (str): Name of the EEG channel used for wavelet decomposition.
        epoch_to_plot (int, optional): Index of the epoch to plot (0-based).
                                       If None, the middle epoch is chosen.
    """
    print(f"\nGenerating wavelet decomposition plot for Subject {subject_index + 1}, Channel {eeg_channel_name}...")

    # --- 1. Input Validation and Data Selection ---
    if subject_index >= len(wavelet_data):
        print(f"Error: Subject index {subject_index} is out of bounds.")
        return
        
    subject_wavelet_data = wavelet_data[subject_index]

    if eeg_channel_name not in subject_wavelet_data:
        print(f"Error: EEG channel '{eeg_channel_name}' not found in wavelet data for Subject {subject_index + 1}.")
        return
        
    channel_data = subject_wavelet_data[eeg_channel_name]
    bands_available = list(channel_data.keys())
    bands_to_plot = ['Original', 'Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']

    if 'Original' not in channel_data or not channel_data['Original']:
        print(f"Error: No 'Original' epoch data found for {eeg_channel_name}. Cannot plot.")
        return

    num_epochs = len(channel_data['Original'])
    if num_epochs == 0:
        print(f"Warning: No epochs found for channel '{eeg_channel_name}'. Cannot generate plot.")
        return

    # Choose epoch to plot
    if epoch_to_plot is None:
        epoch_to_plot = num_epochs // 2 # Default to middle epoch
    elif epoch_to_plot >= num_epochs:
        print(f"Warning: Requested epoch {epoch_to_plot} out of bounds (0-{num_epochs-1}). Plotting epoch 0 instead.")
        epoch_to_plot = 0
        
    print(f"  Plotting epoch index: {epoch_to_plot}")

    # Find channel key for sampling rate
    channel_key = None
    for key, name in channels.items():
        if name == eeg_channel_name:
            channel_key = key
            break
    if channel_key is None: 
        print(f"Warning: Could not find channel key for '{eeg_channel_name}'. Cannot determine time axis accurately.")
        fs = 1 # Assign dummy sampling rate
    else:
        fs = sampling_rate[channel_key]

    # --- 2. Prepare Data for Plotting --- 
    epoch_signals = {}
    max_samples = 0
    for band_name in bands_to_plot:
        if band_name in channel_data and len(channel_data[band_name]) > epoch_to_plot:
            signal = channel_data[band_name][epoch_to_plot]
            epoch_signals[band_name] = signal
            if len(signal) > max_samples:
                max_samples = len(signal)
        else:
            print(f"Warning: Data for band '{band_name}' not found for epoch {epoch_to_plot}. Skipping band.")
            # Remove band from plotting list if data is missing
            if band_name in bands_to_plot: 
                bands_to_plot.remove(band_name)
                
    if not epoch_signals or 'Original' not in epoch_signals:
        print("Error: Could not retrieve any band data for the selected epoch. Cannot plot.")
        return
        
    # --- 3. Plotting --- 
    n_bands_to_plot = len(epoch_signals)
    fig, axes = plt.subplots(n_bands_to_plot, 1, figsize=(12, 2 * n_bands_to_plot), sharex=True)
    fig.suptitle(f'Subject {subject_index + 1} - {eeg_channel_name} - Wavelet Decomposition (Epoch {epoch_to_plot})')

    if n_bands_to_plot == 1:
        axes = [axes] # Ensure axes is iterable even with one subplot

    plot_idx = 0
    for band_name in bands_to_plot: # Iterate in the defined order
        if band_name in epoch_signals:
            signal = epoch_signals[band_name]
            current_time_axis = np.arange(len(signal)) / fs # Use actual length for this band
            axes[plot_idx].plot(current_time_axis, signal)
            axes[plot_idx].set_title(band_name)
            axes[plot_idx].set_ylabel('Amplitude')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
            plot_idx += 1

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # --- 4. Save Plot --- 
    plot_output_dir = os.path.join(output_dir, 'Figures', 'WaveletDecomposition')
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filename = os.path.join(plot_output_dir, f'subject_{subject_index + 1}_{eeg_channel_name}_epoch_{epoch_to_plot}_wavelets.png')
    
    try:
        plt.savefig(plot_filename)
        print(f"  Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory
