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
from scipy import signal, stats, integrate
from scipy.integrate import simpson
import mne
from mne.datasets.sleep_physionet.age import fetch_data
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
import nolds # Optional: for fractal dimension, Lyapunov exponent etc. pip install nolds
from pywt import wavedec # If you want wavelet-based features beyond just denoising
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.utils import class_weight # Import this

OUTPUT_FIGURES_DIR = 'Output_Figures'
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
            channel_names_to_extract = list(channels_dict) # channel_names_to_extract = list(channels_dict.values())

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


def extract_data(id, CHANNELS, FS):
    patient_data, patient_xml = load_patient_data(id, 'Input',CHANNELS, 'Output')

    patient_data = patient_data[0]
    patient_stage = patient_xml[0]['stages']

    data_length = len(patient_data['EEG'])
    print(f'Patient data is {data_length} samples long with a sampling rate of {FS} Hz')
    print(f'Patient stage is {len(patient_stage)} samples long with a sampling rate of 1 Hz')

    #raw.set_channel_types({'ECG': 'ecg', 'EEG(sec)': 'eeg', 'EMG': 'emg', 'EOG(L)': 'eog', 'EOG(R)': 'eog'})

    # align the data length to the stage length by removing the last 30 seconds of data
    for channel in patient_data:
        patient_data[channel] = patient_data[channel][:-30*FS]
        
    TIME = np.arange(0, len(patient_data['EEG'])//125, 1/125)
    print(TIME)
    print(len(TIME))

    # Reoving start and finish
    SECONDS_TO_REMOVE = 30 * 60 # 30 minutes in seconds
    SAMPLES_TO_REMOVE = SECONDS_TO_REMOVE * FS

    # Remove the first and last 30 min of data 
    for channel in patient_data:
        patient_data[channel] = patient_data[channel][SAMPLES_TO_REMOVE:-SAMPLES_TO_REMOVE]
    patient_stage = patient_stage[SECONDS_TO_REMOVE:-SECONDS_TO_REMOVE]

    print(len(patient_data['EEG'])//125)
    print(len(patient_stage))
    return patient_data,patient_stage,TIME

def filter_data(patient_data, FS, OUTPUT_FIGURES_DIR):
        # Simple filters
        filtered_data = {}
        for channel in patient_data:
            # Band pass filter
            temp_filtered = mne.filter.filter_data(patient_data[channel], FS, 0.5, 62.0, fir_design='firwin', verbose=False) # Max freq < Fs/2
            # Notch filter
            filtered_data[channel] = mne.filter.notch_filter(temp_filtered, FS, 60, method='fir', fir_design='firwin', verbose=False)

        # Plot patient data and filtered data
        plt.figure(figsize=(15, 5))
        plt.subplot(2, 1, 1)
        plt.plot(patient_data['EEG'], label='Patient Data')
        plt.subplot(2, 1, 2)
        plt.plot(filtered_data['EEG'], label='Filtered Data')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'patient_data_vs_filtered.png'))
        plt.close()
        ########################################################

        # Plot the psd of the patient data and filtered data
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.psd(patient_data['EEG'], 2**10, Fs=FS)
        plt.subplot(2, 1, 2)
        plt.psd(filtered_data['EEG'], 2**10, Fs=FS)
        plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'psd_comparison.png'))
        plt.close()
        ########################################################
        
        return filtered_data
    
def adaptive_filtering(filtered_data, FS, OUTPUT_FIGURES_DIR):
    from scipy.signal import lfilter, butter
    import neurokit2 as nk
    import padasip as pa

    # --- Configuration ---
    N_EOG_FILTER = 25  # Order for EOG filter
    MU_EOG = 0.005     # Learning rate for EOG NLMS
    N_ECG_FILTER = 35  # Order for ECG filter (often needs to be higher than EOG)
    MU_ECG = 0.001     # Learning rate for ECG NLMS
    EEG_CHANNEL_NAME = 'EEG' # Or 'EEG(sec)' or loop through a list
    EOG_L_NAME = 'EOG(L)'
    EOG_R_NAME = 'EOG(R)'
    ECG_NAME = 'ECG'

    # --- Create a new dictionary for adaptively filtered data ---
    adaptively_filtered_data_out = filtered_data.copy() # Work on a copy

    # --- Get Signals ---
    # Make sure these channels exist in filtered_data
    if EEG_CHANNEL_NAME not in filtered_data or \
    EOG_L_NAME not in filtered_data or \
    EOG_R_NAME not in filtered_data or \
    ECG_NAME not in filtered_data:
        print("Error: One or more required channels not found in filtered_data. Skipping adaptive filtering.")
        # return adaptively_filtered_data_out # Or handle error as appropriate
    else:
        eeg_signal_current = filtered_data[EEG_CHANNEL_NAME].copy() # Start with the simply filtered EEG
        eog_l_ref = filtered_data[EOG_L_NAME]
        eog_r_ref = filtered_data[EOG_R_NAME]
        ecg_ref_signal = filtered_data[ECG_NAME]

        # --- 1. EOG Artifact Removal using Differential EOG ---
        print(f"Applying EOG filtering to {EEG_CHANNEL_NAME}...")
        eog_diff_ref = eog_l_ref - eog_r_ref

        if len(eog_diff_ref) < N_EOG_FILTER or len(eeg_signal_current) < N_EOG_FILTER :
            print(f"  Signal too short for EOG filter order {N_EOG_FILTER}. Skipping EOG filtering.")
        else:
            # Prepare inputs for padasip
            # x_eog_history will be (len(eog_diff_ref) - N_EOG_FILTER + 1, N_EOG_FILTER)
            x_eog_history = pa.input_from_history(eog_diff_ref, N_EOG_FILTER)
            # d_eeg_target_eog must align with x_eog_history
            d_eeg_target_eog = eeg_signal_current[N_EOG_FILTER-1:]

            # Initialize and run EOG filter
            f_eog = pa.filters.AdaptiveFilter(model="NLMS", n=N_EOG_FILTER, mu=MU_EOG, w="random")
            y_predicted_eog_artifact, e_eeg_cleaned_from_eog, w_eog = f_eog.run(d_eeg_target_eog, x_eog_history)

            # Update the current EEG signal:
            # The first N_EOG_FILTER-1 samples of eeg_signal_current remain as they were.
            # The rest are replaced with the EOG-cleaned segment.
            eeg_signal_current[N_EOG_FILTER-1:] = e_eeg_cleaned_from_eog
            print(f"  EOG filtering applied. EEG length: {len(eeg_signal_current)}")

        # At this point, eeg_signal_current contains the EOG-cleaned EEG signal
        # (or original if EOG filtering was skipped due to length)

        # --- 2. ECG Artifact Removal (on EOG-cleaned EEG) ---
        print(f"Applying ECG filtering to {EEG_CHANNEL_NAME} (post-EOG)...")
        if len(ecg_ref_signal) < N_ECG_FILTER or len(eeg_signal_current) < N_ECG_FILTER:
            print(f"  Signal too short for ECG filter order {N_ECG_FILTER}. Skipping ECG filtering.")
        else:
            # Prepare inputs for padasip
            x_ecg_history = pa.input_from_history(ecg_ref_signal, N_ECG_FILTER)
            # d_eeg_target_ecg is the EOG-cleaned EEG, aligned for ECG filtering
            d_eeg_target_ecg = eeg_signal_current[N_ECG_FILTER-1:]

            # Initialize and run ECG filter
            f_ecg = pa.filters.AdaptiveFilter(model="NLMS", n=N_ECG_FILTER, mu=MU_ECG, w="random")
            y_predicted_ecg_artifact, e_eeg_cleaned_from_ecg, w_ecg = f_ecg.run(d_eeg_target_ecg, x_ecg_history)

            # Update the current EEG signal (which was already EOG-cleaned):
            # The first N_ECG_FILTER-1 samples of eeg_signal_current remain as they were (EOG cleaned).
            # The rest are replaced with the ECG-cleaned (and EOG-cleaned) segment.
            eeg_signal_current[N_ECG_FILTER-1:] = e_eeg_cleaned_from_ecg
            print(f"  ECG filtering applied. EEG length: {len(eeg_signal_current)}")

        # --- Store the fully processed EEG signal ---
        adaptively_filtered_data_out[EEG_CHANNEL_NAME] = eeg_signal_current

        # --- Plotting (Optional Example for 'EEG' channel) ---
        if EEG_CHANNEL_NAME == 'EEG': # Example plot condition
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            plot_len = min(len(filtered_data[EEG_CHANNEL_NAME]), 10 * FS) # Plot 10 seconds
            time_axis = np.arange(plot_len) / FS

            axs[0].plot(time_axis, filtered_data[EEG_CHANNEL_NAME][:plot_len], label='Original Filtered EEG')
            axs[0].set_title('Original Filtered EEG')
            axs[0].legend()

            # Need to reconstruct the EOG-cleaned signal for plotting if not saved separately
            # For simplicity, we plot the final output here:
            # To plot intermediate EOG-cleaned, you'd save 'eeg_signal_current' after EOG step.

            axs[1].plot(time_axis, adaptively_filtered_data_out[EEG_CHANNEL_NAME][:plot_len], label='EEG after EOG & ECG cleaning')
            axs[1].set_title('EEG after EOG & ECG Adaptive Cleaning')
            axs[1].legend()
            
            # Plot one of the artifacts removed
            if len(eog_diff_ref) >= N_EOG_FILTER and len(filtered_data[EEG_CHANNEL_NAME]) >= N_EOG_FILTER :
                # Re-run EOG filter just for plotting the artifact (if not stored)
                _x_eog_hist = pa.input_from_history(eog_diff_ref, N_EOG_FILTER)
                _d_eeg_target_eog = filtered_data[EEG_CHANNEL_NAME][N_EOG_FILTER-1:]
                _f_eog = pa.filters.AdaptiveFilter(model="NLMS", n=N_EOG_FILTER, mu=MU_EOG, w="random")
                _y_predicted_eog_artifact, _, _ = _f_eog.run(_d_eeg_target_eog, _x_eog_hist)
                
                # Pad y_predicted_eog_artifact to align with original EEG for plotting
                padded_eog_artifact = np.zeros_like(filtered_data[EEG_CHANNEL_NAME])
                padded_eog_artifact[N_EOG_FILTER-1:N_EOG_FILTER-1+len(_y_predicted_eog_artifact)] = _y_predicted_eog_artifact

                axs[2].plot(time_axis, padded_eog_artifact[:plot_len], label='Predicted EOG Artifact (example)', color='green')
                axs[2].set_title('Predicted EOG Artifact Component')
                axs[2].legend()

            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'adaptive_filtering_results.png'))
            plt.close()
        
    return adaptively_filtered_data_out
    
def wavelet_denoising(filtered_data, FS, OUTPUT_FIGURES_DIR, TIME):
        import pywt
        import numpy as np

        # --- EEG signal to be filtered ---
        wavelet_cleaned_data = filtered_data.copy()
        signal_to_process = wavelet_cleaned_data['EEG']

        # --- Wavelet Denoising Parameters ---
        wavelet = 'db4'  # Daubechies 4 is a common choice for biomedical signals.
        level = 4        # Decomposition level. This should be chosen based on the signal's sampling rate
                        # and the frequency bands of interest. For Fs=125Hz, level 4 means:
                        # cA_level, cD_level, cD_level-1, ..., cD1
                        # cD1 corresponds to the highest frequencies.
                        # For Fs=125Hz:
                        # cD1: 31.25 - 62.5 Hz
                        # cD2: 15.625 - 31.25 Hz
                        # cD3: 7.8125 - 15.625 Hz
                        # cD4: 3.90625 - 7.8125 Hz
                        # cA4: 0 - 3.90625 Hz
                        # Max level can be determined by: pywt.dwt_max_level(len(signal_to_process), pywt.Wavelet(wavelet))

        # --- 1. Wavelet Decomposition ---
        # Decompose the signal into wavelet coefficients
        coeffs = pywt.wavedec(signal_to_process, wavelet, level=level)
        # coeffs is a list of arrays: [cA_level, cD_level, cD_level-1, ..., cD1]

        # --- 2. Thresholding Detail Coefficients ---
        # We will apply a threshold to the detail coefficients (cD_level down to cD1).
        # The approximation coefficients (cA_level) are usually kept.
        thresholded_coeffs = [coeffs[0]]  # Keep the approximation coefficients

        # Iterate through the detail coefficient sets (from cD_level to cD1)
        for i in range(1, len(coeffs)):
            detail_coeff_set = coeffs[i]
            
            # Estimate noise standard deviation (sigma) using Median Absolute Deviation (MAD)
            # This is a robust estimator, less sensitive to outliers than standard deviation.
            # sigma_hat = MAD / 0.6745 (0.6745 is the Gaussian equivalent scaling factor)
            median_val = np.median(detail_coeff_set)
            mad = np.median(np.abs(detail_coeff_set - median_val))
            sigma = mad / 0.6745
            
            # Calculate Universal Threshold (VisuShrink by Donoho & Johnstone)
            # T = sigma * sqrt(2 * log(N)), where N is the length of the signal
            N = len(signal_to_process)
            threshold_value = sigma * np.sqrt(2 * np.log(N)) if N > 1 and sigma > 0 else 0
            # If sigma is 0 (e.g., coefficients are constant), threshold is 0 (no thresholding).
            
            # Apply soft thresholding
            # Soft thresholding: shrinks coefficients towards zero.
            # y = sgn(x) * max(0, |x| - T)
            thresholded_detail_set = pywt.threshold(detail_coeff_set, value=threshold_value, mode='soft')
            thresholded_coeffs.append(thresholded_detail_set)

        # --- 3. Wavelet Reconstruction ---
        # Reconstruct the signal from the (mostly thresholded) coefficients
        eeg_wavelet_denoised = pywt.waverec(thresholded_coeffs, wavelet)

        # Ensure the reconstructed signal has the same length as the original input signal.
        # Wavelet reconstruction can sometimes result in a length mismatch of one sample.
        if len(eeg_wavelet_denoised) != len(signal_to_process):
            eeg_wavelet_denoised = eeg_wavelet_denoised[:len(signal_to_process)]

        print("Wavelet denoising process complete.")

        # plot the original and filtered signal
        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.plot(TIME[:FS*30],signal_to_process[:FS*30])
        plt.subplot(2,1,2)
        plt.plot(TIME[:FS*30], eeg_wavelet_denoised[:FS*30])
        plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'wavelet_denoising_comparison.png'))
        plt.close()
        
        return wavelet_cleaned_data

def create_epochs(filtered_data, patient_stage, FS, SAMPLES_PER_EPOCH, CHANNELS):
        NB_EPOCHS = len(filtered_data['EEG'])//SAMPLES_PER_EPOCH

        # Creat epochs of 30 seconds
        patient_epoch = {} # Initialize as a dictionary

        # Process each channel
        for channel_name in CHANNELS:
            # Get the channel data for this patient
            channel_data = filtered_data[channel_name]

            # Create a list to store all epochs for this channel
            channel_epochs = []

            # Split the data into epochs
            for epoch_idx in range(NB_EPOCHS):
                start_idx = epoch_idx * SAMPLES_PER_EPOCH
                end_idx = start_idx + SAMPLES_PER_EPOCH
                epoch_data = channel_data[start_idx:end_idx]
                channel_epochs.append(epoch_data)

            # Store all epochs for this channel in the patient's dictionary
            patient_epoch[channel_name] = channel_epochs

            print(f"  - Channel {channel_name}: Created {len(channel_epochs)} epochs of length {SAMPLES_PER_EPOCH}")

        # In the same data structure, store the epoch of the hypnogram
        patient_epoch['hypnogram'] = []
        # Split the data into epochs
        for epoch_idx in range(NB_EPOCHS):
            start_idx = epoch_idx * 30
            end_idx = start_idx + 30
            epoch_data = patient_stage[start_idx:end_idx]
            # AASM scoring rules typically use a majority rule for the 30-second epoch
            # Use scipy.stats.mode with keepdims=False to get the mode value directly
            dominant_stage = stats.mode(epoch_data, keepdims=False)[0]
            # Ensure the stage is one of the valid classes (0, 2, 3, 4, 5)
            if dominant_stage not in [0, 1, 2, 3, 4, 5]:
                print(f"Warning: Invalid stage {dominant_stage} found at epoch {epoch_idx}")
                # You might want to handle this case differently
                # For now, we'll keep it as is
            patient_epoch['hypnogram'].append(dominant_stage)
                
        # Print class distribution
        unique_stages, stage_counts = np.unique(patient_epoch['hypnogram'], return_counts=True)
        print("\nStage distribution in hypnogram:")
        for stage, count in zip(unique_stages, stage_counts):
            print(f"Stage {stage}: {count} epochs")

        return patient_epoch

def approximate_entropy(signal, m=2, r=0.2):
    """
    Calculate approximate entropy of a time series.
    
    Args:
        signal (numpy.ndarray): Input time series
        m (int): Embedding dimension
        r (float): Threshold (usually 0.2 * std of signal)
    
    Returns:
        float: Approximate entropy value
    """
    N = len(signal)
    r = r * np.std(signal)
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = [[signal[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C) / (N - m + 1.0)
    
    return abs(_phi(m+1) - _phi(m))

def extract_eeg_features(eeg_data, sampling_freq):
    """
    Extract features from EEG data for a single epoch.
    
    Args:
        eeg_data (numpy.ndarray): EEG data for a single epoch
        sampling_freq (int): Sampling frequency of the EEG data
        
    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}
    
    # Time domain features
    features['mean'] = np.mean(eeg_data)
    features['std'] = np.std(eeg_data)
    features['variance'] = np.var(eeg_data)
    features['skewness'] = stats.skew(eeg_data)
    features['kurtosis'] = stats.kurtosis(eeg_data)
    features['zero_crossings'] = np.sum(np.diff(np.signbit(eeg_data)))
    
    # Frequency domain features
    freqs, psd = signal.welch(eeg_data, sampling_freq, nperseg=min(256, len(eeg_data)))
    
    # Band power features
    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    theta_mask = (freqs >= 4) & (freqs <= 8)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    gamma_mask = (freqs >= 30) & (freqs <= 50)
    
    features['delta_power'] = simpson(psd[delta_mask], freqs[delta_mask])
    features['theta_power'] = simpson(psd[theta_mask], freqs[theta_mask])
    features['alpha_power'] = simpson(psd[alpha_mask], freqs[alpha_mask])
    features['beta_power'] = simpson(psd[beta_mask], freqs[beta_mask])
    features['gamma_power'] = simpson(psd[gamma_mask], freqs[gamma_mask])
    
    # Total power
    total_power = features['delta_power'] + features['theta_power'] + features['alpha_power'] + features['beta_power'] + features['gamma_power']
    
    # Relative power features
    features['delta_rel_power'] = features['delta_power'] / total_power
    features['theta_rel_power'] = features['theta_power'] / total_power
    features['alpha_rel_power'] = features['alpha_power'] / total_power
    features['beta_rel_power'] = features['beta_power'] / total_power
    features['gamma_rel_power'] = features['gamma_power'] / total_power
    
    # Spectral edge frequency (95%)
    cumsum = np.cumsum(psd)
    cumsum = cumsum / cumsum[-1]
    features['spectral_edge_freq'] = freqs[np.where(cumsum >= 0.95)[0][0]]
    
    # Spectral entropy
    psd_norm = psd / np.sum(psd)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # Wavelet features
    coeffs = wavedec(eeg_data, 'db4', level=4)
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_coeff_{i}_mean'] = np.mean(coeff)
        features[f'wavelet_coeff_{i}_std'] = np.std(coeff)
        features[f'wavelet_coeff_{i}_energy'] = np.sum(coeff ** 2)
    
    # Nonlinear features
    # features['approximate_entropy'] = approximate_entropy(eeg_data)
    features['sample_entropy'] = nolds.sampen(eeg_data, emb_dim=2)
    features['hurst_exponent'] = nolds.hurst_rs(eeg_data)
    
    return features

def create_feature_dataframe(patient_epoch_dict, eeg_channel_name, sampling_freq, feature_filename=None, append_mode=False):
    """
    Creates a DataFrame with extracted features and hypnogram.
    Can append features for multiple patients if append_mode is True.
    
    Args:
        patient_epoch_dict (dict): Dictionary containing EEG epochs and hypnogram
        eeg_channel_name (str): Name of the EEG channel to process
        sampling_freq (int): Sampling frequency of the EEG data
        feature_filename (str, optional): If provided, will try to load features from this file first
        append_mode (bool): If True, will append features to existing DataFrame instead of creating new one
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features and hypnogram
    """
    if feature_filename and os.path.exists(feature_filename) and not append_mode:
        print(f"Feature file '{feature_filename}' found. Loading features from file.")
        try:
            df_final_features = pd.read_csv(feature_filename)
            if not df_final_features.empty:
                print("Loaded features:")
                print(df_final_features.head())
                return df_final_features
            else:
                print(f"Warning: Loaded feature file '{feature_filename}' is empty.")
        except Exception as e:
            print(f"Error loading {feature_filename}: {e}. Proceeding with feature extraction.")

    all_epochs_features = []
    eeg_epochs_list = patient_epoch_dict[eeg_channel_name]
    hypnogram_list = patient_epoch_dict['hypnogram']

    # Ensure lengths match, truncate to shorter if they don't (simplification)
    min_len = min(len(eeg_epochs_list), len(hypnogram_list))
    eeg_epochs_list = eeg_epochs_list[:min_len]
    hypnogram_list = hypnogram_list[:min_len]
    
    print(f"Processing {min_len} epochs for channel {eeg_channel_name}...")

    for i, eeg_epoch_data in enumerate(eeg_epochs_list):
        if (i + 1) % 100 == 0 or i == min_len -1:
            print(f"  Extracting features for epoch {i+1}/{min_len}")
        try:
            epoch_features = extract_eeg_features(eeg_epoch_data, sampling_freq)
            all_epochs_features.append(epoch_features)
        except Exception as e:
            print(f"    Error in epoch {i+1} for channel {eeg_channel_name}: {e}. Appending NaNs.")
            # Fallback: create a NaN dict based on keys from a successful prior extraction or a dummy one
            if all_epochs_features:
                nan_features = {key: np.nan for key in all_epochs_features[0].keys()}
            else:
                try:
                    dummy_data = np.random.rand(len(eeg_epoch_data) if eeg_epoch_data is not None and len(eeg_epoch_data) > 0 else 3750) # Default length
                    nan_features = {key: np.nan for key in extract_eeg_features(dummy_data, sampling_freq).keys()}
                except: # Absolute fallback if even dummy extraction fails
                    print("    Critical error: Could not determine feature keys for NaN padding. This epoch's features will be missing or DataFrame might error.")
                    continue # Skip this problematic epoch entirely
            all_epochs_features.append(nan_features)

    if not all_epochs_features:
        print("No features extracted. Returning empty DataFrame.")
        return pd.DataFrame()

    df_features_only = pd.DataFrame(all_epochs_features)
    
    # Ensure hypnogram_list is aligned with the number of feature sets actually generated
    final_hypnogram_list = hypnogram_list[:len(df_features_only)]

    df_sleep_features = pd.concat([df_features_only, pd.Series(final_hypnogram_list, name='Hypnogram_Stage', index=df_features_only.index)], axis=1)
    df_sleep_features.insert(0, 'Epoch_Number', np.arange(len(df_sleep_features)))
    
    print("\n--- Sleep DataFrame with Extracted EEG Features ---")
    print(f"DataFrame Shape: {df_sleep_features.shape}")
    
    nan_counts = df_sleep_features.isnull().sum()
    if nan_counts.sum() > 0:
        print("\nNaN values found in features:")
        print(nan_counts[nan_counts > 0])
    
    if feature_filename:
        if append_mode and os.path.exists(feature_filename):
            # Append to existing file
            df_sleep_features.to_csv(feature_filename, mode='a', header=False, index=False)
            print(f"Features appended to {feature_filename}")
        else:
            # Create new file or overwrite existing
            df_sleep_features.to_csv(feature_filename, index=False)
            print(f"Features saved to {feature_filename}")
    
    return df_sleep_features

def prepare_data_for_models(df_features, target_column='Hypnogram_Stage', test_size=0.2, random_state=42, scale_features=True, for_cnn_sequence=False, raw_eeg_epochs=None):
    """
    Prepares data for XGBoost and a Keras DL model.
    """
    # Drop rows with NaN values that might have resulted from feature extraction
    df_clean = df_features.dropna()
    if len(df_clean) < len(df_features):
        print(f"Dropped {len(df_features) - len(df_clean)} rows with NaN values.")
        # If using raw_eeg_epochs, they need to be aligned with df_clean
        if raw_eeg_epochs is not None:
            raw_eeg_epochs = [epoch for i, epoch in enumerate(raw_eeg_epochs) if i in df_clean.index]

    # Print class distribution before splitting
    print("\nClass distribution in full dataset:")
    class_counts = df_clean[target_column].value_counts().sort_index()
    print(class_counts)

    y = df_clean[target_column]
    X = df_clean.drop(columns=[target_column, 'Epoch_Number'], errors='ignore')

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"\nClasses found by LabelEncoder: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")

    if for_cnn_sequence:
        if raw_eeg_epochs is None:
            raise ValueError("`raw_eeg_epochs` must be provided when `for_cnn_sequence` is True.")
        
        X_seq = np.array(raw_eeg_epochs)
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, axis=-1)
        
        X_train_seq, X_test_seq, y_train_enc, y_test_enc = train_test_split(
            X_seq, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Print class distribution after splitting
        print("\nClass distribution in training set:")
        print(pd.Series(y_train_enc).value_counts().sort_index())
        print("\nClass distribution in test set:")
        print(pd.Series(y_test_enc).value_counts().sort_index())

        y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
        
        print(f"X_train_seq shape: {X_train_seq.shape}, y_train_cat shape: {y_train_cat.shape}")
        return X_train_seq, X_test_seq, y_train_cat, y_test_cat, label_encoder, num_classes

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Print class distribution after splitting
        print("\nClass distribution in training set:")
        print(pd.Series(y_train).value_counts().sort_index())
        print("\nClass distribution in test set:")
        print(pd.Series(y_test).value_counts().sort_index())

        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print("Features scaled using StandardScaler.")
        
        print(f"X_train (features) shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test, label_encoder, num_classes


def evaluate_model(y_true, y_pred_encoded, label_encoder, model_name="Model"):
    """Prints classification report and displays confusion matrix."""
    print(f"\n--- Evaluation Report for {model_name} ---")
    
    # Print diagnostic information
    print("Unique classes in y_true:", np.unique(y_true))
    print("Unique classes in y_pred:", np.unique(y_pred_encoded))
    
    # Get all possible classes (0, 2, 3, 4, 5)
    all_possible_classes = np.array([0, 2, 3, 4, 5])
    class_names = all_possible_classes.astype(str)
    
    # Ensure y_true and y_pred_encoded are numpy arrays
    y_true = np.array(y_true)
    y_pred_encoded = np.array(y_pred_encoded)
    
    # Use labels parameter to specify all possible classes
    try:
        report = classification_report(y_true, y_pred_encoded, 
                                     target_names=class_names,
                                     labels=all_possible_classes,
                                     zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error in classification_report: {e}")
        print("Falling back to basic metrics...")
        # Calculate basic metrics manually
        accuracy = accuracy_score(y_true, y_pred_encoded)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy, 0
    
    accuracy = accuracy_score(y_true, y_pred_encoded)
    kappa = cohen_kappa_score(y_true, y_pred_encoded)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    try:
        # Create confusion matrix with all possible classes
        cm = confusion_matrix(y_true, y_pred_encoded, labels=all_possible_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    
    return accuracy, kappa

def train_xgboost_classifier(X_train, y_train, X_test, y_test, label_encoder, num_classes):
    """
    Trains an XGBoost classifier, attempting to use early_stopping_rounds
    in the constructor (common for older XGBoost versions).
    """
    print("\n--- Training XGBoost Classifier (Attempting older API for early stopping) ---")
    
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    y_train = np.ascontiguousarray(y_train)
    y_test = np.ascontiguousarray(y_test)

    model_xgb = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        n_estimators=200,          # Max number of boosting rounds.
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',     # Often specified here in older versions for early stopping
        early_stopping_rounds=20,   # Early stopping parameter in constructor
        use_label_encoder=False     # Try keeping this; remove if it causes a new error
    )
    
    # Fit the model. eval_set is crucial for early stopping to work.
    model_xgb.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], # Evaluation data for early stopping
                  verbose=False) # Suppress training progress output

    # After fitting with early_stopping_rounds in the constructor,
    # the model should have used the best number of trees.
    
    y_pred_xgb = model_xgb.predict(X_test)
    
    print("XGBoost training complete.")
    accuracy, kappa = evaluate_model(y_test, y_pred_xgb, label_encoder, "XGBoost (Older API)")
    
    return model_xgb, accuracy, kappa

def train_1d_cnn_on_features(X_train, y_train, X_test, y_test, label_encoder, num_classes, input_dim):
    """
    Trains a 1D CNN on extracted features, WITH CLASS WEIGHTING.
    """
    print("\n--- Training 1D CNN on Extracted Features (with Class Weights) ---")

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Calculate class weights
    # y_train is the label-encoded training target (e.g., [0, 1, 2, 1, 0, ...])
    # These are the integer labels the model is trying to predict.
    
    # Get the unique classes *as they appear in y_train*
    # (which are the encoded versions from LabelEncoder, e.g., 0, 1, 2, 3, 4)
    unique_encoded_labels = np.unique(y_train) 
    
    class_weights_calculated = class_weight.compute_class_weight(
        class_weight='balanced',                # Strategy to balance classes
        classes=unique_encoded_labels,          # The unique encoded classes actually in y_train
        y=y_train                               # The training labels (encoded)
    )
    
    # Keras expects class_weight as a dictionary mapping class indices (0 to num_classes-1) to weights.
    # The `unique_encoded_labels` should directly map to the indices if LabelEncoder was used.
    class_weights_dict = {label: weight for label, weight in zip(unique_encoded_labels, class_weights_calculated)}
    
    print(f"Original classes found by LabelEncoder: {label_encoder.classes_}")
    print(f"Unique encoded labels in y_train for class_weight: {unique_encoded_labels}")
    print(f"Calculated class weights array: {class_weights_calculated}")
    print(f"Class weights dictionary to be used by Keras: {class_weights_dict}")


    model_cnn = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1), padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Output layer
    ])

    model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']) 
    
    model_cnn.summary() # This will be printed by your script already

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001, verbose=1)
    ]

    history = model_cnn.fit(X_train_cnn, y_train_cat,
                            epochs=100, 
                            batch_size=64,
                            validation_data=(X_test_cnn, y_test_cat),
                            callbacks=callbacks,
                            class_weight=class_weights_dict, # <<<--- ADDED THIS LINE
                            verbose=1)

    loss, acc_metric = model_cnn.evaluate(X_test_cnn, y_test_cat, verbose=0) 
    print(f"1D CNN (on features) Test Loss: {loss:.4f}, Test Accuracy: {acc_metric:.4f}")

    y_pred_proba_cnn = model_cnn.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1) 

    # y_test is already label-encoded (not one-hot) from prepare_data_for_models
    accuracy, kappa = evaluate_model(y_test, y_pred_cnn, label_encoder, "1D CNN (on Features with Class Weights)")

    return model_cnn, history, accuracy, kappa

def train_1d_cnn_on_sequences(X_train_seq, y_train_cat, X_test_seq, y_test_cat, label_encoder, num_classes, sequence_length, num_channels=1):
    """
    Trains a 1D CNN directly on EEG sequences.

    Args:
        X_train_seq, y_train_cat: Training sequences and categorical labels.
        X_test_seq, y_test_cat: Testing sequences and categorical labels.
        label_encoder: Fitted LabelEncoder.
        num_classes (int): Number of classes.
        sequence_length (int): Number of time steps (samples) in each EEG epoch.
        num_channels (int): Number of EEG channels (typically 1 if processing single channel epochs).

    Returns:
        tuple: (trained_model, history, accuracy, kappa_score)
    """
    print("\n--- Training 1D CNN on Raw EEG Sequences ---")

    # X_train_seq and X_test_seq should already be in shape (num_samples, sequence_length, num_channels)
    # y_train_cat and y_test_cat should be one-hot encoded

    model_cnn_seq = Sequential([
        Conv1D(filters=64, kernel_size=50, strides=6, activation='relu', 
               input_shape=(sequence_length, num_channels), padding='same'), # Larger kernel for raw EEG
        BatchNormalization(),
        MaxPooling1D(pool_size=8, strides=2), # Larger pool size
        Dropout(0.4),

        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        # MaxPooling1D(pool_size=2), # Optional second pooling
        # Dropout(0.4),

        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        # MaxPooling1D(pool_size=2),
        # Dropout(0.4),
        
        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'), # Additional layer
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=2), # Final pooling before flatten
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'), # Or more units, e.g., 256 or 512
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    # This architecture is inspired by some successful sleep staging CNNs like DeepSleepNet, but simplified.
    # Kernel sizes, strides, and number of filters are crucial hyperparameters for raw EEG.

    model_cnn_seq.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Potentially smaller LR
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    model_cnn_seq.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1), # More patience for seq models
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    ]

    history = model_cnn_seq.fit(X_train_seq, y_train_cat,
                                epochs=150, # May need more epochs
                                batch_size=128, # Larger batch size often works well for sequence models
                                validation_data=(X_test_seq, y_test_cat),
                                callbacks=callbacks,
                                verbose=1)
    
    loss, acc_metric = model_cnn_seq.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"1D CNN (on sequences) Test Loss: {loss:.4f}, Test Accuracy: {acc_metric:.4f}")

    y_pred_proba_cnn_seq = model_cnn_seq.predict(X_test_seq)
    y_pred_cnn_seq_encoded = np.argmax(y_pred_proba_cnn_seq, axis=1)
    
    # y_test_original_labels are needed for evaluate_model
    y_test_original_labels = np.argmax(y_test_cat, axis=1)


    accuracy, kappa = evaluate_model(y_test_original_labels, y_pred_cnn_seq_encoded, label_encoder, "1D CNN (on Sequences)")

    return model_cnn_seq, history, accuracy, kappa

def train_models(df_final_features, patient_epoch, EEG_CHANNEL_NAME, FS):
    """
    Trains multiple models on the extracted features and raw sequences.
    
    Args:
        df_final_features (pd.DataFrame): DataFrame containing extracted features and hypnogram
        patient_epoch (dict): Dictionary containing EEG epochs and hypnogram
        EEG_CHANNEL_NAME (str): Name of the EEG channel to process
        FS (int): Sampling frequency of the EEG data
    """
    if df_final_features.empty:
        print("No features available for model training. Exiting.")
        return

    print("\n--- Training Models on Extracted Features ---")
    
    # Prepare data for feature-based models
    X_train_feat, X_test_feat, y_train_enc, y_test_enc, le, n_classes = prepare_data_for_models(
        df_final_features, 
        target_column='Hypnogram_Stage'
    )

    # Train XGBoost
    if X_train_feat.size > 0:
        model_xgb, acc_xgb, kappa_xgb = train_xgboost_classifier(
            X_train_feat, y_train_enc, X_test_feat, y_test_enc, le, n_classes
        )
        print(f"XGBoost Final Test Accuracy: {acc_xgb:.4f}, Kappa: {kappa_xgb:.4f}")
    else:
        print("Skipping XGBoost due to empty training data after preparation.")

    # Train 1D CNN on Features
    if X_train_feat.size > 0:
        num_features = X_train_feat.shape[1]
        model_cnn_feat, hist_cnn_feat, acc_cnn_feat, kappa_cnn_feat = train_1d_cnn_on_features(
            X_train_feat, y_train_enc, X_test_feat, y_test_enc, le, n_classes, num_features
        )
        print(f"1D CNN (Features) Final Test Accuracy: {acc_cnn_feat:.4f}, Kappa: {kappa_cnn_feat:.4f}")
    else:
        print("Skipping 1D CNN on Features due to empty training data after preparation.")

    # Train 1D CNN on Raw Sequences
    if EEG_CHANNEL_NAME in patient_epoch:
        print("\nPreparing data for 1D CNN on raw sequences...")
        raw_eegs_for_cnn = patient_epoch[EEG_CHANNEL_NAME]

        X_train_seq, X_test_seq, y_train_cat, y_test_cat, le_seq, n_classes_seq = prepare_data_for_models(
            df_final_features,
            target_column='Hypnogram_Stage',
            for_cnn_sequence=True,
            raw_eeg_epochs=raw_eegs_for_cnn
        )
        
        if X_train_seq.size > 0:
            sequence_len = X_train_seq.shape[1]
            num_raw_channels = X_train_seq.shape[2]
            
            model_cnn_seq, hist_cnn_seq, acc_cnn_seq, kappa_cnn_seq = train_1d_cnn_on_sequences(
                X_train_seq, y_train_cat, X_test_seq, y_test_cat, le_seq, n_classes_seq, 
                sequence_len, num_raw_channels
            )
            print(f"1D CNN (Sequences) Final Test Accuracy: {acc_cnn_seq:.4f}, Kappa: {kappa_cnn_seq:.4f}")
        else:
            print("Skipping 1D CNN on Sequences due to empty training data after preparation.")
    else:
        print(f"\nSkipping 1D CNN on Raw Sequences: Channel '{EEG_CHANNEL_NAME}' not found in patient_epoch.")