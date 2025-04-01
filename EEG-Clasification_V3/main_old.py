import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal as sig
from sklearn.decomposition import FastICA
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import learning_curve
import pickle
import os

def load_and_preprocess_data():
    # Load the .mat file
    mat_data = sio.loadmat('exportData.mat')

    fs_eeg = 125
    epoch_length = 30 #in second
    
    # Create eeg_data with eeg1 and eeg2
    eeg_data = np.zeros((2, len(mat_data['eeg1'][0])))
    eeg_data[0, :] = mat_data['eeg1'][0]
    eeg_data[1, :] = mat_data['eeg2'][0]

    # Create eog_data with eog1 and eog2
    eog_data = np.zeros((2, len(mat_data['eogl'][0])))
    eog_data[0, :] = mat_data['eogl'][0]
    eog_data[1, :] = mat_data['eogr'][0]

    # Create emg_data with one emg channel, "emg"
    emg_data = np.zeros((1, len(mat_data['emg'][0])))
    emg_data[0, :] = mat_data['emg'][0]

    # Create emg_data with one ecg channel, "ecg"
    ecg_data = np.zeros((1, len(mat_data['ecg'][0])))
    ecg_data[0, :] = mat_data['ecg'][0]
    
    return eeg_data, eog_data, emg_data, ecg_data, fs_eeg, epoch_length

def filter_eeg_data(eeg_data, fs_eeg):
    # Filter the eeg_data with a bandpass filter between 0.5 and 45 Hz
    nyquist = fs_eeg / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 45 / nyquist
    b, a = sig.butter(4, [low_cutoff, high_cutoff], btype='band')
    
    # Apply the filter to each EEG channel
    filtered_eeg = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        filtered_eeg[i, :] = sig.filtfilt(b, a, eeg_data[i, :])
    
    return filtered_eeg

def apply_ica(eeg_data, eog_data, emg_data, ecg_data, fs_eeg, epoch_length, output_dir):
    # Combine all channels for ICA
    all_channels = np.vstack([
        eeg_data,      # 2 EEG channels (already filtered)
        eog_data,      # 2 EOG channels 
        emg_data,      # 1 EMG channel
        ecg_data       # 1 ECG channel
    ])
    
    # Transpose for ICA (ICA expects [n_samples, n_features])
    X = all_channels.T
    
    # Apply FastICA to extract 6 components
    ica = FastICA(n_components=6, random_state=42)
    S = ica.fit_transform(X)  # S contains the independent components
    A = ica.mixing_  # A is the mixing matrix
    
    # Get the independent components (transposed back to [n_components, n_samples])
    components = S.T
    
    # Plot the 6 ICA components
    plt.figure(figsize=(12, 10))
    sample_length = fs_eeg * epoch_length * 5  # Show 5 epochs
    
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(components[i, :sample_length])
        plt.title(f'ICA Component {i+1}')
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'ica_components.png'))
    plt.show()
    
    # Plot contribution of each original channel to each ICA component (mixing matrix)
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(A), aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('ICA Components')
    plt.ylabel('Channels (EEG1, EEG2, EOGL, EOGR, EMG, ECG)')
    plt.title('Mixing Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixing_matrix.png'))
    plt.show()

    # Calculate correlation between each component and each artifact channel
    artifact_channels = np.vstack([eog_data, emg_data, ecg_data])
    correlation_matrix = np.zeros((artifact_channels.shape[0], components.shape[0]))
    
    for i in range(artifact_channels.shape[0]):
        for j in range(components.shape[0]):
            correlation = np.abs(np.corrcoef(artifact_channels[i, :], components[j, :])[0, 1])
            correlation_matrix[i, j] = correlation
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(correlation_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Correlation Coefficient')
    plt.xlabel('ICA Components')
    plt.ylabel('Artifact Channels (EOGL, EOGR, EMG, ECG)')
    plt.title('Correlation between ICA Components and Artifact Channels')
    channel_names = ['EOGL', 'EOGR', 'EMG', 'ECG']
    plt.yticks(np.arange(len(channel_names)), channel_names)
    plt.xticks(np.arange(components.shape[0]), [f'Comp {i+1}' for i in range(components.shape[0])])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.show()
    
    # Identify components to remove (high correlation with artifact channels)
    correlation_threshold = 0.4
    components_to_remove = set()
    
    for i in range(correlation_matrix.shape[0]):
        # Find components with correlation above threshold
        high_corr_idx = np.where(correlation_matrix[i, :] > correlation_threshold)[0]
        if len(high_corr_idx) > 0:
            print(f"Channel {channel_names[i]} is highly correlated with components: {high_corr_idx}")
            components_to_remove.update(high_corr_idx)
    
    components_to_remove = list(components_to_remove)
    components_to_keep = [i for i in range(components.shape[0]) if i not in components_to_remove]
    
    print(f"Components to remove: {components_to_remove}")
    print(f"Components to keep: {components_to_keep}")
    
    # Reconstruct cleaned signal
    # Zero out the artifact components and reconstruct
    S_clean = S.copy()
    for idx in components_to_remove:
        S_clean[:, idx] = 0
    
    X_clean = np.dot(S_clean, ica.mixing_.T) + ica.mean_
    
    # Reshape back to original format [channels, samples]
    cleaned_channels = X_clean.T
    
    # Extract only the first EEG channel which shows better cleaning
    cleaned_eeg = np.zeros((1, cleaned_channels.shape[1]))
    cleaned_eeg[0, :] = cleaned_channels[0, :]  # Keep only the first EEG channel
    
    # Plot original filtered EEG vs cleaned EEG (only for the first channel)
    plt.figure(figsize=(12, 6))
    
    plt.plot(eeg_data[0, :fs_eeg*epoch_length*5], 'b-', label='Filtered')
    plt.plot(cleaned_eeg[0, :fs_eeg*epoch_length*5], 'r-', label='ICA Cleaned')
    plt.title('EEG Channel 1: Filtered vs ICA Cleaned')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cleaned_eeg.png'))
    plt.show()
    
    return cleaned_eeg

def apply_wavelet_transform(cleaned_eeg, fs_eeg, epoch_length, output_dir):
    # Define wavelet and decomposition level
    wavelet = 'db4'
    level = 6  # Decomposition level (changed from 5 to 6)
    
    # Frequency bands (approximate ranges for each level with fs=125Hz)
    # A6: 0-1 Hz (slow delta)
    # D6: 1-2 Hz (delta)
    # D5: 2-4 Hz (delta)
    # D4: 4-8 Hz (theta)
    # D3: 8-16 Hz (alpha + spindles)
    # D2: 16-32 Hz (beta)
    # D1: 32-62.5 Hz (gamma)
    
    # Perform DWT on cleaned EEG data (using only channel, index 0)
    coeffs = pywt.wavedec(cleaned_eeg[0, :], wavelet, level=level)
    
    # Extract coefficients (now we have 7 coefficient arrays - 1 approximation + 6 details)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    # Create zero arrays with the same shape as the coefficients
    zero_cA6 = np.zeros_like(cA6)
    zero_cD6 = np.zeros_like(cD6)
    zero_cD5 = np.zeros_like(cD5)
    zero_cD4 = np.zeros_like(cD4)
    zero_cD3 = np.zeros_like(cD3)
    zero_cD2 = np.zeros_like(cD2)
    zero_cD1 = np.zeros_like(cD1)
    
    # Reconstruct each frequency band separately
    
    # Slow Delta (0-1 Hz): use only cA6
    slow_delta_coeffs = [cA6, zero_cD6, zero_cD5, zero_cD4, zero_cD3, zero_cD2, zero_cD1]
    slow_delta = pywt.waverec(slow_delta_coeffs, wavelet)
    
    # Delta (1-4 Hz): use cD6 and cD5
    delta_coeffs = [zero_cA6, cD6, cD5, zero_cD4, zero_cD3, zero_cD2, zero_cD1]
    delta = pywt.waverec(delta_coeffs, wavelet)
    
    # Theta (4-8 Hz): use only cD4
    theta_coeffs = [zero_cA6, zero_cD6, zero_cD5, cD4, zero_cD3, zero_cD2, zero_cD1]
    theta = pywt.waverec(theta_coeffs, wavelet)
    
    # Alpha (8-13 Hz): use part of cD3 (approximate)
    # We can filter cD3 to better isolate the alpha band
    alpha_coeffs = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, cD3, zero_cD2, zero_cD1]
    alpha_broad = pywt.waverec(alpha_coeffs, wavelet)
    
    # Apply a bandpass filter to isolate alpha (8-13 Hz) from alpha_broad (8-16 Hz)
    nyquist = fs_eeg / 2
    low_cutoff = 8 / nyquist
    high_cutoff = 13 / nyquist
    b, a = sig.butter(4, [low_cutoff, high_cutoff], btype='band')
    alpha = sig.filtfilt(b, a, alpha_broad)
    
    # Beta (13-30 Hz): filter cD3 and cD2
    beta_coeffs = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, cD3, cD2, zero_cD1]
    beta_broad = pywt.waverec(beta_coeffs, wavelet)
    
    # Apply a bandpass filter to isolate beta (13-30 Hz)
    low_cutoff = 13 / nyquist
    high_cutoff = 30 / nyquist
    b, a = sig.butter(4, [low_cutoff, high_cutoff], btype='band')
    beta = sig.filtfilt(b, a, beta_broad)
    
    # Gamma (30-45 Hz): use part of cD1
    gamma_coeffs = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, zero_cD3, zero_cD2, cD1]
    gamma_broad = pywt.waverec(gamma_coeffs, wavelet)
    
    # Apply a bandpass filter to isolate gamma (30-45 Hz)
    low_cutoff = 30 / nyquist
    high_cutoff = 45 / nyquist
    b, a = sig.butter(4, [low_cutoff, high_cutoff], btype='band')
    gamma = sig.filtfilt(b, a, gamma_broad)
    
    # Plot the decomposed signals
    plt.figure(figsize=(12, 10))
    
    # Calculate time vector
    time = np.arange(len(cleaned_eeg[0, :])) / fs_eeg
    
    # Determine the length to plot (use the same length for all)
    plot_length = min(len(time), len(slow_delta), len(delta), len(theta), len(alpha), len(beta), len(gamma))
    plot_time = time[:plot_length]
    
    # 5 seconds window for better visualization (adjust from 30*fs_eeg to 5*fs_eeg)
    window_length = 5 * fs_eeg
    window_start = int(10 * 60 * fs_eeg)  # Start at 10 minutes
    window_end = min(window_start + window_length, plot_length)
    
    plt.subplot(7, 1, 1)
    plt.plot(plot_time[window_start:window_end], cleaned_eeg[0, window_start:window_end])
    plt.title('Original Signal')
    
    plt.subplot(7, 1, 2)
    plt.plot(plot_time[window_start:window_end], slow_delta[window_start:window_end])
    plt.title('Slow Delta (0-1 Hz)')
    
    plt.subplot(7, 1, 3)
    plt.plot(plot_time[window_start:window_end], delta[window_start:window_end])
    plt.title('Delta (1-4 Hz)')
    
    plt.subplot(7, 1, 4)
    plt.plot(plot_time[window_start:window_end], theta[window_start:window_end])
    plt.title('Theta (4-8 Hz)')
    
    plt.subplot(7, 1, 5)
    plt.plot(plot_time[window_start:window_end], alpha[window_start:window_end])
    plt.title('Alpha (8-13 Hz)')
    
    plt.subplot(7, 1, 6)
    plt.plot(plot_time[window_start:window_end], beta[window_start:window_end])
    plt.title('Beta (13-30 Hz)')
    
    plt.subplot(7, 1, 7)
    plt.plot(plot_time[window_start:window_end], gamma[window_start:window_end])
    plt.title('Gamma (30-45 Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eeg_bands.png'))
    plt.show()
    
    return slow_delta, delta, theta, alpha, beta, gamma

def detect_sleep_spindles(alpha, fs_eeg, output_dir):
    """
    Detect sleep spindles in the 10-15 Hz range.
    
    Args:
        alpha: The alpha/sigma band signal (should contain frequencies up to 16 Hz)
        fs_eeg: Sampling frequency
        
    Returns:
        spindle_properties: List of dictionaries containing spindle properties
    """
    # Apply a more precise bandpass filter for spindle frequency range (10-15 Hz)
    nyquist = fs_eeg / 2
    low_cutoff = 10 / nyquist  # Changed from 12 to 10 Hz
    high_cutoff = 15 / nyquist
    b, a = sig.butter(4, [low_cutoff, high_cutoff], btype='band')
    sigma = sig.filtfilt(b, a, alpha)
    
    # Compute the envelope using Hilbert transform
    sigma_envelope = np.abs(sig.hilbert(sigma))
    
    # Apply a low-pass filter to smooth the envelope
    b_smooth, a_smooth = sig.butter(2, 4/nyquist, btype='low')
    sigma_envelope_smooth = sig.filtfilt(b_smooth, a_smooth, sigma_envelope)
    
    # Calculate amplitude threshold (mean + 1.5 standard deviations)
    # This adapts better to different signal qualities than using a fixed multiplier
    amplitude_threshold = np.mean(sigma_envelope_smooth) + 1.5 * np.std(sigma_envelope_smooth)
    
    # Detect potential spindles
    above_threshold = sigma_envelope_smooth > amplitude_threshold
    
    # Apply minimum duration constraints to reduce false positives
    # Sleep spindles typically last 0.5-2 seconds
    from scipy import ndimage
    min_samples = int(0.5 * fs_eeg)  # Minimum 0.5 seconds
    max_samples = int(2.0 * fs_eeg)  # Maximum 2.0 seconds
    
    # Label connected regions above threshold
    labeled_spindles, num_spindles = ndimage.label(above_threshold)
    
    # Process each potential spindle and collect their properties
    spindle_properties = []
    
    for i in range(1, num_spindles + 1):
        # Get indices for this labeled region
        spindle_indices = np.where(labeled_spindles == i)[0]
        
        # Calculate duration in seconds
        duration = len(spindle_indices) / fs_eeg
        
        # Only keep spindles within valid duration range
        if min_samples <= len(spindle_indices) <= max_samples:
            start_idx = spindle_indices[0]
            end_idx = spindle_indices[-1]
            
            # Extract the spindle segment
            spindle_segment = sigma[start_idx:end_idx + 1]
            
            # Calculate additional properties
            peak_amplitude = np.max(sigma_envelope_smooth[start_idx:end_idx + 1])
            mean_amplitude = np.mean(sigma_envelope_smooth[start_idx:end_idx + 1])
            
            # Calculate frequency using zero-crossings or spectral peak
            if len(spindle_segment) > 4:  # Ensure enough samples for frequency estimation
                # Count zero-crossings for a rough frequency estimate
                zero_crossings = np.sum(np.diff(np.signbit(spindle_segment)) != 0)
                frequency = zero_crossings * fs_eeg / (2 * len(spindle_segment))
                
                # Only include if frequency is within spindle range (as a double-check)
                if 9 <= frequency <= 16:  # Slightly wider range for verification
                    spindle_properties.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_time': start_idx / fs_eeg,
                        'end_time': end_idx / fs_eeg,
                        'duration': duration,
                        'peak_amplitude': peak_amplitude,
                        'mean_amplitude': mean_amplitude,
                        'frequency': frequency
                    })
    
    # Sort spindles by start time
    spindle_properties = sorted(spindle_properties, key=lambda x: x['start_time'])
    
    # Plot some examples for verification
    if len(spindle_properties) > 0:
        plt.figure(figsize=(15, 10))
        
        # Plot raw signal
        plt.subplot(3, 1, 1)
        # Show 60 seconds of data centered around the first spindle
        if len(spindle_properties) > 0:
            center = int(spindle_properties[0]['start_idx'] + spindle_properties[0]['duration'] * fs_eeg / 2)
            start = max(0, center - 30 * fs_eeg)
            end = min(len(alpha), center + 30 * fs_eeg)
            plt.plot(np.arange(start, end) / fs_eeg, alpha[start:end])
            plt.title('Alpha Band Signal (8-16 Hz)')
            plt.xlabel('Time (s)')
        
        # Plot sigma band signal
        plt.subplot(3, 1, 2)
        if len(spindle_properties) > 0:
            plt.plot(np.arange(start, end) / fs_eeg, sigma[start:end])
            plt.title('Spindle Band Signal (10-15 Hz)')
            plt.xlabel('Time (s)')
        
        # Plot envelope with threshold and detected spindles
        plt.subplot(3, 1, 3)
        if len(spindle_properties) > 0:
            plt.plot(np.arange(start, end) / fs_eeg, sigma_envelope_smooth[start:end])
            plt.axhline(y=amplitude_threshold, color='r', linestyle='--', label='Threshold')
            
            # Mark detected spindles
            for spindle in spindle_properties:
                if start <= spindle['start_idx'] <= end and start <= spindle['end_idx'] <= end:
                    plt.axvspan(spindle['start_time'], spindle['end_time'], 
                               color='green', alpha=0.3, 
                               label=f"Spindle: {spindle['frequency']:.1f} Hz")
            
            plt.title('Spindle Detection')
            plt.xlabel('Time (s)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spindle_detection.png'))
        plt.show()
    
    # Print spindle statistics
    valid_spindles = len(spindle_properties)
    print(f"Detected {valid_spindles} sleep spindles")
    
    # Print details of first 10 spindles
    for i, spindle in enumerate(spindle_properties[:10]):
        print(f"Spindle {i+1}: Start: {spindle['start_time']:.2f}s, Duration: {spindle['duration']:.2f}s, Frequency: {spindle['frequency']:.1f} Hz")
    
    # Calculate number of spindles per epoch
    # Assuming 30-second epochs
    epoch_length = 30  # seconds
    max_time = alpha.shape[0] / fs_eeg
    num_epochs = int(np.ceil(max_time / epoch_length))
    
    spindles_per_epoch = np.zeros(num_epochs)
    
    # Count spindles in each epoch
    for spindle in spindle_properties:
        epoch_idx = int(spindle['start_time'] / epoch_length)
        if epoch_idx < num_epochs:
            spindles_per_epoch[epoch_idx] += 1
    
    # Plot barplot of spindles per epoch
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(num_epochs), spindles_per_epoch, color='steelblue')
    plt.xlabel('Epoch Number')
    plt.ylabel('Number of Spindles')
    plt.title('Sleep Spindles Per 30-Second Epoch')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a moving average line to show trends
    window_size = min(20, num_epochs // 5) if num_epochs > 20 else 5
    if window_size > 1:
        moving_avg = np.convolve(spindles_per_epoch, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', linewidth=2, 
                label=f'{window_size}-Epoch Moving Average')
        plt.legend()
    
    # Calculate some statistics
    max_spindles = np.max(spindles_per_epoch)
    max_epoch = np.argmax(spindles_per_epoch)
    mean_spindles = np.mean(spindles_per_epoch)
    
    # Add text annotation with statistics
    plt.text(0.02, 0.95, f"Max: {max_spindles:.0f} spindles (epoch {max_epoch})\nMean: {mean_spindles:.2f} spindles/epoch", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spindles_per_epoch.png'))
    plt.show()
    
    return spindle_properties

def calculate_psd(cleaned_eeg, fs_eeg, epoch_length, output_dir):
    # Calculate power spectral density (PSD) for each EEG channel
    from scipy import signal
    
    # Total length of the EEG data
    total_length = cleaned_eeg.shape[1]
    
    # Number of epochs
    num_epochs = total_length // (fs_eeg * epoch_length)
    print(f"Total number of epochs: {num_epochs}")
    
    # Create frequency array for plotting
    f = np.linspace(0, fs_eeg/2, fs_eeg*epoch_length//2+1)
    
    # Compute PSD for each epoch
    psd_epochs = np.zeros((1, num_epochs, len(f)))  # Changed from 2 channels to 1
    
    for epoch in range(num_epochs):
        start_idx = epoch * fs_eeg * epoch_length
        end_idx = start_idx + fs_eeg * epoch_length
        
        # Extract this epoch's data
        epoch_data = cleaned_eeg[0, start_idx:end_idx]  # Only use channel 0
        
        # Compute PSD using Welch's method
        f, psd = signal.welch(epoch_data, fs=fs_eeg, nperseg=fs_eeg*4)
        
        # Store the PSD data
        psd_epochs[0, epoch, :len(f)] = psd  # Store in channel 0
    
    # Define frequency bands of interest
    bands = {
        'slow_delta': (0.1, 1),
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Calculate total power for each frequency band in each epoch
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_powers[band_name] = np.zeros((1, num_epochs))  # Changed from 2 channels to 1
        
        # Find indices corresponding to the frequency band
        idx_low = np.argmax(f >= low_freq)
        idx_high = np.argmax(f >= high_freq) if high_freq < fs_eeg/2 else len(f)-1
        
        # Calculate power in this band for each epoch
        for epoch in range(num_epochs):
            band_powers[band_name][0, epoch] = np.sum(psd_epochs[0, epoch, idx_low:idx_high+1])  # Only for channel 0
    
    # Compute epoch timestamps in hours
    epoch_timestamps = np.arange(num_epochs) * epoch_length
    
    # Plot band powers over time
    fig, axs = plt.subplots(len(bands), 1, figsize=(15, 10), sharex=True)
    
    # Plot band powers for the single channel
    for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
        ax = axs[i]
        ax.plot(epoch_timestamps / 3600, band_powers[band_name][0, :], color='blue', label='EEG Channel')
        
        ax.set_title(f'{band_name.capitalize()} Band Power ({low_freq}-{high_freq} Hz)')
        ax.set_ylabel('Power')
        ax.legend()
    
    axs[-1].set_xlabel('Time (hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'band_powers_over_time.png'))
    plt.show()
    
    return psd_epochs, band_powers, f, epoch_timestamps

def load_sleep_stages(num_epochs, epoch_timestamps, output_dir):
    # Load sleep stages from .mat file
    sleep_stages = sio.loadmat('hypnogram.mat')['stages']
    print(f"Shape of sleep stages: {sleep_stages.shape}")
    
    # Flatten if necessary
    if sleep_stages.ndim > 1:
        sleep_stages = sleep_stages.flatten()
    
    # Since sleep_stages have many more data points, we'll take one value per chunk
    if len(sleep_stages) != num_epochs:
        print(f"Resampling sleep stages from {len(sleep_stages)} to {num_epochs}")
        resample_factor = len(sleep_stages) / num_epochs
        resampled_sleep_stages = np.zeros(num_epochs)
        
        for i in range(num_epochs):
            # Calculate the corresponding indices in the original sleep_stages
            start_idx = int(i * resample_factor)
            end_idx = int((i + 1) * resample_factor)
            end_idx = min(end_idx, len(sleep_stages))  # Ensure we don't exceed array bounds
            
            # Take the most common stage in this chunk
            chunk = sleep_stages[start_idx:end_idx]
            values, counts = np.unique(chunk, return_counts=True)
            resampled_sleep_stages[i] = values[np.argmax(counts)]
        
        sleep_stages = resampled_sleep_stages
    
    # Plot the sleep stages
    plt.figure(figsize=(15, 4))
    plt.step(epoch_timestamps / 3600, sleep_stages, where='post')
    plt.title('Sleep Stages Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Sleep Stage')
    plt.yticks([0, 2, 3, 4, 5], ['REM', 'N3', 'N2', 'N1', 'Wake'])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sleep_stages.png'))
    plt.show()
    
    return sleep_stages

def plot_band_powers_with_sleep_stages(band_powers, sleep_stages, epoch_timestamps, output_dir):
    # Plot band powers with sleep stages
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: Band powers (only showing a few bands for clarity)
    ax1 = axs[0]
    bands_to_show = ['delta', 'theta', 'alpha', 'beta']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, band in enumerate(bands_to_show):
        # Show power from the single channel (index 0)
        power = band_powers[band][0, :]
        ax1.plot(epoch_timestamps / 3600, power, color=colors[i], label=band.capitalize())
    
    ax1.set_title('EEG Band Powers and Sleep Stages')
    ax1.set_ylabel('Power')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Sleep stages
    ax2 = axs[1]
    ax2.step(epoch_timestamps / 3600, sleep_stages, where='post', color='black')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Sleep Stage')
    ax2.set_yticks([0, 2, 3, 4, 5])
    ax2.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'band_powers_with_sleep_stages.png'))
    plt.show()
    
    # Create a stacked area plot of band powers
    plt.figure(figsize=(15, 8))
    
    # Calculate average across channels for each band
    avg_powers = {}
    bands_to_stack = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    colors = ['darkblue', 'blue', 'green', 'red', 'purple', 'orange']
    
    # Create a stacked area plot
    plt.figure(figsize=(15, 6))
    
    # Get power data for each band from the single channel (index 0)
    band_data = [band_powers[band][0, :] for band in bands_to_stack]
    
    # Create the stacked area plot
    plt.stackplot(epoch_timestamps / 3600, band_data, labels=bands_to_stack, colors=colors, alpha=0.7)
    
    # Add sleep stages as step plot on a twin axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.step(epoch_timestamps / 3600, sleep_stages, where='post', color='black', linewidth=1.5)
    ax2.set_yticks([0, 2, 3, 4, 5])
    ax2.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax2.set_ylim(-1, 6)
    
    plt.title('Stacked EEG Band Powers with Sleep Stages')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Power')
    ax2.set_ylabel('Sleep Stage')
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_stages_with_band_powers.png'))
    plt.show()
    
    return

def extract_eeg_features(cleaned_eeg, slow_delta, delta, theta, alpha, beta, gamma, 
                       spindle_properties, band_powers, sleep_stages, fs_eeg, epoch_length, output_dir):
    print("Extracting EEG features...")
    
    # Calculate the number of epochs
    total_length = cleaned_eeg.shape[1]
    num_epochs = total_length // (fs_eeg * epoch_length)
    
    # Unique sleep stages
    unique_stages = np.unique(sleep_stages)
    
    # Ensure sleep_stages array matches the number of epochs we have power data for
    min_length = min(len(sleep_stages), num_epochs)
    
    # Analyze power distribution for each sleep stage
    for stage in unique_stages:
        # Find indices for this stage
        stage_indices = np.where(sleep_stages[:min_length] == stage)[0]
        if len(stage_indices) == 0:
            continue
            
        stage_name = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}.get(stage, f'Unknown {stage}')
        print(f"\nSleep Stage: {stage_name}, Count: {len(stage_indices)}")
        
        # Calculate average band power for this stage
        for band_name in band_powers.keys():
            # Average power for the single channel (index 0)
            avg_power = np.mean(band_powers[band_name][0, stage_indices])
            print(f"  Average {band_name} power: {avg_power:.2f}")
    
    # Initialize feature containers
    features = {
        'absolute_power': {},
        'relative_power': {},
        'power_ratios': {},
        'spectral_measures': {},
        'time_domain': {}
    }
    
    # Calculate spectral entropy
    spectral_entropy = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        start_idx = epoch * fs_eeg * epoch_length
        end_idx = start_idx + fs_eeg * epoch_length
        
        # Extract epoch data from the single channel
        epoch_data = cleaned_eeg[0, start_idx:end_idx]
        
        # Calculate PSD
        f, psd = welch(epoch_data, fs=fs_eeg, nperseg=fs_eeg*2)
        
        # Normalize the PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy (higher entropy = more complex/random signal)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        spectral_entropy[epoch] = entropy
    
    # Calculate spectral edge frequencies
    spectral_edge_95 = np.zeros(num_epochs)
    spectral_edge_50 = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        start_idx = epoch * fs_eeg * epoch_length
        end_idx = start_idx + fs_eeg * epoch_length
        
        # Extract epoch data from the single channel
        epoch_data = cleaned_eeg[0, start_idx:end_idx]
        
        # Calculate PSD
        f, psd = welch(epoch_data, fs=fs_eeg, nperseg=fs_eeg*2)
        
        # Calculate cumulative sum of PSD
        cum_psd = np.cumsum(psd) / np.sum(psd)
        
        # Find the frequency below which 95% of power is contained
        idx_95 = np.where(cum_psd >= 0.95)[0]
        spectral_edge_95[epoch] = f[idx_95[0]] if len(idx_95) > 0 else 0
        
        # Find the frequency below which 50% of power is contained (median frequency)
        idx_50 = np.where(cum_psd >= 0.50)[0]
        spectral_edge_50[epoch] = f[idx_50[0]] if len(idx_50) > 0 else 0
    
    # Time domain features
    # Calculate variance, skewness, kurtosis, and zero crossings for each epoch
    variance = np.zeros(num_epochs)
    skewness = np.zeros(num_epochs)
    kurtosis_values = np.zeros(num_epochs)
    zero_crossings = np.zeros(num_epochs)
    
    # Hjorth parameters (activity, mobility, complexity)
    hjorth_activity = np.zeros(num_epochs)     # variance of signal (first derivative)
    hjorth_mobility = np.zeros(num_epochs)     # sqrt(variance of first derivative / variance of signal)
    hjorth_complexity = np.zeros(num_epochs)   # ratio of mobility of first derivative to mobility of signal
    
    for epoch in range(num_epochs):
        start_idx = epoch * fs_eeg * epoch_length
        end_idx = start_idx + fs_eeg * epoch_length
        
        # Extract epoch data from the single channel
        epoch_data = cleaned_eeg[0, start_idx:end_idx]
        
        # Calculate statistics
        variance[epoch] = np.var(epoch_data)
        skewness[epoch] = skew(epoch_data)
        kurtosis_values[epoch] = kurtosis(epoch_data)
        
        # Count zero crossings
        zero_crossings[epoch] = np.sum(np.diff(np.signbit(epoch_data)))
        
        # Calculate Hjorth parameters
        # First derivative
        diff1 = np.diff(epoch_data)
        # Second derivative
        diff2 = np.diff(diff1)
        
        # Activity = variance of signal
        hjorth_activity[epoch] = np.var(epoch_data)
        
        # Mobility = sqrt(variance of first derivative / variance of signal)
        hjorth_mobility[epoch] = np.sqrt(np.var(diff1) / np.var(epoch_data))
        
        # Complexity = mobility of first derivative / mobility of signal
        mobility_diff1 = np.sqrt(np.var(diff2) / np.var(diff1))
        hjorth_complexity[epoch] = mobility_diff1 / hjorth_mobility[epoch]
    
    # Calculate spindle density per epoch (number of spindles / epoch duration in minutes)
    spindle_density = np.zeros(num_epochs)
    
    for spindle in spindle_properties:
        start_time = spindle['start_time']
        epoch_idx = int(start_time / epoch_length)
        if epoch_idx < num_epochs:
            spindle_density[epoch_idx] += 1
    
    # Convert from spindles per epoch to spindles per minute
    spindle_density = spindle_density / 0.5  # per minute
    
    # Store all features
    # Absolute power features (using only channel 0)
    features['absolute_power'] = {
        'slow_delta': band_powers['slow_delta'][0, :],
        'delta': band_powers['delta'][0, :],
        'theta': band_powers['theta'][0, :],
        'alpha': band_powers['alpha'][0, :],
        'beta': band_powers['beta'][0, :],
        'gamma': band_powers['gamma'][0, :]
    }
    
    # Calculate total power for relative measures
    total_power = np.sum([band_powers[band][0, :] for band in band_powers.keys()], axis=0)
    
    # Relative power features
    features['relative_power'] = {
        'delta': band_powers['delta'][0, :] / total_power,
        'theta': band_powers['theta'][0, :] / total_power,
        'alpha': band_powers['alpha'][0, :] / total_power,
        'beta': band_powers['beta'][0, :] / total_power
    }
    
    # Power ratios - useful for sleep stage classification
    features['power_ratios'] = {
        'theta_to_alpha': band_powers['theta'][0, :] / (band_powers['alpha'][0, :] + 1e-10),
        'delta_to_beta': band_powers['delta'][0, :] / (band_powers['beta'][0, :] + 1e-10),
        'theta_to_beta': band_powers['theta'][0, :] / (band_powers['beta'][0, :] + 1e-10),
        'slow_delta_to_gamma': band_powers['slow_delta'][0, :] / (band_powers['gamma'][0, :] + 1e-10)
    }
    
    # Spectral complexity measures
    features['spectral_measures'] = {
        'spectral_entropy': spectral_entropy,
        'spectral_edge_95': spectral_edge_95,
        'spectral_edge_50': spectral_edge_50
    }
    
    # Time domain statistics
    features['time_domain'] = {
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis_values,
        'zero_crossings': zero_crossings,
        'hjorth_activity': hjorth_activity,
        'hjorth_mobility': hjorth_mobility,
        'hjorth_complexity': hjorth_complexity,
        'spindle_density': spindle_density
    }
    
    # Print some statistics for each sleep stage
    stage_names = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
    
    for stage in unique_stages:
        stage_indices = np.where(sleep_stages == stage)[0]
        if len(stage_indices) == 0:
            continue
            
        stage_name = stage_names.get(stage, f'Unknown {stage}')
        print(f"\nStatistics for {stage_name} stage:")
        
        # Spectral measures
        avg_entropy = np.mean(features['spectral_measures']['spectral_entropy'][stage_indices])
        print(f"  Average spectral entropy: {avg_entropy:.4f}")
        
        # Power ratios
        avg_ratio = np.mean(features['power_ratios']['delta_to_beta'][stage_indices])
        print(f"  Average delta/beta ratio: {avg_ratio:.4f}")
        
        # Time domain
        avg_variance = np.mean(features['time_domain']['variance'][stage_indices])
        print(f"  Average variance: {avg_variance:.4f}")
        
        # Spindles
        if 'spindle_density' in features['time_domain']:
            avg_density = np.mean(features['time_domain']['spindle_density'][stage_indices])
            print(f"  Average spindle density: {avg_density:.4f} spindles/min")
    
    # Save features to a .mat file for further analysis
    feature_dict = {
        'timestamps': np.arange(num_epochs) * epoch_length,
        'sleep_stage': sleep_stages
    }
    
    # Flatten the nested dictionary for saving
    for category in features:
        for feature_name, feature_data in features[category].items():
            # Limit to the valid range
            feature_dict[f"{category}_{feature_name}"] = feature_data[:min_length]
    
    # Save to .mat file
    sio.savemat(os.path.join(output_dir, 'eeg_features.mat'), feature_dict)
    
    # Also save as CSV for easier access
    df_data = {'timestamp': np.arange(min_length) * epoch_length}
    df_data['sleep_stage'] = sleep_stages[:min_length]
    
    for category in features:
        for feature_name, feature_data in features[category].items():
            df_data[feature_name] = feature_data[:min_length]
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, 'eeg_features.csv'), index=False)
    
    print(f"Feature extraction completed. Saved {len(df.columns) - 2} features for {min_length} epochs.")
    
    return features

def build_sleep_stage_classifier(output_dir):
    print("\nBuilding sleep stage classification models...")
    
    # Load the DataFrame if not already in memory
    try:
        feature_df
    except NameError:
        import pandas as pd
        feature_df = pd.read_csv(os.path.join(output_dir, 'eeg_features.csv'))
    
    # Remove any NaN values
    feature_df = feature_df.dropna()
    
    # Remove rows where sleep_stage = 1 (empty label) or any invalid stage
    valid_stages = [0, 2, 3, 4, 5]  # REM, N3, N2, N1, Wake
    feature_df = feature_df[feature_df['sleep_stage'].isin(valid_stages)]
    
    # Define X (features) and y (target labels)
    X = feature_df.drop(['timestamp', 'sleep_stage'], axis=1)
    y = feature_df['sleep_stage']
    
    # Print the number of samples for each sleep stage
    print("Sleep stage distribution:")
    stage_counts = y.value_counts().sort_index()
    stage_mapping = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
    for stage, count in stage_counts.items():
        stage_name = stage_mapping.get(stage, f'Unknown {stage}')
        print(f"  {stage_name}: {count} samples")
    
    # Use a LabelEncoder to transform non-consecutive labels to consecutive integers (0, 1, 2, 3, 4)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Store the original class names in the same order as the encoded labels
    encoded_classes = [stage_mapping[stage] for stage in label_encoder.classes_]
    
    # Create training and validation sets with 40:60 ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.6, random_state=42, stratify=y_encoded)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # First, let's check for feature importance using a preliminary Random Forest
    preliminary_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    preliminary_rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = preliminary_rf.feature_importances_
    feature_names = X.columns
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(indices[:20])), importances[indices[:20]], align='center')
    plt.xticks(range(len(indices[:20])), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.show()
    
    # Select the top 20 features for models to prevent overfitting
    selector = SelectFromModel(preliminary_rf, threshold=-np.inf, max_features=20)
    selector.fit(X_train, y_train)
    
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    
    # Get names of selected features
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]
    print("\nTop 20 selected features:")
    for i, feature in enumerate(selected_feature_names):
        print(f"{i+1}. {feature}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # 1. Random Forest Classifier
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_val_scaled)
    rf_accuracy = accuracy_score(y_val, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # 2. Support Vector Machine
    print("\nTraining SVM classifier...")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    svm_pred = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # 3. XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(encoded_classes),  # Use the correct number of classes
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_val_scaled)
    xgb_accuracy = accuracy_score(y_val, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Compare model accuracies
    model_accuracies = {
        'Random Forest': rf_accuracy,
        'SVM': svm_accuracy,
        'XGBoost': xgb_accuracy
    }
    
    # Find the best model
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model_accuracy = model_accuracies[best_model_name]
    print(f"\nBest model: {best_model_name} with accuracy {best_model_accuracy:.4f}")
    
    # Detailed classification report for all models
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_val, rf_pred, target_names=encoded_classes))
    
    print("\n--- SVM Classification Report ---")
    print(classification_report(y_val, svm_pred, target_names=encoded_classes))
    
    print("\n--- XGBoost Classification Report ---")
    print(classification_report(y_val, xgb_pred, target_names=encoded_classes))
    
    # Plot confusion matrices
    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=encoded_classes,
                   yticklabels=encoded_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_").lower()}.png'))
        plt.show()
    
    # Plot confusion matrices for all models
    plot_confusion_matrix(y_val, rf_pred, 'Random Forest Confusion Matrix')
    plot_confusion_matrix(y_val, svm_pred, 'SVM Confusion Matrix')
    plot_confusion_matrix(y_val, xgb_pred, 'XGBoost Confusion Matrix')
    
    # Bar plot comparing model accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values())
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, (model, acc) in enumerate(model_accuracies.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.show()
    
    # Save the best model
    best_model = None
    if best_model_name == 'Random Forest':
        best_model = rf_model
    elif best_model_name == 'SVM':
        best_model = svm_model
    else:  # XGBoost
        best_model = xgb_model
    
    with open(os.path.join(output_dir, 'best_sleep_classifier.pkl'), 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'selector': selector,
            'feature_names': feature_names,
            'selected_features': selected_feature_names
        }, f)
    
    print(f"\nBest model saved as 'best_sleep_classifier.pkl'")
    
    # Generate learning curves to evaluate if more data would help
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt
    
    # Plot learning curve for the best model
    print(f"\nGenerating learning curve for {best_model_name}...")
    
    # Use a smaller subset for learning curve to speed up computation
    X_subset = X_train_scaled[:min(1000, len(X_train_scaled))]
    y_subset = y_train[:min(1000, len(y_train))]
    
    if best_model_name == 'Random Forest':
        lc_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    elif best_model_name == 'SVM':
        lc_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    else:  # XGBoost
        lc_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=7, random_state=42)
        
    plot_learning_curve(lc_model, f'Learning Curve ({best_model_name})', 
                       X_subset, y_subset[:len(X_subset)], ylim=(0.5, 1.01), cv=5, n_jobs=-1)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.show()
    
    return

def main():
    # Ensure matplotlib is accessible within this function
    import matplotlib.pyplot as plt
    
    # Create Output folder if it doesn't exist
    output_dir = 'Output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess the data
    eeg_data, eog_data, emg_data, ecg_data, fs_eeg, epoch_length = load_and_preprocess_data()

    # Simple plot of the two EEG channel at half of the dataset for 30 seconds
    plt.figure(figsize=(12, 6))
    plt.plot(eeg_data[0, :fs_eeg*epoch_length*30])
    plt.title('Raw EEG Channel 1')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_eeg.png'))
    plt.show()

    # Filter the EEG data
    filtered_eeg = filter_eeg_data(eeg_data, fs_eeg)
    
    # Keep original data for comparison in plots
    original_eeg = eeg_data.copy()
    eeg_data = filtered_eeg

    # Simple plot of the filtered eeg_data over the original eeg_data
    plt.figure(figsize=(12, 6))
    plt.plot(original_eeg[0, :fs_eeg*epoch_length*30], label='Original')
    plt.plot(filtered_eeg[0, :fs_eeg*epoch_length*30], 'r-', label='Filtered')
    plt.title('Original vs Filtered EEG Channel 1')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtered_eeg.png'))
    plt.show()

    # Apply ICA to the filtered eeg_data - now returns only one channel
    cleaned_eeg = apply_ica(eeg_data, eog_data, emg_data, ecg_data, fs_eeg, epoch_length, output_dir)
    
    # Apply wavelet transform
    slow_delta, delta, theta, alpha, beta, gamma = apply_wavelet_transform(cleaned_eeg, fs_eeg, epoch_length, output_dir)
    
    # Detect sleep spindles
    spindle_properties = detect_sleep_spindles(alpha, fs_eeg, output_dir)
    
    # Calculate PSD and band powers
    psd_epochs, band_powers, freq, epoch_timestamps = calculate_psd(cleaned_eeg, fs_eeg, epoch_length, output_dir)
    
    # Load sleep stages
    sleep_stages = load_sleep_stages(len(epoch_timestamps), epoch_timestamps, output_dir)
    
    # Plot band powers with sleep stages
    plot_band_powers_with_sleep_stages(band_powers, sleep_stages, epoch_timestamps, output_dir)
    
    # Extract EEG features
    features = extract_eeg_features(cleaned_eeg, slow_delta, delta, theta, alpha, beta, gamma,
                                   spindle_properties, band_powers, sleep_stages, fs_eeg, epoch_length, output_dir)
    
    # Build sleep stage classifier
    build_sleep_stage_classifier(output_dir)

if __name__ == "__main__":
    main()