"""
EEG Processing Library

This module provides a set of functions for processing and analyzing EEG data,
including filtering, artifact removal using ICA, frequency band extraction,
feature extraction, and classification.

The functions are designed to be modular and reusable for various EEG processing projects.
"""

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
import os
from scipy.stats import entropy, skew, kurtosis


def setup_output_directory(base_dir=None):
    """
    Create an output directory for saving results.
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory path. If None, uses current working directory.
    
    Returns:
    --------
    str
        Path to the output directory
    """
    if base_dir is None:
        base_dir = os.getcwd()
        
    output_dir = os.path.join(base_dir, 'Output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    return output_dir


def get_output_path(filename, output_dir=None):
    """
    Generate a full path for a file in the output directory.
    
    Parameters:
    -----------
    filename : str
        Name of the output file
    output_dir : str, optional
        Output directory path. If None, creates a default output directory.
    
    Returns:
    --------
    str
        Full path to the output file
    """
    if output_dir is None:
        output_dir = setup_output_directory()
        
    return os.path.join(output_dir, filename)


def load_data(file_path):
    """
    Load EEG data from a .mat file.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing EEG data
    
    Returns:
    --------
    dict
        Dictionary containing loaded EEG data channels
    """
    mat_data = sio.loadmat(file_path)
    
    # Create structures for different channel types
    data = {}
    
    # EEG channels
    if 'eeg1' in mat_data and 'eeg2' in mat_data:
        eeg_data = np.zeros((2, len(mat_data['eeg1'][0])))
        eeg_data[0, :] = mat_data['eeg1'][0]
        eeg_data[1, :] = mat_data['eeg2'][0]
        data['eeg'] = eeg_data
    
    # EOG channels
    if 'eogl' in mat_data and 'eogr' in mat_data:
        eog_data = np.zeros((2, len(mat_data['eogl'][0])))
        eog_data[0, :] = mat_data['eogl'][0]
        eog_data[1, :] = mat_data['eogr'][0]
        data['eog'] = eog_data
    
    # EMG channel
    if 'emg' in mat_data:
        emg_data = np.zeros((1, len(mat_data['emg'][0])))
        emg_data[0, :] = mat_data['emg'][0]
        data['emg'] = emg_data
    
    # ECG channel
    if 'ecg' in mat_data:
        ecg_data = np.zeros((1, len(mat_data['ecg'][0])))
        ecg_data[0, :] = mat_data['ecg'][0]
        data['ecg'] = ecg_data
    
    return data


def bandpass_filter(signal_data, fs, low_cutoff=0.5, high_cutoff=45, order=4):
    """
    Apply a bandpass filter to signal data.
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        Signal data with shape [n_channels, n_samples]
    fs : float
        Sampling frequency in Hz
    low_cutoff : float, optional
        Lower cutoff frequency in Hz, default 0.5 Hz
    high_cutoff : float, optional
        Higher cutoff frequency in Hz, default 45 Hz
    order : int, optional
        Filter order, default 4
    
    Returns:
    --------
    numpy.ndarray
        Filtered signal with same shape as input
    """
    nyquist = fs / 2
    low_cutoff_norm = low_cutoff / nyquist
    high_cutoff_norm = high_cutoff / nyquist
    
    # Create bandpass filter
    b, a = sig.butter(order, [low_cutoff_norm, high_cutoff_norm], btype='band')
    
    # Apply filter to each channel
    filtered_signal = np.zeros_like(signal_data)
    for i in range(signal_data.shape[0]):
        filtered_signal[i, :] = sig.filtfilt(b, a, signal_data[i, :])
    
    return filtered_signal


def plot_eeg_comparison(original_eeg, filtered_eeg, fs, duration=30, title=None, save_path=None):
    """
    Plot comparison between original and filtered EEG signals.
    
    Parameters:
    -----------
    original_eeg : numpy.ndarray
        Original EEG signal with shape [n_channels, n_samples]
    filtered_eeg : numpy.ndarray
        Filtered EEG signal with shape [n_channels, n_samples]
    fs : float
        Sampling frequency in Hz
    duration : int, optional
        Duration to plot in seconds, default 30
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    n_channels = original_eeg.shape[0]
    samples_to_plot = int(fs * duration)
    
    fig, axs = plt.subplots(n_channels, 1, figsize=(12, 4*n_channels))
    if n_channels == 1:
        axs = [axs]
    
    for i in range(n_channels):
        axs[i].plot(original_eeg[i, :samples_to_plot], 'b-', label='Original')
        axs[i].plot(filtered_eeg[i, :samples_to_plot], 'r-', label='Filtered')
        axs[i].set_title(f'Channel {i+1}')
        axs[i].legend()
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig 


def apply_ica(eeg_data, artifact_data, n_components=None):
    """
    Apply Independent Component Analysis (ICA) to EEG data combined with artifact channels.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape [n_eeg_channels, n_samples]
    artifact_data : dict
        Dictionary of artifact data channels with keys like 'eog', 'emg', 'ecg'
        Each entry should be a numpy array with shape [n_channels, n_samples]
    n_components : int, optional
        Number of ICA components to extract. If None, uses the number of channels.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'ica': The fitted ICA object
        - 'components': ICA components with shape [n_components, n_samples]
        - 'mixing_matrix': Mixing matrix A
        - 'sources': Source signals S
        - 'combined_data': All channels combined for reference
    """
    # Combine all channels for ICA
    channels_list = [eeg_data]
    
    for artifact_type, data in artifact_data.items():
        channels_list.append(data)
    
    all_channels = np.vstack(channels_list)
    
    # Transpose for ICA (ICA expects [n_samples, n_features])
    X = all_channels.T
    
    # Determine number of components
    if n_components is None:
        n_components = all_channels.shape[0]
    
    # Apply FastICA
    ica = FastICA(n_components=n_components, random_state=42)
    S = ica.fit_transform(X)  # S contains the independent components
    A = ica.mixing_  # A is the mixing matrix
    
    # Get the independent components (transposed back to [n_components, n_samples])
    components = S.T
    
    return {
        'ica': ica,
        'components': components,
        'mixing_matrix': A,
        'sources': S,
        'combined_data': all_channels
    }


def plot_ica_components(ica_result, fs, epoch_length=30, n_epochs=5, save_path=None):
    """
    Plot the ICA components.
    
    Parameters:
    -----------
    ica_result : dict
        Result from apply_ica function
    fs : float
        Sampling frequency in Hz
    epoch_length : int, optional
        Length of one epoch in seconds, default 30
    n_epochs : int, optional
        Number of epochs to display, default 5
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    components = ica_result['components']
    n_components = components.shape[0]
    
    sample_length = fs * epoch_length * n_epochs
    
    fig = plt.figure(figsize=(12, 2*n_components))
    
    for i in range(n_components):
        plt.subplot(n_components, 1, i+1)
        plt.plot(components[i, :sample_length])
        plt.title(f'ICA Component {i+1}')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_mixing_matrix(ica_result, channel_names=None, save_path=None):
    """
    Plot the mixing matrix from ICA.
    
    Parameters:
    -----------
    ica_result : dict
        Result from apply_ica function
    channel_names : list, optional
        List of channel names. If None, uses generic channel labels.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    A = ica_result['mixing_matrix']
    
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(A), aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('ICA Components')
    plt.ylabel('Channels')
    plt.title('Mixing Matrix')
    
    if channel_names:
        plt.yticks(np.arange(len(channel_names)), channel_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def calculate_artifact_correlations(ica_result, artifact_data):
    """
    Calculate correlations between ICA components and artifact channels.
    
    Parameters:
    -----------
    ica_result : dict
        Result from apply_ica function
    artifact_data : dict
        Dictionary of artifact data channels with keys like 'eog', 'emg', 'ecg'
        Each entry should be a numpy array with shape [n_channels, n_samples]
    
    Returns:
    --------
    dict
        Dictionary containing correlation matrix and channel names
    """
    components = ica_result['components']
    
    # Flatten all artifact channels into a list of (name, data) tuples
    artifact_channels = []
    channel_names = []
    
    for artifact_type, data in artifact_data.items():
        for i in range(data.shape[0]):
            channel_name = f"{artifact_type.upper()}{i+1}"
            artifact_channels.append(data[i, :])
            channel_names.append(channel_name)
    
    artifact_channels = np.array(artifact_channels)
    
    # Calculate correlation between each component and each artifact channel
    n_artifact_channels = len(artifact_channels)
    n_components = components.shape[0]
    correlation_matrix = np.zeros((n_artifact_channels, n_components))
    
    for i in range(n_artifact_channels):
        for j in range(n_components):
            correlation = np.abs(np.corrcoef(artifact_channels[i], components[j])[0, 1])
            correlation_matrix[i, j] = correlation
    
    return {
        'correlation_matrix': correlation_matrix,
        'channel_names': channel_names
    }


def plot_correlation_matrix(correlation_result, save_path=None):
    """
    Plot the correlation matrix between ICA components and artifact channels.
    
    Parameters:
    -----------
    correlation_result : dict
        Result from calculate_artifact_correlations function
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    correlation_matrix = correlation_result['correlation_matrix']
    channel_names = correlation_result['channel_names']
    
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(correlation_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Correlation Coefficient')
    plt.xlabel('ICA Components')
    plt.ylabel('Artifact Channels')
    plt.title('Correlation between ICA Components and Artifact Channels')
    
    plt.yticks(np.arange(len(channel_names)), channel_names)
    plt.xticks(np.arange(correlation_matrix.shape[1]), 
              [f'Comp {i+1}' for i in range(correlation_matrix.shape[1])])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def identify_artifact_components(correlation_result, threshold=0.6):
    """
    Identify which ICA components are related to artifacts based on correlation.
    
    Parameters:
    -----------
    correlation_result : dict
        Result from calculate_artifact_correlations function
    threshold : float, optional
        Correlation threshold for component classification, default 0.6
    
    Returns:
    --------
    dict
        Dictionary containing identified components for each artifact type
    """
    correlation_matrix = correlation_result['correlation_matrix']
    channel_names = correlation_result['channel_names']
    
    artifact_types = {}
    for i, channel in enumerate(channel_names):
        artifact_type = channel[:3]  # Extract artifact type (EOG, EMG, ECG)
        if artifact_type not in artifact_types:
            artifact_types[artifact_type] = []
        
        # Find components with correlation above threshold
        high_corr_idx = np.where(correlation_matrix[i, :] > threshold)[0]
        if len(high_corr_idx) > 0:
            print(f"Channel {channel} is highly correlated with components: {[idx+1 for idx in high_corr_idx]}")
            artifact_types[artifact_type].extend(high_corr_idx)
    
    # Remove duplicates
    for key in artifact_types:
        artifact_types[key] = list(set(artifact_types[key]))
    
    # Create a unified list of components to remove
    components_to_remove = []
    for components in artifact_types.values():
        components_to_remove.extend(components)
    components_to_remove = list(set(components_to_remove))
    
    # Create list of components to keep
    n_components = correlation_matrix.shape[1]
    components_to_keep = [i for i in range(n_components) if i not in components_to_remove]
    
    print(f"Components to remove: {[idx+1 for idx in components_to_remove]}")
    print(f"Components to keep: {[idx+1 for idx in components_to_keep]}")
    
    return {
        'by_type': artifact_types,
        'to_remove': components_to_remove,
        'to_keep': components_to_keep
    }


def reconstruct_cleaned_signal(ica_result, artifact_components, n_eeg_channels=2):
    """
    Reconstruct cleaned EEG signal by removing artifact components.
    
    Parameters:
    -----------
    ica_result : dict
        Result from apply_ica function
    artifact_components : dict
        Result from identify_artifact_components function
    n_eeg_channels : int, optional
        Number of EEG channels to extract from the cleaned signal, default 2
    
    Returns:
    --------
    numpy.ndarray
        Cleaned EEG signal with shape [n_eeg_channels, n_samples]
    """
    components_to_remove = artifact_components['to_remove']
    
    # Get sources and mixing matrix
    S = ica_result['sources'].copy()
    A = ica_result['mixing_matrix']
    
    # Zero out artifact components
    for idx in components_to_remove:
        S[:, idx] = 0
    
    # Reconstruct signal
    X_clean = np.dot(S, A.T) + ica_result['ica'].mean_
    
    # Reshape back to original format [channels, samples]
    cleaned_channels = X_clean.T
    
    # Extract only EEG channels
    cleaned_eeg = cleaned_channels[:n_eeg_channels, :]
    
    return cleaned_eeg


def plot_cleaned_comparison(original_eeg, cleaned_eeg, fs, epoch_length=30, n_epochs=5, save_path=None):
    """
    Plot comparison between original and cleaned EEG signals.
    
    Parameters:
    -----------
    original_eeg : numpy.ndarray
        Original EEG signal with shape [n_channels, n_samples]
    cleaned_eeg : numpy.ndarray
        Cleaned EEG signal with shape [n_channels, n_samples]
    fs : float
        Sampling frequency in Hz
    epoch_length : int, optional
        Length of one epoch in seconds, default 30
    n_epochs : int, optional
        Number of epochs to display, default 5
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    n_channels = original_eeg.shape[0]
    samples_to_plot = fs * epoch_length * n_epochs
    
    fig = plt.figure(figsize=(12, 3*n_channels))
    
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(original_eeg[i, :samples_to_plot], 'b-', label='Original')
        plt.plot(cleaned_eeg[i, :samples_to_plot], 'r-', label='ICA Cleaned')
        plt.title(f'EEG Channel {i+1}: Original vs ICA Cleaned')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig 


def extract_frequency_bands_dwt(eeg_signal, fs=125, wavelet='db4', level=6):
    """
    Extract frequency bands from EEG signal using Discrete Wavelet Transform.
    
    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        EEG signal with shape [n_samples] or [n_channels, n_samples]
    fs : float, optional
        Sampling frequency in Hz, default 125
    wavelet : str, optional
        Wavelet family to use, default 'db4'
    level : int, optional
        Decomposition level, default 6
    
    Returns:
    --------
    dict
        Dictionary containing extracted frequency bands:
        - 'slow_delta': ~0-1 Hz
        - 'delta': ~1-4 Hz
        - 'theta': ~4-8 Hz
        - 'alpha': ~8-16 Hz
        - 'beta': ~16-32 Hz
        - 'gamma': ~32+ Hz
    """
    # Ensure eeg_signal is 1D
    if len(eeg_signal.shape) > 1:
        signal = eeg_signal[0, :]  # Use first channel by default
    else:
        signal = eeg_signal
    
    # Perform DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Extract coefficients (1 approximation + 6 details)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    # Create zero arrays with the same shape as the coefficients
    zero_cA6 = np.zeros_like(cA6)
    zero_cD6 = np.zeros_like(cD6)
    zero_cD5 = np.zeros_like(cD5)
    zero_cD4 = np.zeros_like(cD4)
    zero_cD3 = np.zeros_like(cD3)
    zero_cD2 = np.zeros_like(cD2)
    zero_cD1 = np.zeros_like(cD1)
    
    # Reconstruct each frequency band
    # For slow delta band (0-1 Hz): Use cA6
    coeffs_slow_delta = [cA6, zero_cD6, zero_cD5, zero_cD4, zero_cD3, zero_cD2, zero_cD1]
    slow_delta_band = pywt.waverec(coeffs_slow_delta, wavelet)[:len(signal)]
    
    # For delta band (1-4 Hz): Use cD6 (1-2 Hz) and cD5 (2-4 Hz)
    coeffs_delta = [zero_cA6, cD6, cD5, zero_cD4, zero_cD3, zero_cD2, zero_cD1]
    delta_band = pywt.waverec(coeffs_delta, wavelet)[:len(signal)]
    
    # For theta band (4-8 Hz): Use cD4
    coeffs_theta = [zero_cA6, zero_cD6, zero_cD5, cD4, zero_cD3, zero_cD2, zero_cD1]
    theta_band = pywt.waverec(coeffs_theta, wavelet)[:len(signal)]
    
    # For alpha band (8-16 Hz): Use cD3
    coeffs_alpha = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, cD3, zero_cD2, zero_cD1]
    alpha_band = pywt.waverec(coeffs_alpha, wavelet)[:len(signal)]
    
    # For beta band (16-32 Hz): Use cD2
    coeffs_beta = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, zero_cD3, cD2, zero_cD1]
    beta_band = pywt.waverec(coeffs_beta, wavelet)[:len(signal)]
    
    # For gamma band (32-62.5 Hz): Use cD1
    coeffs_gamma = [zero_cA6, zero_cD6, zero_cD5, zero_cD4, zero_cD3, zero_cD2, cD1]
    gamma_band = pywt.waverec(coeffs_gamma, wavelet)[:len(signal)]
    
    return {
        'slow_delta': slow_delta_band,
        'delta': delta_band,
        'theta': theta_band,
        'alpha': alpha_band,
        'beta': beta_band,
        'gamma': gamma_band,
        'coeffs': coeffs  # Return raw coefficients for advanced usage
    }


def plot_frequency_bands(bands, fs, duration=30, save_path=None):
    """
    Plot extracted frequency bands from EEG signal.
    
    Parameters:
    -----------
    bands : dict
        Dictionary containing extracted frequency bands from extract_frequency_bands_dwt
    fs : float
        Sampling frequency in Hz
    duration : int, optional
        Duration to plot in seconds, default 30
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    samples_to_plot = int(fs * duration)
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    band_labels = ['Slow Delta (0-1 Hz)', 'Delta (1-4 Hz)', 'Theta (4-8 Hz)', 
                  'Alpha (8-16 Hz)', 'Beta (16-32 Hz)', 'Gamma (32+ Hz)']
    
    # Get full signal by summing all bands
    full_signal = np.zeros_like(bands[band_names[0]][:samples_to_plot])
    for band in band_names:
        full_signal += bands[band][:samples_to_plot]
    
    fig = plt.figure(figsize=(15, 14))
    
    # Plot original signal
    plt.subplot(7, 1, 1)
    plt.plot(full_signal)
    plt.title('Full EEG Signal')
    
    # Plot each frequency band
    for i, (name, label) in enumerate(zip(band_names, band_labels)):
        plt.subplot(7, 1, i+2)
        plt.plot(bands[name][:samples_to_plot])
        plt.title(label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def calculate_band_powers(eeg_signal, frequency_bands, fs, epoch_length=30):
    """
    Calculate power in each frequency band for each epoch.
    
    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        EEG signal with shape [n_channels, n_samples] or [n_samples]
    frequency_bands : dict
        Dictionary containing extracted frequency bands from extract_frequency_bands_dwt
    fs : float
        Sampling frequency in Hz
    epoch_length : int, optional
        Length of one epoch in seconds, default 30
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'timestamps': Center time of each epoch in seconds
        - 'band_powers': Dictionary with band names as keys and arrays of powers as values
        - 'relative_powers': Dictionary with band names as keys and arrays of relative powers as values
    """
    # Ensure eeg_signal is 1D if it's not already
    if len(eeg_signal.shape) > 1:
        signal = eeg_signal[0, :]  # Use first channel by default
    else:
        signal = eeg_signal
    
    # Calculate number of epochs
    n_samples = len(signal)
    samples_per_epoch = fs * epoch_length
    num_epochs = n_samples // samples_per_epoch
    
    # Initialize arrays for results
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    band_powers = {band: np.zeros(num_epochs) for band in band_names}
    epoch_timestamps = np.zeros(num_epochs)
    
    # Process each epoch
    for epoch_idx in range(num_epochs):
        # Calculate epoch boundaries
        start_idx = epoch_idx * samples_per_epoch
        end_idx = (epoch_idx + 1) * samples_per_epoch
        
        # Store epoch center timestamp (in seconds)
        epoch_timestamps[epoch_idx] = (start_idx + samples_per_epoch/2) / fs
        
        # Calculate power for each band in this epoch
        for band_name in band_names:
            # Get the epoch segment for this band
            band_signal = frequency_bands[band_name][start_idx:end_idx]
            
            # Calculate PSD using Welch's method
            f, Pxx = sig.welch(band_signal, fs=fs, nperseg=min(len(band_signal), fs*4))
            
            # Calculate total power (area under PSD curve)
            total_power = np.sum(Pxx)
            band_powers[band_name][epoch_idx] = total_power
    
    # Calculate relative powers
    total_power_by_epoch = np.zeros(num_epochs)
    for band in band_names:
        total_power_by_epoch += band_powers[band]
    
    relative_powers = {band: band_powers[band] / total_power_by_epoch for band in band_names}
    
    return {
        'timestamps': epoch_timestamps,
        'band_powers': band_powers,
        'relative_powers': relative_powers
    }


def plot_band_powers(band_powers_result, save_path=None, time_unit='hours'):
    """
    Plot power in each frequency band over time.
    
    Parameters:
    -----------
    band_powers_result : dict
        Result from calculate_band_powers function
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    time_unit : str, optional
        Time unit for x-axis ('seconds', 'minutes', 'hours'), default 'hours'
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    band_powers = band_powers_result['band_powers']
    timestamps = band_powers_result['timestamps']
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Convert timestamps to selected time unit
    if time_unit == 'minutes':
        t = timestamps / 60
        xlabel = 'Time (minutes)'
    elif time_unit == 'hours':
        t = timestamps / 3600
        xlabel = 'Time (hours)'
    else:  # default to seconds
        t = timestamps
        xlabel = 'Time (seconds)'
    
    fig = plt.figure(figsize=(15, 12))
    
    for i, band_name in enumerate(band_names):
        plt.subplot(len(band_names), 1, i+1)
        plt.plot(t, band_powers[band_name])
        plt.title(f'{band_name.capitalize()} Band Power')
        plt.xlabel(xlabel)
        plt.ylabel('Power')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_relative_band_powers(band_powers_result, save_path=None, time_unit='hours'):
    """
    Plot stacked relative powers of each frequency band over time.
    
    Parameters:
    -----------
    band_powers_result : dict
        Result from calculate_band_powers function
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    time_unit : str, optional
        Time unit for x-axis ('seconds', 'minutes', 'hours'), default 'hours'
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    relative_powers = band_powers_result['relative_powers']
    timestamps = band_powers_result['timestamps']
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Convert timestamps to selected time unit
    if time_unit == 'minutes':
        t = timestamps / 60
        xlabel = 'Time (minutes)'
    elif time_unit == 'hours':
        t = timestamps / 3600
        xlabel = 'Time (hours)'
    else:  # default to seconds
        t = timestamps
        xlabel = 'Time (seconds)'
    
    # Prepare data for stacked plot
    rel_powers_array = np.zeros((len(band_names), len(timestamps)))
    for i, band_name in enumerate(band_names):
        rel_powers_array[i, :] = relative_powers[band_name]
    
    # Create the stacked plot
    fig = plt.figure(figsize=(15, 8))
    colors = ['darkblue', 'royalblue', 'green', 'orange', 'red', 'purple']
    
    plt.stackplot(t, rel_powers_array, labels=band_names, colors=colors)
    plt.xlabel(xlabel)
    plt.ylabel('Relative Power')
    plt.title('Relative Band Powers Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig 


def detect_sleep_spindles(alpha_band, fs, 
                          threshold_factor_lower=1.0, 
                          threshold_factor_upper=5.0,
                          min_duration=0.4, 
                          max_duration=2.0):
    """
    Detect sleep spindles in EEG alpha/sigma band.
    
    Parameters:
    -----------
    alpha_band : numpy.ndarray
        Alpha band (8-16 Hz) signal with shape [n_samples]
    fs : float
        Sampling frequency in Hz
    threshold_factor_lower : float, optional
        Lower threshold factor, default 1.0
    threshold_factor_upper : float, optional
        Upper threshold factor, default 5.0
    min_duration : float, optional
        Minimum spindle duration in seconds, default 0.4
    max_duration : float, optional
        Maximum spindle duration in seconds, default 2.0
    
    Returns:
    --------
    list
        List of detected spindles, each as a tuple (start_idx, end_idx, duration)
    dict
        Additional data for visualization including envelope and thresholds
    """
    # Calculate the signal envelope
    spindle_envelope = np.abs(sig.hilbert(alpha_band))
    
    # Define thresholds based on the mean envelope
    mean_envelope = np.mean(spindle_envelope)
    lower_threshold = threshold_factor_lower * mean_envelope
    upper_threshold = threshold_factor_upper * mean_envelope
    
    # Create a mask that is True only where envelope is between lower and upper thresholds
    valid_mask = (spindle_envelope >= lower_threshold) & (spindle_envelope <= upper_threshold)
    
    # Scan the mask to detect contiguous segments that qualify as spindles
    spindles = []
    i = 0
    n_samples = len(alpha_band)
    
    while i < n_samples:
        if valid_mask[i]:
            start = i
            # Move forward while we remain within valid_mask == True
            while i < n_samples and valid_mask[i]:
                i += 1
            end = i
            duration = (end - start) / fs
            
            # Check duration constraints
            if min_duration <= duration <= max_duration:
                spindles.append((start, end, duration))
        else:
            i += 1
    
    # Additional data for visualization
    spindle_data = {
        'envelope': spindle_envelope,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold,
        'valid_mask': valid_mask
    }
    
    return spindles, spindle_data


def plot_spindle_detection(alpha_band, spindles, spindle_data, fs, 
                          display_duration=30, save_path=None):
    """
    Plot sleep spindle detection results.
    
    Parameters:
    -----------
    alpha_band : numpy.ndarray
        Alpha band (8-16 Hz) signal with shape [n_samples]
    spindles : list
        List of detected spindles from detect_sleep_spindles
    spindle_data : dict
        Additional data from detect_sleep_spindles
    fs : float
        Sampling frequency in Hz
    display_duration : int, optional
        Duration to display in seconds, default 30
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create time array for x-axis
    t = np.arange(len(alpha_band)) / fs
    
    # Plot only a subset of the data for better visualization
    display_samples = min(display_duration * fs, len(alpha_band))
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.plot(t[:display_samples], alpha_band[:display_samples])
    plt.title('Alpha/Sigma Band (8-16 Hz)')
    
    # Plot envelope and thresholds
    plt.subplot(3, 1, 2)
    plt.plot(t[:display_samples], spindle_data['envelope'][:display_samples])
    plt.axhline(y=spindle_data['lower_threshold'], color='r', linestyle='--', 
               label=f'Lower Threshold ({spindle_data["lower_threshold"]:.2f})')
    plt.axhline(y=spindle_data['upper_threshold'], color='g', linestyle='--', 
               label=f'Upper Threshold ({spindle_data["upper_threshold"]:.2f})')
    plt.legend()
    plt.title('Signal Envelope with Thresholds')
    
    # Highlight detected spindles
    plt.subplot(3, 1, 3)
    plt.plot(t[:display_samples], alpha_band[:display_samples])
    
    # Highlight detected spindle intervals
    for start, end, duration in spindles:
        if start < display_samples:  # Only highlight spindles in the displayed time window
            plt.axvspan(t[start], t[end], color='yellow', alpha=0.3)
    
    plt.title(f'Detected Spindles (Found {len(spindles)} spindles)')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def calculate_spindle_density(spindles, total_duration_seconds):
    """
    Calculate spindle density (spindles per minute).
    
    Parameters:
    -----------
    spindles : list
        List of detected spindles from detect_sleep_spindles
    total_duration_seconds : float
        Total duration of the EEG recording in seconds
    
    Returns:
    --------
    float
        Spindle density in spindles per minute
    """
    total_minutes = total_duration_seconds / 60
    spindle_density = len(spindles) / total_minutes
    return spindle_density


def calculate_spindle_density_by_epoch(spindles, fs, epoch_length=30, num_epochs=None):
    """
    Calculate spindle density for each epoch.
    
    Parameters:
    -----------
    spindles : list
        List of detected spindles from detect_sleep_spindles
    fs : float
        Sampling frequency in Hz
    epoch_length : int, optional
        Length of one epoch in seconds, default 30
    num_epochs : int, optional
        Total number of epochs. If None, calculated from the last spindle.
    
    Returns:
    --------
    numpy.ndarray
        Array of spindle densities per minute for each epoch
    """
    samples_per_epoch = fs * epoch_length
    
    # If num_epochs is not provided, calculate from the last spindle
    if num_epochs is None:
        if not spindles:
            return np.array([])
        
        last_spindle_end = max([end for _, end, _ in spindles])
        num_epochs = int(np.ceil(last_spindle_end / samples_per_epoch))
    
    # Initialize array to count spindles per epoch
    spindle_count = np.zeros(num_epochs)
    
    # Count spindles in each epoch
    for start, _, _ in spindles:
        epoch_idx = int(start / samples_per_epoch)
        if epoch_idx < num_epochs:
            spindle_count[epoch_idx] += 1
    
    # Convert to spindles per minute (30-second epochs = 2 epochs per minute)
    spindle_density = spindle_count / (epoch_length / 60)
    
    return spindle_density


def plot_spindle_density(spindle_density, epoch_timestamps, sleep_stages=None, 
                        save_path=None, time_unit='hours'):
    """
    Plot spindle density over time, optionally with sleep stages.
    
    Parameters:
    -----------
    spindle_density : numpy.ndarray
        Array of spindle densities per minute for each epoch
    epoch_timestamps : numpy.ndarray
        Array of timestamps for each epoch in seconds
    sleep_stages : numpy.ndarray, optional
        Array of sleep stages for each epoch. If None, not plotted.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    time_unit : str, optional
        Time unit for x-axis ('seconds', 'minutes', 'hours'), default 'hours'
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert timestamps to selected time unit
    if time_unit == 'minutes':
        t = epoch_timestamps / 60
        xlabel = 'Time (minutes)'
    elif time_unit == 'hours':
        t = epoch_timestamps / 3600
        xlabel = 'Time (hours)'
    else:  # default to seconds
        t = epoch_timestamps
        xlabel = 'Time (seconds)'
    
    # Create figure with one or two subplots depending on whether sleep stages are provided
    if sleep_stages is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                      gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot hypnogram on top subplot
        ax1.step(t, sleep_stages, where='post', color='black', linewidth=1.5)
        ax1.set_yticks([0, 1, 2, 3, 4, 5])
        ax1.set_yticklabels(['REM', '', 'N3', 'N2', 'N1', 'Wake'])
        ax1.set_ylabel('Sleep Stage')
        ax1.set_title('Sleep Stages and Spindle Density')
        ax1.grid(True, axis='y')
        ax1.set_ylim(-0.5, 5.5)
        
        # Plot spindle density on bottom subplot
        ax2.bar(t, spindle_density, width=0.02)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Spindles per Minute')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Ensure the x-axes are aligned
        ax1.set_xlim(ax2.get_xlim())
        ax1.tick_params(labelbottom=False)  # Hide x-ticks on top plot
    else:
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(t, spindle_density, width=0.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Spindles per Minute')
        ax.set_title('Spindle Density')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig 


def extract_eeg_features(eeg_signal, bands, spindles, fs, epoch_length=30):
    """
    Extract comprehensive set of EEG features for analysis.
    
    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        EEG signal with shape [n_channels, n_samples] or [n_samples]
    bands : dict
        Dictionary of frequency bands from extract_frequency_bands_dwt
    spindles : list
        List of detected spindles from detect_sleep_spindles
    fs : float
        Sampling frequency in Hz
    epoch_length : int, optional
        Length of one epoch in seconds, default 30
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'timestamps': Center time of each epoch in seconds
        - 'absolute_power': Dictionary of absolute powers for each band
        - 'relative_power': Dictionary of relative powers for each band
        - 'power_ratios': Dictionary of power ratios between bands
        - 'spectral_features': Dictionary of spectral features
        - 'time_domain': Dictionary of time-domain features
    """
    # Ensure eeg_signal is 1D if it's not already
    if len(eeg_signal.shape) > 1:
        signal = eeg_signal[0, :]  # Use first channel by default
    else:
        signal = eeg_signal
    
    # Calculate number of epochs
    n_samples = len(signal)
    samples_per_epoch = fs * epoch_length
    num_epochs = n_samples // samples_per_epoch
    
    # Initialize timestamps
    epoch_timestamps = np.zeros(num_epochs)
    
    # Initialize dictionaries to store features
    features = {
        'timestamps': epoch_timestamps,
        'absolute_power': {},
        'relative_power': {},
        'power_ratios': {},
        'spectral_features': {},
        'time_domain': {}
    }
    
    # Calculate band powers
    band_powers_result = calculate_band_powers(signal, bands, fs, epoch_length)
    
    # Store band powers
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    total_power = np.zeros(num_epochs)
    
    for band in band_names:
        features['absolute_power'][band] = band_powers_result['band_powers'][band]
        total_power += band_powers_result['band_powers'][band]
        
    # Calculate and store relative powers
    for band in band_names:
        features['relative_power'][band] = features['absolute_power'][band] / total_power
    
    # Calculate and store power ratios
    ratio_pairs = [
        ('slow_delta', 'delta'),  # Slow delta to delta ratio
        ('delta', 'theta'),       # Delta to theta ratio
        ('theta', 'alpha'),       # Theta to alpha ratio
        ('alpha', 'beta'),        # Alpha to beta ratio
        ('delta', 'beta'),        # Delta to beta ratio (often used in sleep studies)
        ('theta', 'beta'),        # Theta to beta ratio (attention/concentration)
        ('slow_delta', 'gamma'),  # Slow delta to gamma ratio
        ('delta', 'gamma')        # Delta to gamma ratio
    ]
    
    for band1, band2 in ratio_pairs:
        ratio_name = f"{band1}_to_{band2}"
        # Avoid division by zero with small epsilon
        features['power_ratios'][ratio_name] = features['absolute_power'][band1] / (features['absolute_power'][band2] + 1e-10)
    
    # Calculate and store spectral features
    spectral_entropy = np.zeros(num_epochs)
    spectral_edge_50 = np.zeros(num_epochs)
    spectral_edge_95 = np.zeros(num_epochs)
    
    for epoch_idx in range(num_epochs):
        # Calculate epoch boundaries
        start_idx = epoch_idx * samples_per_epoch
        end_idx = (epoch_idx + 1) * samples_per_epoch
        
        # Store epoch timestamp (in seconds)
        epoch_timestamps[epoch_idx] = (start_idx + samples_per_epoch/2) / fs
        
        # Extract epoch data
        epoch_data = signal[start_idx:end_idx]
        
        # Calculate PSD for spectral features
        f, Pxx = sig.welch(epoch_data, fs=fs, nperseg=min(len(epoch_data), fs*4))
        
        # Normalize PSD for entropy calculation
        Pxx_norm = Pxx / np.sum(Pxx)
        
        # Calculate spectral entropy
        spectral_entropy[epoch_idx] = entropy(Pxx_norm)
        
        # Calculate spectral edge frequencies
        cumsum_pxx = np.cumsum(Pxx) / np.sum(Pxx)
        idx_50 = np.argmax(cumsum_pxx >= 0.5)
        idx_95 = np.argmax(cumsum_pxx >= 0.95)
        
        spectral_edge_50[epoch_idx] = f[idx_50]
        spectral_edge_95[epoch_idx] = f[idx_95]
    
    features['spectral_features']['spectral_entropy'] = spectral_entropy
    features['spectral_features']['spectral_edge_50'] = spectral_edge_50
    features['spectral_features']['spectral_edge_95'] = spectral_edge_95
    
    # Calculate and store time-domain features
    variance = np.zeros(num_epochs)
    skewness_vals = np.zeros(num_epochs)
    kurtosis_vals = np.zeros(num_epochs)
    zero_crossings = np.zeros(num_epochs)
    hjorth_activity = np.zeros(num_epochs)
    hjorth_mobility = np.zeros(num_epochs)
    hjorth_complexity = np.zeros(num_epochs)
    
    for epoch_idx in range(num_epochs):
        start_idx = epoch_idx * samples_per_epoch
        end_idx = (epoch_idx + 1) * samples_per_epoch
        epoch_data = signal[start_idx:end_idx]
        
        # Basic statistics
        variance[epoch_idx] = np.var(epoch_data)
        skewness_vals[epoch_idx] = skew(epoch_data)
        kurtosis_vals[epoch_idx] = kurtosis(epoch_data)
        
        # Zero crossings
        zero_crossings[epoch_idx] = np.sum(np.diff(np.signbit(epoch_data)))
        
        # Hjorth parameters
        # First derivative
        d1 = np.diff(epoch_data)
        # Second derivative
        d2 = np.diff(d1)
        
        # Activity = variance of signal
        hjorth_activity[epoch_idx] = np.var(epoch_data)
        
        # Mobility = sqrt(variance of first derivative / variance of signal)
        mobility_signal = np.sqrt(np.var(d1) / (np.var(epoch_data) + 1e-10))
        hjorth_mobility[epoch_idx] = mobility_signal
        
        # Complexity = mobility of first derivative / mobility of signal
        mobility_d1 = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10))
        hjorth_complexity[epoch_idx] = mobility_d1 / (mobility_signal + 1e-10)
    
    features['time_domain']['variance'] = variance
    features['time_domain']['skewness'] = skewness_vals
    features['time_domain']['kurtosis'] = kurtosis_vals
    features['time_domain']['zero_crossings'] = zero_crossings
    features['time_domain']['hjorth_activity'] = hjorth_activity
    features['time_domain']['hjorth_mobility'] = hjorth_mobility
    features['time_domain']['hjorth_complexity'] = hjorth_complexity
    
    # Calculate spindle density by epoch
    spindle_density = calculate_spindle_density_by_epoch(spindles, fs, epoch_length, num_epochs)
    features['time_domain']['spindle_density'] = spindle_density
    
    return features


def save_features(features, output_mat=None, output_csv=None):
    """
    Save extracted features to MAT and/or CSV files.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features from extract_eeg_features
    output_mat : str, optional
        Path to save MAT file. If None, no MAT file is saved.
    output_csv : str, optional
        Path to save CSV file. If None, no CSV file is saved.
    
    Returns:
    --------
    None
    """
    # Save to MAT file
    if output_mat:
        sio.savemat(output_mat, features)
        print(f"Features saved to {output_mat}")
    
    # Save to CSV file
    if output_csv:
        import pandas as pd
        
        # Create a DataFrame with timestamps
        feature_df = pd.DataFrame({'timestamp': features['timestamps']})
        
        # Add all calculated features to the DataFrame
        for category in ['absolute_power', 'relative_power', 'power_ratios', 
                        'spectral_features', 'time_domain']:
            for feature_name, feature_values in features[category].items():
                col_name = f"{category.split('_')[0]}_{feature_name}"
                feature_df[col_name] = feature_values
        
        # Save to CSV
        feature_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")


def analyze_features_by_sleep_stage(features, sleep_stages, output_dir=None):
    """
    Analyze EEG features by sleep stage and create visualizations.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features from extract_eeg_features
    sleep_stages : numpy.ndarray
        Array of sleep stages for each epoch
    output_dir : str, optional
        Directory to save output plots. If None, plots are not saved.
    
    Returns:
    --------
    dict
        Dictionary containing average values for each feature by sleep stage
    """
    # Define stage mapping for labels
    stage_mapping = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
    
    # Get unique stages excluding invalid stages
    unique_stages = np.unique(sleep_stages)
    valid_stages = [stage for stage in unique_stages if stage in stage_mapping]
    
    # Initialize dictionary to store average values by stage
    avg_by_stage = {}
    
    # Process each feature category
    for category in ['absolute_power', 'relative_power', 'power_ratios', 
                    'spectral_features', 'time_domain']:
        avg_by_stage[category] = {}
        
        for feature_name, feature_values in features[category].items():
            # Skip any features with incompatible shapes
            if len(feature_values) != len(sleep_stages):
                continue
                
            # Calculate average by stage
            stage_averages = {}
            for stage in valid_stages:
                stage_indices = np.where(sleep_stages == stage)[0]
                if len(stage_indices) > 0:
                    stage_averages[stage_mapping[stage]] = np.mean(feature_values[stage_indices])
            
            avg_by_stage[category][feature_name] = stage_averages
    
    # Create visualizations if output_dir is provided
    if output_dir:
        # 1. Delta vs Theta scatter plot by sleep stage
        plt.figure(figsize=(10, 8))
        for stage in valid_stages:
            stage_indices = np.where(sleep_stages == stage)[0]
            if len(stage_indices) > 0:
                plt.scatter(
                    features['relative_power']['delta'][stage_indices],
                    features['relative_power']['theta'][stage_indices],
                    alpha=0.3,
                    label=stage_mapping[stage]
                )
        
        plt.xlabel('Relative Delta Power')
        plt.ylabel('Relative Theta Power')
        plt.title('Delta vs. Theta Relative Power by Sleep Stage')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'delta_vs_theta.png'))
        plt.close()
        
        # 2. Spectral entropy boxplot by stage
        plt.figure(figsize=(10, 6))
        stage_entropies = []
        stage_labels = []
        
        for stage in valid_stages:
            stage_indices = np.where(sleep_stages == stage)[0]
            if len(stage_indices) > 0:
                stage_entropies.append(features['spectral_features']['spectral_entropy'][stage_indices])
                stage_labels.append(stage_mapping[stage])
        
        plt.boxplot(stage_entropies)
        plt.xticks(range(1, len(stage_labels) + 1), stage_labels)
        plt.ylabel('Spectral Entropy')
        plt.title('Spectral Entropy by Sleep Stage')
        plt.savefig(os.path.join(output_dir, 'spectral_entropy_by_stage.png'))
        plt.close()
        
        # 3. Spindle density by stage (bar chart)
        plt.figure(figsize=(10, 6))
        stage_densities = []
        
        for stage in valid_stages:
            if stage in stage_mapping:
                stage_indices = np.where(sleep_stages == stage)[0]
                if len(stage_indices) > 0 and 'spindle_density' in features['time_domain']:
                    avg_density = np.mean(features['time_domain']['spindle_density'][stage_indices])
                    stage_densities.append((stage_mapping[stage], avg_density))
        
        if stage_densities:
            labels, values = zip(*sorted(stage_densities, key=lambda x: x[1], reverse=True))
            plt.bar(labels, values)
            plt.ylabel('Spindles per Minute')
            plt.title('Average Spindle Density by Sleep Stage')
            plt.savefig(os.path.join(output_dir, 'spindle_density_by_stage.png'))
            plt.close()
        
        # 4. Delta/Beta ratio by stage (boxplot)
        plt.figure(figsize=(10, 6))
        stage_ratios = []
        stage_labels = []
        
        for stage in valid_stages:
            stage_indices = np.where(sleep_stages == stage)[0]
            if len(stage_indices) > 0 and 'delta_to_beta' in features['power_ratios']:
                stage_ratios.append(features['power_ratios']['delta_to_beta'][stage_indices])
                stage_labels.append(stage_mapping[stage])
        
        if stage_ratios:
            plt.boxplot(stage_ratios)
            plt.xticks(range(1, len(stage_labels) + 1), stage_labels)
            plt.ylabel('Delta/Beta Ratio')
            plt.title('Delta/Beta Ratio by Sleep Stage')
            plt.savefig(os.path.join(output_dir, 'delta_beta_ratio_by_stage.png'))
            plt.close()
    
    return avg_by_stage


def create_feature_summary_plots(features, sleep_stages=None, output_dir=None):
    """
    Create summary plots of extracted features.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features from extract_eeg_features
    sleep_stages : numpy.ndarray, optional
        Array of sleep stages for each epoch. If None, sleep stage plots are not created.
    output_dir : str, optional
        Directory to save output plots. If None, plots are not saved.
    
    Returns:
    --------
    None
    """
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Time unit for plots (can be 'seconds', 'minutes', or 'hours')
    time_unit = 'hours'
    
    # Convert timestamps to selected time unit
    timestamps = features['timestamps']
    if time_unit == 'minutes':
        t = timestamps / 60
        xlabel = 'Time (minutes)'
    elif time_unit == 'hours':
        t = timestamps / 3600
        xlabel = 'Time (hours)'
    else:  # default to seconds
        t = timestamps
        xlabel = 'Time (seconds)'
    
    # 1. Relative band powers stacked plot
    plt.figure(figsize=(15, 8))
    
    # Prepare data for stacked plot
    band_names = ['slow_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    rel_powers = np.zeros((len(band_names), len(timestamps)))
    
    for i, band in enumerate(band_names):
        rel_powers[i, :] = features['relative_power'][band]
    
    colors = ['darkblue', 'royalblue', 'green', 'orange', 'red', 'purple']
    plt.stackplot(t, rel_powers, labels=band_names, colors=colors)
    plt.xlabel(xlabel)
    plt.ylabel('Relative Power')
    plt.title('Relative Band Powers Over Time')
    plt.legend(loc='upper right')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'relative_band_powers.png'))
    
    plt.close()
    
    # 2. Spindle density over time
    if 'spindle_density' in features['time_domain']:
        plt.figure(figsize=(15, 6))
        plt.bar(t, features['time_domain']['spindle_density'], width=0.02)
        plt.xlabel(xlabel)
        plt.ylabel('Spindles per Minute')
        plt.title('Spindle Density')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'spindle_density_over_time.png'))
        
        plt.close()
    
    # 3. If sleep stages are provided, create combined plots
    if sleep_stages is not None:
        # Stage mapping for labels
        stage_mapping = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
        
        # Combined plot with hypnogram and band powers
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                      gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot hypnogram on top subplot
        ax1.step(t, sleep_stages, where='post', color='black', linewidth=1.5)
        ax1.set_yticks([0, 1, 2, 3, 4, 5])
        ax1.set_yticklabels(['REM', '', 'N3', 'N2', 'N1', 'Wake'])
        ax1.set_ylabel('Sleep Stage')
        ax1.set_title('Sleep Architecture and EEG Band Powers')
        ax1.grid(True, axis='y')
        ax1.set_ylim(-0.5, 5.5)
        
        # Plot stacked relative powers on bottom subplot
        ax2.stackplot(t, rel_powers, labels=band_names, colors=colors)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Relative Power')
        ax2.legend(loc='upper right')
        
        # Ensure the x-axes are aligned
        ax1.set_xlim(ax2.get_xlim())
        ax1.tick_params(labelbottom=False)  # Hide x-ticks on top plot
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'sleep_stages_with_band_powers.png'))
        
        plt.close()
        
        # Combined plot with hypnogram and spindle density
        if 'spindle_density' in features['time_domain']:
            plot_spindle_density(
                features['time_domain']['spindle_density'], 
                features['timestamps'],
                sleep_stages,
                os.path.join(output_dir, 'sleep_stages_with_spindles.png') if output_dir else None,
                time_unit
            )


def prepare_classification_data(features_df, sleep_stages, valid_stages=None):
    """
    Prepare data for sleep stage classification.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features (can be loaded from CSV)
    sleep_stages : numpy.ndarray
        Array of sleep stages for each epoch
    valid_stages : list, optional
        List of valid sleep stage values to include. If None, uses [0, 2, 3, 4, 5].
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'X': Feature matrix
        - 'y': Target labels
        - 'feature_names': List of feature names
        - 'stage_mapping': Dictionary mapping stage codes to names
        - 'encoded_classes': List of stage names in encoded order
    """
    import pandas as pd
    
    # Default valid stages if not provided
    if valid_stages is None:
        valid_stages = [0, 2, 3, 4, 5]  # REM, N3, N2, N1, Wake
    
    # Define stage mapping for labels
    stage_mapping = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
    
    # Ensure features_df is a DataFrame
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame")
    
    # Remove any NaN values
    features_df = features_df.dropna()
    
    # Create a copy with sleep stages
    df = features_df.copy()
    df['sleep_stage'] = sleep_stages[:len(df)]
    
    # Filter for valid sleep stages
    df = df[df['sleep_stage'].isin(valid_stages)]
    
    # Define X (features) and y (target labels)
    if 'timestamp' in df.columns:
        X = df.drop(['timestamp', 'sleep_stage'], axis=1)
    else:
        X = df.drop(['sleep_stage'], axis=1)
    
    y = df['sleep_stage']
    
    # Print the number of samples for each sleep stage
    print("Sleep stage distribution:")
    stage_counts = y.value_counts().sort_index()
    for stage, count in stage_counts.items():
        stage_name = stage_mapping.get(stage, f'Unknown {stage}')
        print(f"  {stage_name}: {count} samples")
    
    # Use a LabelEncoder to transform non-consecutive labels to consecutive integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Store the original class names in the same order as the encoded labels
    encoded_classes = [stage_mapping[stage] for stage in label_encoder.classes_]
    
    return {
        'X': X,
        'y': y_encoded,
        'feature_names': X.columns.tolist(),
        'stage_mapping': stage_mapping,
        'encoded_classes': encoded_classes,
        'label_encoder': label_encoder
    }


def select_important_features(X_train, y_train, n_features=20):
    """
    Select the most important features for classification.
    
    Parameters:
    -----------
    X_train : numpy.ndarray or pandas.DataFrame
        Training feature matrix
    y_train : numpy.ndarray
        Training target labels
    n_features : int, optional
        Number of top features to select, default 20
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'selector': Fitted feature selector
        - 'selected_features': Names of selected features (if X_train is a DataFrame)
        - 'importances': Feature importances
    """
    from sklearn.feature_selection import SelectFromModel
    
    # Use Random Forest to get feature importance
    preliminary_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    preliminary_rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = preliminary_rf.feature_importances_
    
    # Select top features
    selector = SelectFromModel(preliminary_rf, threshold=-np.inf, max_features=n_features)
    selector.fit(X_train, y_train)
    
    # Get names of selected features if input is DataFrame
    selected_features = None
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        print(f"\nTop {n_features} selected features:")
        for i, feature in enumerate(selected_features):
            print(f"{i+1}. {feature}")
    
    return {
        'selector': selector,
        'selected_features': selected_features,
        'importances': importances
    }


def train_sleep_stage_classifiers(X_train, y_train, encoded_classes):
    """
    Train multiple classifiers for sleep stage classification.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target labels
    encoded_classes : list
        List of class names in encoded order
    
    Returns:
    --------
    dict
        Dictionary containing trained models and their accuracy scores
    """
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 1. Random Forest Classifier
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # 2. Support Vector Machine
    print("\nTraining SVM classifier...")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    
    # 3. XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(encoded_classes),
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    return {
        'scaler': scaler,
        'random_forest': rf_model,
        'svm': svm_model,
        'xgboost': xgb_model
    }


def evaluate_classifiers(classifiers, X_val, y_val, encoded_classes, output_dir=None):
    """
    Evaluate sleep stage classifiers and generate performance metrics.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary of trained classifiers from train_sleep_stage_classifiers
    X_val : numpy.ndarray
        Validation feature matrix
    y_val : numpy.ndarray
        Validation target labels
    encoded_classes : list
        List of class names in encoded order
    output_dir : str, optional
        Directory to save output plots and results. If None, plots are not saved.
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for each classifier
    """
    # Scale validation data
    X_val_scaled = classifiers['scaler'].transform(X_val)
    
    # Evaluate each classifier
    results = {}
    
    for model_name in ['random_forest', 'svm', 'xgboost']:
        model = classifiers[model_name]
        
        # Make predictions
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'report': classification_report(y_val, y_pred, target_names=encoded_classes, output_dict=True)
        }
        
        print(f"\n--- {model_name.replace('_', ' ').title()} Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred, target_names=encoded_classes))
        
        # Create confusion matrix
        if output_dir:
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=encoded_classes,
                        yticklabels=encoded_classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()
    
    # Find the best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    print(f"\nBest model: {best_model.replace('_', ' ').title()} with accuracy {best_accuracy:.4f}")
    
    # Create a comparison bar chart
    if output_dir:
        plt.figure(figsize=(10, 6))
        accuracies = [results[model]['accuracy'] for model in ['random_forest', 'svm', 'xgboost']]
        model_names = ['Random Forest', 'SVM', 'XGBoost']
        
        plt.bar(model_names, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()
    
    return {
        'results': results,
        'best_model': best_model,
        'best_accuracy': best_accuracy
    }


def save_classification_model(classifiers, best_model, selector, feature_names, 
                            selected_features, encoded_classes, output_path):
    """
    Save the best classification model and preprocessing components.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary of trained classifiers
    best_model : str
        Name of the best model (key in classifiers dict)
    selector : object
        Feature selector
    feature_names : list
        Names of all features
    selected_features : list
        Names of selected features
    encoded_classes : list
        List of class names in encoded order
    output_path : str
        Path to save the model
    
    Returns:
    --------
    None
    """
    import pickle
    
    # Create model package
    model_package = {
        'model': classifiers[best_model],
        'model_type': best_model,
        'scaler': classifiers['scaler'],
        'selector': selector,
        'feature_names': feature_names,
        'selected_features': selected_features,
        'encoded_classes': encoded_classes
    }
    
    # Save to file
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\nBest model saved to {output_path}")


def predict_with_saved_model(model_path, new_data):
    """
    Use a saved model to predict sleep stages for new data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    new_data : numpy.ndarray or pandas.DataFrame
        New data for prediction
    
    Returns:
    --------
    numpy.ndarray
        Predicted sleep stages
    """
    import pickle
    
    # Load model package
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    # Extract components
    model = model_package['model']
    scaler = model_package['scaler']
    selector = model_package['selector']
    feature_names = model_package['feature_names']
    encoded_classes = model_package['encoded_classes']
    
    # If input is DataFrame, ensure it has the correct features
    if hasattr(new_data, 'columns'):
        # Check if all required features are present
        missing_features = [feat for feat in feature_names if feat not in new_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Reorder columns to match training data
        new_data = new_data[feature_names]
    
    # Apply feature selection and scaling
    X_selected = selector.transform(new_data)
    X_scaled = scaler.transform(X_selected)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # If model returns probabilities, get class indices
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return y_pred, encoded_classes


def complete_eeg_pipeline(eeg_file, hypnogram_file=None, output_dir=None):
    """
    Run the complete EEG processing pipeline from data loading to classification.
    
    Parameters:
    -----------
    eeg_file : str
        Path to the EEG data file (.mat format)
    hypnogram_file : str, optional
        Path to the hypnogram file (.mat format). If None, sleep stage analysis is skipped.
    output_dir : str, optional
        Directory to save output files and plots. If None, uses default.
    
    Returns:
    --------
    dict
        Dictionary containing results from the pipeline
    """
    # Setup output directory
    if output_dir is None:
        output_dir = setup_output_directory()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Starting EEG processing pipeline. Output will be saved to: {output_dir}")
    results = {}
    
    # 1. Load EEG data
    print("\n1. Loading EEG data...")
    data = load_data(eeg_file)
    
    # Sample rate - update if different in your data
    fs = 125  # Hz
    results['fs'] = fs
    
    # 2. Apply bandpass filter to EEG channels
    print("\n2. Filtering EEG data...")
    original_eeg = data['eeg'].copy()
    filtered_eeg = bandpass_filter(data['eeg'], fs, low_cutoff=0.5, high_cutoff=45)
    
    # Plot and save filtered vs original comparison
    plot_eeg_comparison(
        original_eeg, 
        filtered_eeg, 
        fs, 
        duration=30, 
        title="Original vs Filtered EEG",
        save_path=os.path.join(output_dir, 'filtered_eeg.png')
    )
    
    # 3. Apply ICA for artifact removal
    print("\n3. Applying ICA for artifact removal...")
    
    # Prepare artifact data
    artifact_data = {}
    if 'eog' in data:
        artifact_data['eog'] = data['eog']
    if 'emg' in data:
        artifact_data['emg'] = data['emg']
    if 'ecg' in data:
        artifact_data['ecg'] = data['ecg']
    
    # Apply ICA
    ica_result = apply_ica(filtered_eeg, artifact_data)
    
    # Plot ICA components
    plot_ica_components(
        ica_result, 
        fs, 
        epoch_length=30, 
        n_epochs=5,
        save_path=os.path.join(output_dir, 'ica_components.png')
    )
    
    # Plot mixing matrix
    channel_names = ['EEG1', 'EEG2']
    for key in artifact_data:
        for i in range(artifact_data[key].shape[0]):
            channel_names.append(f"{key.upper()}{i+1}")
    
    plot_mixing_matrix(
        ica_result, 
        channel_names=channel_names,
        save_path=os.path.join(output_dir, 'mixing_matrix.png')
    )
    
    # Calculate correlations with artifact channels
    correlation_result = calculate_artifact_correlations(ica_result, artifact_data)
    
    # Plot correlation matrix
    plot_correlation_matrix(
        correlation_result,
        save_path=os.path.join(output_dir, 'correlation_matrix.png')
    )
    
    # Identify artifact components
    artifact_components = identify_artifact_components(correlation_result, threshold=0.6)
    
    # Reconstruct cleaned signal
    cleaned_eeg = reconstruct_cleaned_signal(ica_result, artifact_components, n_eeg_channels=2)
    
    # Plot cleaned vs filtered comparison
    plot_cleaned_comparison(
        filtered_eeg, 
        cleaned_eeg, 
        fs, 
        epoch_length=30, 
        n_epochs=5,
        save_path=os.path.join(output_dir, 'cleaned_eeg.png')
    )
    
    # 4. Extract frequency bands using DWT
    print("\n4. Extracting frequency bands...")
    bands = extract_frequency_bands_dwt(cleaned_eeg[0, :], fs=fs)
    
    # Plot frequency bands
    plot_frequency_bands(
        bands, 
        fs, 
        duration=30,
        save_path=os.path.join(output_dir, 'frequency_bands.png')
    )
    
    # 5. Calculate band powers by epoch
    print("\n5. Calculating band powers by epoch...")
    band_powers_result = calculate_band_powers(cleaned_eeg[0, :], bands, fs, epoch_length=30)
    
    # Plot band powers
    plot_band_powers(
        band_powers_result,
        save_path=os.path.join(output_dir, 'band_powers.png')
    )
    
    # Plot relative band powers
    plot_relative_band_powers(
        band_powers_result,
        save_path=os.path.join(output_dir, 'relative_band_powers.png')
    )
    
    # 6. Detect sleep spindles
    print("\n6. Detecting sleep spindles...")
    spindles, spindle_data = detect_sleep_spindles(
        bands['alpha'], 
        fs,
        threshold_factor_lower=1.0,
        threshold_factor_upper=5.0
    )
    
    print(f"Detected {len(spindles)} sleep spindles")
    
    # Plot spindle detection
    plot_spindle_detection(
        bands['alpha'], 
        spindles, 
        spindle_data, 
        fs,
        display_duration=30,
        save_path=os.path.join(output_dir, 'spindle_detection.png')
    )
    
    # Calculate spindle density by epoch
    spindle_density = calculate_spindle_density_by_epoch(
        spindles, 
        fs, 
        epoch_length=30,
        num_epochs=len(band_powers_result['timestamps'])
    )
    
    # 7. Extract comprehensive features
    print("\n7. Extracting EEG features...")
    features = extract_eeg_features(cleaned_eeg[0, :], bands, spindles, fs, epoch_length=30)
    
    # Save features to files
    save_features(
        features,
        output_mat=os.path.join(output_dir, 'eeg_features.mat'),
        output_csv=os.path.join(output_dir, 'eeg_features.csv')
    )
    
    # Create feature summary plots
    create_feature_summary_plots(
        features,
        output_dir=output_dir
    )
    
    # 8. Load and analyze sleep stages if hypnogram file is provided
    if hypnogram_file:
        print("\n8. Analyzing sleep stages...")
        # Load sleep stages
        sleep_stages_data = sio.loadmat(hypnogram_file)
        
        # Different files may have different variable names
        stage_var_name = 'stages'
        if stage_var_name in sleep_stages_data:
            sleep_stages = sleep_stages_data[stage_var_name]
            
            # Convert to 1D array if needed
            if sleep_stages.ndim > 1:
                sleep_stages = sleep_stages.flatten()
            
            # Resample if needed to match number of epochs
            num_epochs = len(band_powers_result['timestamps'])
            if len(sleep_stages) != num_epochs:
                print(f"Resampling sleep stages from {len(sleep_stages)} to {num_epochs}")
                
                # Simple resampling using most frequent value in each chunk
                resample_factor = len(sleep_stages) / num_epochs
                resampled_stages = np.zeros(num_epochs)
                
                for i in range(num_epochs):
                    start_idx = int(i * resample_factor)
                    end_idx = int((i + 1) * resample_factor)
                    
                    # Ensure indices are within bounds
                    start_idx = max(0, min(start_idx, len(sleep_stages)-1))
                    end_idx = max(start_idx+1, min(end_idx, len(sleep_stages)))
                    
                    # Extract chunk and use most frequent value
                    chunk = sleep_stages[start_idx:end_idx]
                    values, counts = np.unique(chunk, return_counts=True)
                    resampled_stages[i] = values[np.argmax(counts)]
                
                sleep_stages = resampled_stages
            
            # Create plots with sleep stages
            create_feature_summary_plots(
                features,
                sleep_stages=sleep_stages,
                output_dir=output_dir
            )
            
            # Analyze features by sleep stage
            analyze_features_by_sleep_stage(
                features,
                sleep_stages,
                output_dir=output_dir
            )
            
            # Plot spindle density with sleep stages
            plot_spindle_density(
                spindle_density,
                band_powers_result['timestamps'],
                sleep_stages,
                save_path=os.path.join(output_dir, 'spindle_density_by_stage.png')
            )
            
            # Prepare for classification (if there are enough epochs)
            if num_epochs > 50:  # Arbitrary minimum for meaningful classification
                print("\n9. Building sleep stage classification models...")
                
                # Load features from CSV to ensure correct format
                import pandas as pd
                features_df = pd.read_csv(os.path.join(output_dir, 'eeg_features.csv'))
                
                # Prepare data
                classification_data = prepare_classification_data(
                    features_df,
                    sleep_stages
                )
                
                # Split into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(
                    classification_data['X'],
                    classification_data['y'],
                    test_size=0.4,
                    random_state=42,
                    stratify=classification_data['y']
                )
                
                # Select important features
                feature_selection = select_important_features(X_train, y_train, n_features=20)
                
                # Apply feature selection
                X_train_selected = feature_selection['selector'].transform(X_train)
                X_val_selected = feature_selection['selector'].transform(X_val)
                
                # Train classifiers
                classifiers = train_sleep_stage_classifiers(
                    X_train_selected,
                    y_train,
                    classification_data['encoded_classes']
                )
                
                # Evaluate classifiers
                evaluation = evaluate_classifiers(
                    classifiers,
                    X_val_selected,
                    y_val,
                    classification_data['encoded_classes'],
                    output_dir=output_dir
                )
                
                # Save best model
                save_classification_model(
                    classifiers,
                    evaluation['best_model'],
                    feature_selection['selector'],
                    classification_data['feature_names'],
                    feature_selection['selected_features'],
                    classification_data['encoded_classes'],
                    os.path.join(output_dir, 'sleep_classifier.pkl')
                )
                
                results['classification'] = {
                    'evaluation': evaluation,
                    'best_model': evaluation['best_model'],
                    'accuracy': evaluation['best_accuracy']
                }
    
    results['processed'] = True
    print("\nEEG processing pipeline completed successfully!")
    return results


# Example usage of the library
if __name__ == "__main__":
    print("EEG Processing Library - Example Usage")
    print("-------------------------------------")
    print("\nThis script provides a comprehensive set of functions for EEG data processing.")
    print("Below is a simple example of how to use the library:")
    
    print("""
    # Example code:
    
    import eeg_processing as eeg
    
    # 1. Process EEG data with complete pipeline
    results = eeg.complete_eeg_pipeline(
        eeg_file='path/to/eeg_data.mat',
        hypnogram_file='path/to/hypnogram.mat',
        output_dir='path/to/output'
    )
    
    # 2. Or process data step by step:
    
    # Load data
    data = eeg.load_eeg_data('path/to/eeg_data.mat')
    fs = 125  # Hz
    
    # Filter
    filtered_eeg = eeg.bandpass_filter(data['eeg'], fs)
    
    # Apply ICA
    ica_result = eeg.apply_ica(filtered_eeg, {'eog': data['eog'], 'emg': data['emg']})
    correlation_result = eeg.calculate_artifact_correlations(ica_result, {'eog': data['eog']})
    artifact_components = eeg.identify_artifact_components(correlation_result)
    cleaned_eeg = eeg.reconstruct_cleaned_signal(ica_result, artifact_components)
    
    # Extract frequency bands
    bands = eeg.extract_frequency_bands_dwt(cleaned_eeg[0, :], fs)
    
    # Detect spindles
    spindles, spindle_data = eeg.detect_sleep_spindles(bands['alpha'], fs)
    
    # Extract features
    features = eeg.extract_eeg_features(cleaned_eeg[0, :], bands, spindles, fs)
    """)
    
    print("\nTo run the complete pipeline, call this script with appropriate arguments:")
    print("python eeg_processing.py path/to/eeg_data.mat path/to/hypnogram.mat path/to/output_dir")
    
    # Check if command line arguments are provided
    import sys
    if len(sys.argv) > 1:
        eeg_file = sys.argv[1]
        hypnogram_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Run the pipeline with provided arguments
        complete_eeg_pipeline(eeg_file, hypnogram_file, output_dir)