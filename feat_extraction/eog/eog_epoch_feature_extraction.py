import numpy as np
from scipy.signal import find_peaks, welch


def extract_epoch_features(eog_epoch, fs, peakStdFactor=3, window_sec=5):
    """
    Computes EOG features for a single 30-second epoch.

    Parameters:
      eog_epoch     : 1D numpy array with the EOG signal for one epoch.
      fs            : Sampling frequency in Hz.
      peakStdFactor : Factor used to compute an adaptive threshold for blink detection.
      window_sec    : Duration in seconds for mini-windows used in movement density estimation.

    Returns:
      A dictionary with extracted features:
        - 'numBlinks': Number of detected blink peaks.
        - 'blinkRatePerMin': Blink rate (blinks per minute).
        - 'blinkIndices': List of sample indices for detected blinks.
        - 'movementDensityMean': Mean standard deviation over mini-windows (proxy for movement density).
        - 'slowPower': Integrated power in the slow band (0.5–3 Hz).
        - 'rapidPower': Integrated power in the rapid band (3–30 Hz).
        - 'powerRatio': Ratio of slowPower to rapidPower.
    """
    features = {}
    # --- Blink Detection ---
    abs_epoch = np.abs(eog_epoch)
    computed_threshold = peakStdFactor * np.std(abs_epoch)
    capped_threshold = 0.9 * np.max(abs_epoch)
    threshold = min(computed_threshold, capped_threshold)

    min_peak_distance = int(0.3 * fs)  # 0.3 second minimum distance between peaks
    blink_indices, _ = find_peaks(abs_epoch, height=threshold, distance=min_peak_distance)
    num_blinks = len(blink_indices)
    epoch_duration_min = len(eog_epoch) / fs / 60  # Convert duration to minutes
    blink_rate_per_min = num_blinks / epoch_duration_min if epoch_duration_min > 0 else 0

    features['numBlinks'] = num_blinks
    features['blinkRatePerMin'] = blink_rate_per_min
    features['blinkIndices'] = blink_indices.tolist()

    # --- Movement Density ---
    window_samples = int(window_sec * fs)
    num_windows = len(eog_epoch) // window_samples
    std_vals = [np.std(eog_epoch[w * window_samples:(w + 1) * window_samples])
                for w in range(num_windows)]
    movement_density_mean = np.mean(std_vals) if std_vals else np.nan
    features['movementDensityMean'] = movement_density_mean

    # --- Slow vs. Rapid Eye Movement Power ---
    # Use Welch's method with a window length of 2 seconds.
    window_length = int(2 * fs)
    f, Pxx = welch(eog_epoch, fs=fs, nperseg=min(window_length, len(eog_epoch)))
    slow_band = (0.5, 3)
    rapid_band = (3, 30)
    slow_idx = (f >= slow_band[0]) & (f < slow_band[1])
    rapid_idx = (f >= rapid_band[0]) & (f < rapid_band[1])
    slow_power = np.trapz(Pxx[slow_idx], f[slow_idx])
    rapid_power = np.trapz(Pxx[rapid_idx], f[rapid_idx])

    features['slowPower'] = slow_power
    features['rapidPower'] = rapid_power
    features['powerRatio'] = slow_power / rapid_power if rapid_power > 0 else np.inf

    return features


def segment_signal_into_epochs(signal, fs, epoch_length_sec=30):
    """
    Splits a continuous signal into epochs of length epoch_length_sec.

    Parameters:
       signal           : 1D numpy array.
       fs               : Sampling frequency in Hz.
       epoch_length_sec : Duration of each epoch in seconds.

    Returns:
       epochs : List of 1D numpy arrays, one per epoch.
    """
    samples_per_epoch = int(epoch_length_sec * fs)
    n_epochs = len(signal) // samples_per_epoch
    epochs = [signal[i * samples_per_epoch:(i + 1) * samples_per_epoch] for i in range(n_epochs)]
    return epochs