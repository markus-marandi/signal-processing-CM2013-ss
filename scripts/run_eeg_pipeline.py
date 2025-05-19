import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
# import pickle # Unused
import preprocess_library as pre
import pywt # Consolidated import
import pandas as pd
from scipy import signal, stats, integrate
from scipy.integrate import simpson
import mne
# from mne.datasets.sleep_physionet.age import fetch_data # Unused
import nolds # Optional: for fractal dimension, Lyapunov exponent etc. pip install nolds
# from pywt import wavedec # Already imported via pywt
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
from sklearn.utils import class_weight

# --- Global Configuration ---
OUTPUT_DIR = "Output_Figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helper for Spectral Power ---
def get_band_power(epoch_data, fs, band, window_sec=None, relative=False):
    from scipy.signal import welch # Keep local import for clarity if preferred
    # from scipy.integrate import simpson # Already globally imported

    band = np.asarray(band)
    low, high = band
    if window_sec is not None:
        nperseg = int(window_sec * fs)
    else:
        nperseg = len(epoch_data)
    if nperseg == 0: return 0

    freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg)
    if len(freqs) == 0: return 0

    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    band_power = simpson(psd[idx_band], dx=freq_res)

    if relative:
        total_power = simpson(psd, dx=freq_res)
        return (band_power / total_power) * 100 if total_power > 1e-9 else 0
    else:
        return band_power

def extract_eeg_features(epoch_data, fs):
    """
    Extracts a comprehensive set of features from a single EEG epoch.
    """
    features = {}
    epoch_data = np.asarray(epoch_data).squeeze()

    # Time-Domain
    features['mean'] = np.mean(epoch_data)
    features['median'] = np.median(epoch_data)
    features['std'] = np.std(epoch_data)
    features['variance'] = np.var(epoch_data)
    features['kurtosis'] = stats.kurtosis(epoch_data)
    features['skewness'] = stats.skew(epoch_data)
    features['rms'] = np.sqrt(np.mean(epoch_data**2))
    features['peak_to_peak'] = np.ptp(epoch_data)
    features['zero_crossings'] = len(np.where(np.diff(np.sign(epoch_data)))[0])

    # Hjorth Parameters
    diff_epoch = np.diff(epoch_data)
    diff2_epoch = np.diff(diff_epoch)
    var_epoch = np.var(epoch_data)
    var_diff_epoch = np.var(diff_epoch) if len(diff_epoch) > 0 else 0
    var_diff2_epoch = np.var(diff2_epoch) if len(diff2_epoch) > 0 else 0

    features['hjorth_activity'] = var_epoch
    features['hjorth_mobility'] = np.sqrt(var_diff_epoch / var_epoch) if var_epoch > 1e-9 else 0
    if features['hjorth_mobility'] > 1e-9 and var_diff_epoch > 1e-9:
        mobility_diff = np.sqrt(var_diff2_epoch / var_diff_epoch)
        features['hjorth_complexity'] = mobility_diff / features['hjorth_mobility']
    else:
        features['hjorth_complexity'] = 0

    # Spectral Features
    bands = {
        'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
        'sigma': (11.0, 16.0), 'beta':  (13.0, 30.0), 'gamma': (30.0, min(60.0, fs/2 - 0.1))
    }
    # Ensure nperseg is not greater than epoch_data length
    nperseg_welch = min(len(epoch_data), 256 if len(epoch_data) >= 256 else len(epoch_data))
    if nperseg_welch == 0: # handle empty epoch_data
        freqs_welch, psd_welch = np.array([]), np.array([])
    else:
        freqs_welch, psd_welch = signal.welch(epoch_data, fs, nperseg=nperseg_welch)

    total_power = integrate.simpson(psd_welch, dx=(freqs_welch[1]-freqs_welch[0])) if len(freqs_welch)>1 else 0

    for band_name, band_freqs in bands.items():
        abs_p = get_band_power(epoch_data, fs, band_freqs, relative=False, window_sec=nperseg_welch/fs if nperseg_welch > 0 else None)
        features[f'abs_power_{band_name}'] = abs_p
        features[f'rel_power_{band_name}'] = (abs_p / total_power) * 100 if total_power > 1e-9 else 0

    if total_power > 1e-9 and len(freqs_welch) > 1:
        cumulative_power = np.cumsum(psd_welch) * (freqs_welch[1]-freqs_welch[0])
        try:
            features['sef50'] = freqs_welch[np.where(cumulative_power >= 0.50 * total_power)[0][0]]
            features['sef95'] = freqs_welch[np.where(cumulative_power >= 0.95 * total_power)[0][0]]
        except IndexError:
            features['sef50'], features['sef95'] = 0, 0
        features['spectral_centroid'] = np.sum(freqs_welch * psd_welch) / np.sum(psd_welch) if np.sum(psd_welch) > 1e-9 else 0
        norm_psd = psd_welch / (np.sum(psd_welch) + 1e-12)
        features['spectral_entropy'] = -np.sum(norm_psd * np.log2(norm_psd + 1e-12))
    else:
        features['sef50'], features['sef95'], features['spectral_centroid'], features['spectral_entropy'] = 0,0,0,0

    # Nonlinear Features (nolds)
    try:
        if len(epoch_data) > 20:
            features['hurst_exponent'] = nolds.hurst_rs(epoch_data)
            features['dfa'] = nolds.dfa(epoch_data)
            features['sample_entropy'] = nolds.sampen(epoch_data, emb_dim=2, tolerance=0.2 * np.std(epoch_data))
        else:
            raise ValueError("Epoch too short for nolds")
    except Exception:
        features['hurst_exponent'], features['dfa'], features['sample_entropy'] = np.nan, np.nan, np.nan

    # Wavelet-based Features
    num_wavelet_levels_expected = 5 # cA4, cD4, cD3, cD2, cD1 (approx)
    try:
        max_level = pywt.dwt_max_level(len(epoch_data), pywt.Wavelet('db4').dec_len)
        level_to_use = min(4, max_level)
        if level_to_use > 0:
            coeffs = pywt.wavedec(epoch_data, 'db4', level=level_to_use)
            for i, coeff_arr in enumerate(coeffs):
                features[f'wavelet_energy_level_{i}'] = np.sum(coeff_arr**2)
                features[f'wavelet_std_level_{i}'] = np.std(coeff_arr)
                features[f'wavelet_mean_abs_level_{i}'] = np.mean(np.abs(coeff_arr))
            # Fill remaining expected wavelet features with NaN if decomposition level was less than 4
            for i in range(len(coeffs), num_wavelet_levels_expected):
                features[f'wavelet_energy_level_{i}'] = np.nan
                features[f'wavelet_std_level_{i}'] = np.nan
                features[f'wavelet_mean_abs_level_{i}'] = np.nan
        else: # No decomposition possible
            for i in range(num_wavelet_levels_expected):
                features[f'wavelet_energy_level_{i}'] = np.nan
                features[f'wavelet_std_level_{i}'] = np.nan
                features[f'wavelet_mean_abs_level_{i}'] = np.nan
    except Exception:
        for i in range(num_wavelet_levels_expected):
            features[f'wavelet_energy_level_{i}'] = np.nan
            features[f'wavelet_std_level_{i}'] = np.nan
            features[f'wavelet_mean_abs_level_{i}'] = np.nan
    return features


def create_feature_dataframe(patient_epoch_dict, eeg_channel_name, sampling_freq):
    """
    Creates a DataFrame with extracted features and hypnogram.
    """
    all_epochs_features = []
    eeg_epochs_list = patient_epoch_dict[eeg_channel_name]
    hypnogram_list = patient_epoch_dict['hypnogram']

    min_len = min(len(eeg_epochs_list), len(hypnogram_list))
    eeg_epochs_list = eeg_epochs_list[:min_len]
    hypnogram_list = hypnogram_list[:min_len]
    
    print(f"Processing {min_len} epochs for channel {eeg_channel_name}...")

    # Determine feature keys from a dummy or first successful extraction
    feature_keys = None
    try:
        # Try with actual data first if available and long enough
        if min_len > 0 and eeg_epochs_list[0] is not None and len(eeg_epochs_list[0]) > 0:
             feature_keys = extract_eeg_features(eeg_epochs_list[0], sampling_freq).keys()
        else: # Fallback to dummy data if first epoch is problematic
            dummy_data = np.random.rand(sampling_freq * 30) # Default 30s epoch
            feature_keys = extract_eeg_features(dummy_data, sampling_freq).keys()
    except Exception as e:
        print(f"Critical error: Could not determine feature keys for NaN padding due to: {e}. Aborting feature extraction.")
        return pd.DataFrame()
    
    nan_features_template = {key: np.nan for key in feature_keys}

    for i, eeg_epoch_data in enumerate(eeg_epochs_list):
        if (i + 1) % 100 == 0 or i == min_len -1:
            print(f"  Extracting features for epoch {i+1}/{min_len}")
        try:
            # Ensure epoch_data is not empty or too short for basic operations
            if eeg_epoch_data is None or len(eeg_epoch_data) < sampling_freq * 1: # Min 1s for some features
                raise ValueError("Epoch data is None or too short.")
            epoch_features = extract_eeg_features(eeg_epoch_data, sampling_freq)
            all_epochs_features.append(epoch_features)
        except Exception as e:
            print(f"    Error in epoch {i+1} for channel {eeg_channel_name}: {e}. Appending NaNs.")
            all_epochs_features.append(nan_features_template.copy())


    if not all_epochs_features:
        print("No features extracted. Returning empty DataFrame.")
        return pd.DataFrame()

    df_features_only = pd.DataFrame(all_epochs_features)
    final_hypnogram_list = hypnogram_list[:len(df_features_only)]

    df_sleep_features = pd.concat([df_features_only, pd.Series(final_hypnogram_list, name='Hypnogram_Stage', index=df_features_only.index)], axis=1)
    df_sleep_features.insert(0, 'Epoch_Number', np.arange(len(df_sleep_features)))
    
    print("\n--- Sleep DataFrame with Extracted EEG Features ---")
    print(f"DataFrame Shape: {df_sleep_features.shape}")
    
    if df_sleep_features.isnull().any().any():
        print("Note: NaN values found in features (some may be expected from specific feature extractors).")
    
    return df_sleep_features

def prepare_data_for_models(df_features, target_column='Hypnogram_Stage', test_size=0.2, random_state=42, scale_features=True, for_cnn_sequence=False, raw_eeg_epochs=None):
    """Prepares data for XGBoost and Keras DL models."""
    df_clean = df_features.dropna()
    if len(df_clean) < len(df_features) and raw_eeg_epochs is not None:
        # Align raw_eeg_epochs with df_clean if NaNs were dropped
        raw_eeg_epochs = [epoch for i, epoch in enumerate(raw_eeg_epochs) if df_features.index[i] in df_clean.index]

    y = df_clean[target_column]
    X = df_clean.drop(columns=[target_column, 'Epoch_Number'], errors='ignore')

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Classes found: {label_encoder.classes_}, Num classes: {num_classes}")

    if for_cnn_sequence:
        if raw_eeg_epochs is None:
            raise ValueError("`raw_eeg_epochs` must be provided when `for_cnn_sequence` is True.")
        
        X_seq = np.array(raw_eeg_epochs)
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, axis=-1)
        
        X_train_seq, X_test_seq, y_train_enc, y_test_enc = train_test_split(
            X_seq, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
        
        print(f"X_train_seq shape: {X_train_seq.shape}, y_train_cat shape: {y_train_cat.shape}")
        return X_train_seq, X_test_seq, y_train_cat, y_test_cat, label_encoder, num_classes
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print("Features scaled using StandardScaler.")
        
        print(f"X_train (features) shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test, label_encoder, num_classes

def evaluate_model(y_true, y_pred_encoded, label_encoder, model_name="Model", output_dir=OUTPUT_DIR):
    """Prints classification report and saves confusion matrix."""
    print(f"\n--- Evaluation Report for {model_name} ---")
    class_names = label_encoder.classes_.astype(str)
    report = classification_report(y_true, y_pred_encoded, target_names=class_names, zero_division=0)
    print(report)
    
    accuracy = accuracy_score(y_true, y_pred_encoded)
    kappa = cohen_kappa_score(y_true, y_pred_encoded)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    cm = confusion_matrix(y_true, y_pred_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    plt.close() # Close the figure to free memory
    return accuracy, kappa

def train_xgboost_classifier(X_train, y_train, X_test, y_test, label_encoder, num_classes):
    """Trains an XGBoost classifier."""
    print("\n--- Training XGBoost Classifier ---")
    
    model_xgb = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        use_label_encoder=False 
    )
    
    model_xgb.fit(np.ascontiguousarray(X_train), np.ascontiguousarray(y_train),
                  eval_set=[(np.ascontiguousarray(X_test), np.ascontiguousarray(y_test))],
                  verbose=False)
    
    y_pred_xgb = model_xgb.predict(np.ascontiguousarray(X_test))
    print("XGBoost training complete.")
    accuracy, kappa = evaluate_model(y_test, y_pred_xgb, label_encoder, "XGBoost")
    return model_xgb, accuracy, kappa

def train_1d_cnn_on_features(X_train, y_train, X_test, y_test, label_encoder, num_classes, input_dim):
    """Trains a 1D CNN on extracted features, with class weighting."""
    print("\n--- Training 1D CNN on Extracted Features (with Class Weights) ---")

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    unique_encoded_labels = np.unique(y_train) 
    class_weights_calculated = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_encoded_labels,
        y=y_train
    )
    class_weights_dict = {label: weight for label, weight in zip(unique_encoded_labels, class_weights_calculated)}
    print(f"Class weights for Keras: {class_weights_dict}")

    model_cnn = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1), padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-5, verbose=1)
    ]
    history = model_cnn.fit(X_train_cnn, y_train_cat, epochs=100, batch_size=64,
                            validation_data=(X_test_cnn, y_test_cat),
                            callbacks=callbacks, class_weight=class_weights_dict, verbose=1)

    loss, acc_metric = model_cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f"1D CNN (on features) Test Loss: {loss:.4f}, Test Accuracy: {acc_metric:.4f}")
    y_pred_proba_cnn = model_cnn.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)
    accuracy, kappa = evaluate_model(y_test, y_pred_cnn, label_encoder, "1D_CNN_Features_ClassWeights")
    return model_cnn, history, accuracy, kappa

def train_1d_cnn_on_sequences(X_train_seq, y_train_cat, X_test_seq, y_test_cat, label_encoder, num_classes, sequence_length, num_channels=1):
    """Trains a 1D CNN directly on EEG sequences."""
    print("\n--- Training 1D CNN on Raw EEG Sequences ---")

    model_cnn_seq = Sequential([
        Conv1D(filters=64, kernel_size=50, strides=6, activation='relu', 
               input_shape=(sequence_length, num_channels), padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=8, strides=2), Dropout(0.4),
        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=4, strides=2), Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model_cnn_seq.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn_seq.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    ]
    history = model_cnn_seq.fit(X_train_seq, y_train_cat, epochs=150, batch_size=128,
                                validation_data=(X_test_seq, y_test_cat),
                                callbacks=callbacks, verbose=1)
    
    loss, acc_metric = model_cnn_seq.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"1D CNN (on sequences) Test Loss: {loss:.4f}, Test Accuracy: {acc_metric:.4f}")
    y_pred_proba_cnn_seq = model_cnn_seq.predict(X_test_seq)
    y_pred_cnn_seq_encoded = np.argmax(y_pred_proba_cnn_seq, axis=1)
    y_test_original_labels = np.argmax(y_test_cat, axis=1)
    accuracy, kappa = evaluate_model(y_test_original_labels, y_pred_cnn_seq_encoded, label_encoder, "1D_CNN_Sequences")
    return model_cnn_seq, history, accuracy, kappa

if __name__ == "__main__":
    print("Running main_V4.py")

    CHANNELS = ['EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG']
    FS = 125
    EPOCH_SEC_LENGTH = 30
    SAMPLES_PER_EPOCH = FS * EPOCH_SEC_LENGTH

    patient_data_list, patient_xml_list = pre.load_patient_data(1, 'Input', CHANNELS, 'Output')
    patient_data = patient_data_list[0]
    patient_stage = patient_xml_list[0]['stages']

    print(f"Initial data length: {len(patient_data['EEG'])} samples, Stage length: {len(patient_stage)} epochs")

    # Align data length to stage length (assuming stages are 1 per 30s epoch)
    # This step was removing last 30*FS, but if stages define total duration, it's better to trim patient_data
    # to match number of stages * SAMPLES_PER_EPOCH
    num_epochs_from_stages = len(patient_stage)
    expected_data_samples = num_epochs_from_stages * SAMPLES_PER_EPOCH 
    
    for channel in patient_data:
        # Trim based on hypnogram if data is longer
        if len(patient_data[channel]) > expected_data_samples:
             patient_data[channel] = patient_data[channel][:expected_data_samples]
        # Or, if data is shorter (less common if stages from same file), this setup might need adjustment
        # For now, assume data is at least as long as indicated by stages
        
    print(f"Aligned data length: {len(patient_data['EEG'])} samples for {num_epochs_from_stages} stages")

    # Removing start and finish (30 minutes)
    SECONDS_TO_REMOVE = 30 * 60 
    SAMPLES_TO_REMOVE_FROM_DATA = SECONDS_TO_REMOVE * FS
    EPOCHS_TO_REMOVE_FROM_STAGES = SECONDS_TO_REMOVE // EPOCH_SEC_LENGTH # Integer division

    for channel in patient_data:
        patient_data[channel] = patient_data[channel][SAMPLES_TO_REMOVE_FROM_DATA : -SAMPLES_TO_REMOVE_FROM_DATA if SAMPLES_TO_REMOVE_FROM_DATA > 0 else None]
    patient_stage = patient_stage[EPOCHS_TO_REMOVE_FROM_STAGES : -EPOCHS_TO_REMOVE_FROM_STAGES if EPOCHS_TO_REMOVE_FROM_STAGES > 0 else None]

    print(f"Data length after trimming ends: {len(patient_data['EEG'])} samples, Stage length: {len(patient_stage)} epochs")

    # Simple Filters
    filtered_data = {}
    for channel in patient_data:
        temp_filtered = mne.filter.filter_data(patient_data[channel].astype(np.float64), FS, 0.5, 62.0, fir_design='firwin', verbose=False)
        filtered_data[channel] = mne.filter.notch_filter(temp_filtered, FS, 60, method='fir', fir_design='firwin', verbose=False)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1); plt.plot(patient_data['EEG'][:FS*10], label='Raw EEG'); plt.legend() # Plot 10s
    plt.subplot(2, 1, 2); plt.plot(filtered_data['EEG'][:FS*10], label='Filtered EEG'); plt.legend() # Plot 10s
    plt.suptitle("Raw vs. Bandpass+Notch Filtered EEG")
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_vs_filtered_eeg.png")); plt.close()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1); plt.psd(patient_data['EEG'], NFFT=2**10, Fs=FS); plt.title("PSD Raw EEG")
    plt.subplot(2, 1, 2); plt.psd(filtered_data['EEG'], NFFT=2**10, Fs=FS); plt.title("PSD Filtered EEG")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "psd_raw_vs_filtered_eeg.png")); plt.close()

    # Adaptive Filtering
    import padasip as pa
    N_EOG_FILTER, MU_EOG = 25, 0.005
    N_ECG_FILTER, MU_ECG = 35, 0.001
    EEG_CHANNEL_NAME = 'EEG' # Main EEG channel for processing
    EOG_L_NAME, EOG_R_NAME, ECG_NAME = 'EOG(L)', 'EOG(R)', 'ECG'

    adaptively_filtered_data_out = filtered_data.copy()
    eeg_signal_current = filtered_data[EEG_CHANNEL_NAME].copy()
    eog_l_ref = filtered_data[EOG_L_NAME]
    eog_r_ref = filtered_data[EOG_R_NAME]
    ecg_ref_signal = filtered_data[ECG_NAME]

    # EOG Artifact Removal
    print(f"Applying EOG filtering to {EEG_CHANNEL_NAME}...")
    eog_diff_ref = eog_l_ref - eog_r_ref
    if len(eog_diff_ref) >= N_EOG_FILTER and len(eeg_signal_current) >= N_EOG_FILTER :
        x_eog_history = pa.input_from_history(eog_diff_ref, N_EOG_FILTER)
        d_eeg_target_eog = eeg_signal_current[N_EOG_FILTER-1:]
        f_eog = pa.filters.AdaptiveFilter(model="NLMS", n=N_EOG_FILTER, mu=MU_EOG, w="random")
        y_pred_eog, e_cleaned_eog, _ = f_eog.run(d_eeg_target_eog, x_eog_history)
        eeg_signal_current[N_EOG_FILTER-1:] = e_cleaned_eog
    else:
        print(f"  Signal too short for EOG filter order {N_EOG_FILTER}. Skipping EOG filtering.")
        y_pred_eog = np.zeros_like(eeg_signal_current[N_EOG_FILTER-1:]) # For plotting

    # ECG Artifact Removal
    print(f"Applying ECG filtering to {EEG_CHANNEL_NAME} (post-EOG)...")
    if len(ecg_ref_signal) >= N_ECG_FILTER and len(eeg_signal_current) >= N_ECG_FILTER:
        x_ecg_history = pa.input_from_history(ecg_ref_signal, N_ECG_FILTER)
        d_eeg_target_ecg = eeg_signal_current[N_ECG_FILTER-1:]
        f_ecg = pa.filters.AdaptiveFilter(model="NLMS", n=N_ECG_FILTER, mu=MU_ECG, w="random")
        y_pred_ecg, e_cleaned_ecg, _ = f_ecg.run(d_eeg_target_ecg, x_ecg_history)
        eeg_signal_current[N_ECG_FILTER-1:] = e_cleaned_ecg
    else:
        print(f"  Signal too short for ECG filter order {N_ECG_FILTER}. Skipping ECG filtering.")
    
    adaptively_filtered_data_out[EEG_CHANNEL_NAME] = eeg_signal_current

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    plot_len = min(len(filtered_data[EEG_CHANNEL_NAME]), 10 * FS)
    time_axis_plot = np.arange(plot_len) / FS
    axs[0].plot(time_axis_plot, filtered_data[EEG_CHANNEL_NAME][:plot_len], label='Original Filtered EEG')
    axs[0].set_title('Original Filtered EEG'); axs[0].legend()
    axs[1].plot(time_axis_plot, adaptively_filtered_data_out[EEG_CHANNEL_NAME][:plot_len], label='EEG after EOG & ECG cleaning')
    axs[1].set_title('EEG after Adaptive Cleaning'); axs[1].legend()
    
    # Plot EOG artifact component (reconstructed for this plot segment)
    padded_eog_artifact = np.zeros(plot_len)
    if len(y_pred_eog) > 0: # y_pred_eog starts from N_EOG_FILTER-1
        len_to_plot = min(plot_len - (N_EOG_FILTER-1) if plot_len > (N_EOG_FILTER-1) else 0, len(y_pred_eog))
        if len_to_plot > 0:
             padded_eog_artifact[N_EOG_FILTER-1 : N_EOG_FILTER-1 + len_to_plot] = y_pred_eog[:len_to_plot]
    axs[2].plot(time_axis_plot, padded_eog_artifact, label='Predicted EOG Artifact', color='green')
    axs[2].set_title('Predicted EOG Artifact Component'); axs[2].legend()
    plt.xlabel("Time (s)"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_filtering_eeg.png")); plt.close()

    # Wavelet Denoising (on the output of simple bandpass+notch, not adaptive for this example flow)
    # If you want to apply to adaptively filtered, change signal_to_process to adaptively_filtered_data_out['EEG']
    signal_to_process_wavelet = filtered_data['EEG'].copy() 
    wavelet, level = 'db4', 4
    
    coeffs = pywt.wavedec(signal_to_process_wavelet, wavelet, level=level)
    thresholded_coeffs = [coeffs[0]] 
    N_sig = len(signal_to_process_wavelet)

    for i in range(1, len(coeffs)):
        detail_coeff_set = coeffs[i]
        median_val = np.median(detail_coeff_set)
        mad = np.median(np.abs(detail_coeff_set - median_val))
        sigma = mad / 0.6745 if mad > 1e-9 else 0 # Avoid division by zero if MAD is zero
        threshold_value = sigma * np.sqrt(2 * np.log(N_sig)) if N_sig > 1 and sigma > 0 else 0
        thresholded_coeffs.append(pywt.threshold(detail_coeff_set, value=threshold_value, mode='soft'))

    eeg_wavelet_denoised = pywt.waverec(thresholded_coeffs, wavelet)
    eeg_wavelet_denoised = eeg_wavelet_denoised[:N_sig] # Ensure same length

    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1); plt.plot(signal_to_process_wavelet[:FS*10]); plt.title("Signal before Wavelet Denoising (10s)")
    plt.subplot(2,1,2); plt.plot(eeg_wavelet_denoised[:FS*10]); plt.title("Signal after Wavelet Denoising (10s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wavelet_denoising_eeg.png")); plt.close()
    
    # For feature extraction, use the output of the desired preprocessing stage.
    # Here, let's use `adaptively_filtered_data_out` for EEG channel and other channels from `filtered_data`.
    # This means only EEG channel underwent adaptive filtering. Other channels used are from simple filtering.
    # If wavelet denoising is preferred for EEG, then replace 'adaptively_filtered_data_out' with 'eeg_wavelet_denoised'.
    # For simplicity, we'll put the adaptively filtered EEG back into the main dictionary for epoching.
    processed_data_for_epoching = filtered_data.copy() # Start with simple filtered
    processed_data_for_epoching[EEG_CHANNEL_NAME] = adaptively_filtered_data_out[EEG_CHANNEL_NAME] # Replace EEG with adaptively filtered one
    # Or, if you chose wavelet:
    # processed_data_for_epoching[EEG_CHANNEL_NAME] = eeg_wavelet_denoised


    # Create epochs
    NB_EPOCHS = len(processed_data_for_epoching[EEG_CHANNEL_NAME]) // SAMPLES_PER_EPOCH
    # Ensure patient_stage length matches NB_EPOCHS from data
    patient_stage = patient_stage[:NB_EPOCHS]

    patient_epoch = {}
    for channel_name in CHANNELS:
        channel_data = processed_data_for_epoching[channel_name]
        patient_epoch[channel_name] = [channel_data[i*SAMPLES_PER_EPOCH:(i+1)*SAMPLES_PER_EPOCH] for i in range(NB_EPOCHS)]
        print(f"  - Channel {channel_name}: Created {len(patient_epoch[channel_name])} epochs.")

    patient_epoch['hypnogram'] = []
    for epoch_idx in range(NB_EPOCHS): # Use NB_EPOCHS derived from data length
        # Stage data is already 1 sample per epoch after initial processing
        # Dominant stage logic was applied during load_patient_data or an equivalent if it's raw per-second stages
        # Assuming patient_stage is now correctly 1 value per 30s epoch
        # If patient_stage was still 1Hz, then the stats.mode logic would apply:
        start_idx = epoch_idx * EPOCH_SEC_LENGTH
        end_idx = start_idx + EPOCH_SEC_LENGTH
        epoch_stage_segment = patient_stage[start_idx:end_idx]
        dominant_stage = stats.mode(epoch_stage_segment)[0]
        patient_epoch['hypnogram'].append(dominant_stage)
        # But since patient_stage is already epoched:
        # patient_epoch['hypnogram'].append(patient_stage[epoch_idx])


    # Feature Extraction
    EEG_CHANNEL_FOR_FEATURES = 'EEG'
    FEATURE_FILENAME = "sleep_features_patient1.csv" # More specific filename

    if os.path.exists(FEATURE_FILENAME):
        print(f"Loading features from '{FEATURE_FILENAME}'.")
        df_final_features = pd.read_csv(FEATURE_FILENAME)
    else:
        print(f"'{FEATURE_FILENAME}' not found. Extracting features...")
        df_final_features = create_feature_dataframe(patient_epoch, EEG_CHANNEL_FOR_FEATURES, FS)
        if not df_final_features.empty:
            df_final_features.to_csv(FEATURE_FILENAME, index=False)
            print(f"Features saved to '{FEATURE_FILENAME}'.")
        else:
            print("Feature extraction resulted in an empty DataFrame. Cannot proceed with saving or training.")

    if 'df_final_features' in locals() and not df_final_features.empty:
        print("\n--- Proceeding with Model Training ---")
        
        # Prepare data for feature-based models
        X_train_feat, X_test_feat, y_train_enc, y_test_enc, le, n_classes = prepare_data_for_models(
            df_final_features, target_column='Hypnogram_Stage'
        )

        if X_train_feat.size > 0:
            # Train XGBoost
            _, acc_xgb, kappa_xgb = train_xgboost_classifier(
                X_train_feat, y_train_enc, X_test_feat, y_test_enc, le, n_classes
            )
            print(f"XGBoost Final Test Accuracy: {acc_xgb:.4f}, Kappa: {kappa_xgb:.4f}")

            # Train 1D CNN on Features
            num_features = X_train_feat.shape[1]
            _, _, acc_cnn_feat, kappa_cnn_feat = train_1d_cnn_on_features(
                X_train_feat, y_train_enc, X_test_feat, y_test_enc, le, n_classes, num_features
            )
            print(f"1D CNN (Features) Final Test Accuracy: {acc_cnn_feat:.4f}, Kappa: {kappa_cnn_feat:.4f}")
        else:
            print("Training data for feature-based models is empty. Skipping XGBoost and CNN on features.")

        # Prepare and Train 1D CNN on Raw Sequences
        # Ensure raw epochs for CNN align with df_final_features (after potential NaN drops)
        # The `patient_epoch` dictionary holds the raw epochs *before* feature extraction and potential NaN drops.
        # `prepare_data_for_models` (when for_cnn_sequence=True) handles aligning these with `df_final_features`.
        
        # Get the raw EEG epochs that were used to generate features initially
        # These are from `patient_epoch` using `EEG_CHANNEL_FOR_FEATURES`
        raw_eegs_for_cnn = patient_epoch[EEG_CHANNEL_FOR_FEATURES]
        
        # Ensure raw_eegs_for_cnn has the same number of epochs as rows in df_final_features *before* NaN drop,
        # if NaN drop occurs inside prepare_data_for_models.
        # df_final_features is already loaded/created, so its length matches the original number of processable epochs.
        # `prepare_data_for_models` will internally handle `dropna` and align `raw_eegs_for_cnn` if passed.
        
        print("\nPreparing data for 1D CNN on raw sequences...")
        X_train_s, X_test_s, y_train_s_cat, y_test_s_cat, le_s, n_classes_s = prepare_data_for_models(
            df_final_features.copy(), # Pass a copy to avoid modifying original df for labels
            target_column='Hypnogram_Stage',
            for_cnn_sequence=True,
            raw_eeg_epochs=raw_eegs_for_cnn # Pass the original list of raw epochs
        )
            
        if X_train_s.size > 0:
            sequence_len = X_train_s.shape[1]
            num_raw_channels = X_train_s.shape[2]
            _, _, acc_cnn_seq, kappa_cnn_seq = train_1d_cnn_on_sequences(
                X_train_s, y_train_s_cat, X_test_s, y_test_s_cat, le_s, n_classes_s, sequence_len, num_raw_channels
            )
            print(f"1D CNN (Sequences) Final Test Accuracy: {acc_cnn_seq:.4f}, Kappa: {kappa_cnn_seq:.4f}")
        else:
            print("Training data for sequence-based CNN is empty. Skipping.")
    else:
        print("df_final_features is not defined or is empty. Cannot proceed with model training.")

    print("\n--- End of Script ---")