import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import pickle
import processing_lib_V2 as pre
import pywt
import pandas as pd
from scipy import signal, stats, integrate
from scipy.integrate import simpson
import mne
from mne.datasets.sleep_physionet.age import fetch_data
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

# Create output directory for figures
OUTPUT_FIGURES_DIR = "Output_Figures"
if not os.path.exists(OUTPUT_FIGURES_DIR):
    os.makedirs(OUTPUT_FIGURES_DIR)


if __name__ == "__main__":
    
    print("Running main_V4.py")

    CHANNELS = [
        'EEG(sec)',      
        'ECG',   
        'EMG', 
        'EOG(L)', 
        'EOG(R)', 
        'EEG'
    ]
    ########################################################
    # Importing the channels and stages from EDF and XML file of 1 patient
    FS = 125
    EPOCH_SEC_LENGTH = 30
    SAMPLES_PER_EPOCH = FS * EPOCH_SEC_LENGTH
    
    EEG_CHANNEL_FOR_FEATURES = 'EEG' # Or 'EEG(sec)'
    FEATURE_FILENAME = "sleep_features_patientX.csv"

    INFO = mne.create_info(CHANNELS, FS, ch_types=['eeg', 'ecg', 'emg', 'eog', 'eog', 'eeg'])
    print(INFO)
    
    # patient_data, patient_stage, TIME = pre.extract_data(10, CHANNELS, FS)
    patients_data, patients_xml = pre.load_patient_data(10, 'Input',CHANNELS, 'Output')
    
    for id in range(0, 9):
        print(f"\nProcessing patient {id}")

        patient_data = patients_data[id]
        patient_stage = patients_xml[id]['stages']

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

        ########################################################
        
        filtered_data = pre.filter_data(patient_data, FS, OUTPUT_FIGURES_DIR)
        filtered_data = pre.adaptive_filtering(filtered_data, FS, OUTPUT_FIGURES_DIR)
        ########################################################
        
        filtered_data = pre.wavelet_denoising(filtered_data, FS, OUTPUT_FIGURES_DIR, TIME)
        ########################################################
        
        # Create epochs of 30 seconds for the filtered data
        patient_epoch = pre.create_epochs(filtered_data, patient_stage, FS, SAMPLES_PER_EPOCH, CHANNELS)
        
        all_patient_epochs =[]
        all_patient_epochs.append(patient_epoch)
        ########################################################
    
        # Append features for each patient to the same file
        df_final_features = pre.create_feature_dataframe(
            patient_epoch, 
            EEG_CHANNEL_FOR_FEATURES, 
            FS, 
            FEATURE_FILENAME,
            append_mode=(id > 0)  # Append mode for all patients after the first one
        )
            
    # Train models on the combined dataset
    pre.train_models(df_final_features, all_patient_epochs, EEG_CHANNEL_FOR_FEATURES, FS)