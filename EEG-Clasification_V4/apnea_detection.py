import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy import signal
import pywt
import pandas as pd
import mne
import neurokit2 as nk



def main():
    # Load ECG and ABDORES data from EDF file
    base_input_dir = os.path.join(os.getcwd(), 'Input')
    k = 1
    edf_path = os.path.join(base_input_dir, f"R{k}.edf")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    print(raw.ch_names)
    
    # Save ECG and ABDORES data to variables
    ecg_data = raw.get_data(picks='ECG')
    abdo_res_data = raw.get_data(picks='ABDO RES')
    
    signals, info = nk.ecg_process(ecg_data[0], sampling_rate=125)
    print(signals)

if __name__ == "__main__":
    main()