#!/usr/bin/env python3
"""
cross_signal_feature_extraction.py
──────────────────────────────────────────────────────────────────────────────
Extracts cross‑modal physiological features for EACH EDF stored inside a v7.3

Features per 30 s epoch
───────────────────────
  • corr_EEG_EOG      (EEG   × mean‑EOG  Pearson r)
  • corr_EEG_EMG      (EEG   × EMG       Pearson r)
  • corr_ECG_RESP     (ECG   × Resp      Pearson r, NaN if no Resp chan)
  • xcorrLag_EOG_LR   (lag @ max |xcorr| left↔right EOG, samples)
  • coh_EEG_EOG       (mean MSC 0.5‑40 Hz)
  • coh_EEG_EMG
  • coh_ECG_RESP

Outputs (created in ./outputs/)
──────────────────────────────
  cross_signal_features.pkl   # pandas DataFrame, cached
  cross_signal_features.csv   # tidy CSV for ML pipelines
"""

from __future__ import annotations
import os, pickle, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import h5py
from scipy.signal import coherence, correlate

# ────────────────────────────────────────────────────────────────────────────
#  CONF
# ────────────────────────────────────────────────────────────────────────────
MAT_FILE = "../data/EDF_RawData.mat"
EPOCH_LEN_SEC = 30
FS_DEFAULT = 125                            # fallback if no /fs attr
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

# hard‑coded channel INDICES inside each 14‑column record matrix
CH_IDXS = {
    "EEG"  : 7,   # main EEG (column 8)
    "EEG2" : 2,   # optional second EEG (column 3 – not used here)
    "ECG"  : 3,
    "EMG"  : 4,
    "EOG_L": 5,
    "EOG_R": 6,
    "RESP" : 1,   # assumption; if out‑of‑range → Resp features = NaN
}

# ────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────
def epoch_iter(sig: np.ndarray, fs: int, epoch_len: int):
    step = fs * epoch_len
    for s in range(0, len(sig) - step + 1, step):
        yield sig[s : s + step]

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def max_xcorr_lag(x: np.ndarray, y: np.ndarray) -> int:
    corr = correlate(y, x, mode="full")
    lags = np.arange(-len(x)+1, len(x))
    return int(lags[np.argmax(np.abs(corr))])

def mean_ms_coherence(x: np.ndarray, y: np.ndarray, fs: int, nper: int=0) -> float:
    if nper == 0: nper = fs*2
    f, Cxy = coherence(x, y, fs=fs, nperseg=nper)
    band = (f >= .5) & (f <= 40)
    return float(np.nanmean(Cxy[band])) if band.any() else np.nan

def process_single_record(rec: np.ndarray, fs: int) -> pd.DataFrame:
    """
    rec shape: (n_samples, 14).  Returns DataFrame rows (one per epoch).
    """
    # ensure samples first
    if rec.shape[0] < rec.shape[1]:
        rec = rec.T

    # verify indices present
    n_chan = rec.shape[1]
    for name, idx in CH_IDXS.items():
        if idx >= n_chan:
            warnings.warn(f"Channel {name} index {idx} missing → features NaN.")
    # convenience shortcuts (np.ndarray 1‑D views)
    get = lambda k: rec[:, CH_IDXS[k]] if CH_IDXS[k] < n_chan else np.full(len(rec), np.nan)

    eeg   = get("EEG")
    eog_l = get("EOG_L")
    eog_r = get("EOG_R")
    emg   = get("EMG")
    ecg   = get("ECG")
    resp  = get("RESP")

    rows, epoch_id = [], 0
    for e in zip(epoch_iter(eeg, fs, EPOCH_LEN_SEC),
                 epoch_iter(eog_l, fs, EPOCH_LEN_SEC),
                 epoch_iter(eog_r, fs, EPOCH_LEN_SEC),
                 epoch_iter(emg,   fs, EPOCH_LEN_SEC),
                 epoch_iter(ecg,   fs, EPOCH_LEN_SEC),
                 epoch_iter(resp,  fs, EPOCH_LEN_SEC)):

        eeg_e, eogL_e, eogR_e, emg_e, ecg_e, resp_e = e
        rows.append({
            "epoch"           : epoch_id,
            "corr_EEG_EOG"    : pearson_r(eeg_e, (eogL_e+eogR_e)/2),
            "corr_EEG_EMG"    : pearson_r(eeg_e, emg_e),
            "corr_ECG_RESP"   : pearson_r(ecg_e, resp_e),
            "xcorrLag_EOG_LR" : max_xcorr_lag(eogL_e, eogR_e),
            "coh_EEG_EOG"     : mean_ms_coherence(eeg_e, (eogL_e+eogR_e)/2, fs),
            "coh_EEG_EMG"     : mean_ms_coherence(eeg_e, emg_e, fs),
            "coh_ECG_RESP"    : mean_ms_coherence(ecg_e, resp_e, fs),
        })
        epoch_id += 1
    return pd.DataFrame(rows)

def extract_cross_signal_features(mat_file: str) -> pd.DataFrame:
    with h5py.File(mat_file, 'r') as f:
        allData = f['allData']
        record_ds = allData['record']   # (n_records, 1) refs
        n_records = record_ds.shape[0]
        fs = int(np.array(allData.get('fs', FS_DEFAULT)))  # some files store fs here
        out_frames = []

        for i in range(n_records):
            ref = record_ds[i, 0]
            rec = f[ref][()]            # raw HDF5 dataset
            df_rec = process_single_record(rec, fs)
            df_rec.insert(0, 'patient', i)   # tag patient/EDF index
            out_frames.append(df_rec)

    return pd.concat(out_frames, ignore_index=True)


def main():
    print("[i] Loading MAT file and extracting cross‑signal features …")
    feats = extract_cross_signal_features(MAT_FILE)

    pkl = OUTPUT_DIR / "cross_signal_features.pkl"
    csv = OUTPUT_DIR / "cross_signal_features.csv"
    feats.to_pickle(pkl)
    feats.to_csv(csv, index=False)

    print(f"[✓] Finished: {len(feats)} epoch‑rows across "
          f"{feats['patient'].nunique()} EDF(s).")
    print(f"    → {pkl}\n    → {csv}")

if __name__ == "__main__":
    main()