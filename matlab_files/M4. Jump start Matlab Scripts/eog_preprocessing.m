% run_phase2_all_eog.m
% Comprehensive script to process all EDF/XML files and extract EOG features.
%
% Folder structure (your root):
%   /Users/markus/university/signal-processing-CM2013-ss/M4. Jump start Matlab Scripts
%       ├── Data         (contains R1.edf, R1.xml, ..., R10.edf, R10.xml)
%       ├── Scripts      (contains edfread.m, readXML.m, and other toolbox files)
%       ├── eog_preprocessing.m  (if using as separate file, not needed here)
%       └── (other files)
%
% This script:
%   - Sets the root folder and adds the Scripts folder to the path.
%   - Scans the Data folder for EDF files.
%   - For each EDF file, loads the EDF and matching XML files.
%   - Extracts the left EOG channel (assumed to be channel 6) at 50 Hz.
%   - Processes the EOG signal using Phase 2 integration (baseline correction,
%     noise filtering, and feature extraction) and computes the envelope using
%     a custom Hilbert transform implementation.
%   - Stores the processed signals and extracted features, and prints a summary.
%   - Optionally plots the processed signal and envelope for the first file.
%
% Author: [Your Name]
% Date: [Today's Date]

%% --- Initialization ---
clc;
close all;
clear all;

%% --- Setup Paths ---
% Set the root folder
root = '/Users/markus/university/signal-processing-CM2013-ss/M4. Jump start Matlab Scripts';
dataFolder = fullfile(root, 'Data');
scriptsFolder = fullfile(root, 'Scripts');

% Add the Scripts folder to the MATLAB path
addpath(scriptsFolder);

%% --- Get List of EDF Files ---
edfFiles = dir(fullfile(dataFolder, 'R*.edf'));
numFiles = numel(edfFiles);
if numFiles == 0
    error('No EDF files found in the Data folder.');
end

%% --- Preallocate Storage for Results ---
allProcessedEOG = cell(numFiles, 1);
allFeatures   = cell(numFiles, 1);
fileNames     = cell(numFiles, 1);

%% --- Loop over each EDF/XML file pair ---
for i = 1:numFiles
    % Get the current EDF file and its corresponding XML file
    edfFilename = fullfile(dataFolder, edfFiles(i).name);
    
    % Replace .edf with .xml for the annotation file
    [~, baseName, ~] = fileparts(edfFiles(i).name);
    xmlFilename = fullfile(dataFolder, [baseName, '.xml']);
    
    fprintf('Processing file: %s\n', edfFiles(i).name);
    
    % Load the EDF data using the provided edfread function
    try
        [hdr, record] = edfread(edfFilename);
    catch ME
        warning('Failed to read %s: %s', edfFilename, ME.message);
        continue;
    end
    
    % Load XML annotation data using readXML (if needed)
    try
        [events, stages, epochLength, annotation] = readXML(xmlFilename);
    catch ME
        warning('Failed to read %s: %s', xmlFilename, ME.message);
        events = []; stages = []; epochLength = []; annotation = [];
    end
    
    %% --- Select the EOG Channel ---
    % According to the montage, the left EOG is assumed to be channel 6.
    eogChannelIndex = 6;
    if size(record,1) < eogChannelIndex
        warning('File %s does not contain channel %d. Skipping...', edfFiles(i).name, eogChannelIndex);
        continue;
    end
    eogSignal = record(eogChannelIndex, :);
    
    % Sampling frequency for EOG channel as specified in the montage is 50 Hz.
    fs = 50;
    
    %% --- Process EOG using Phase 2 Integration ---
    [processedEOG, features] = phase2_eeg_eog_integration(eogSignal, fs);
    
    %% --- Store Results ---
    allProcessedEOG{i} = processedEOG;
    allFeatures{i} = features;
    fileNames{i} = baseName;
end

%% --- Display Summary of Extracted Features ---
for i = 1:numFiles
    if isempty(allFeatures{i})
        continue;
    end
    fprintf('\nFile: %s\n', fileNames{i});
    disp('Blink Rate (blinks per minute):');
    disp(allFeatures{i}.blinkRate);
    disp('Slow/Rapid Eye Movement Ratio:');
    disp(allFeatures{i}.slowRapidPatterns.ratio);
end

%% --- Optional: Plot Processed EOG for the First File ---
if ~isempty(allProcessedEOG{1})
    figure;
    subplot(2,1,1);
    t = (0:length(allProcessedEOG{1})-1)/fs;
    plot(t, allProcessedEOG{1});
    title(['Processed EOG Signal - ', fileNames{1}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    subplot(2,1,2);
    plot(t, allFeatures{1}.eyeMovement);
    title(['EOG Envelope (Eye Movement) - ', fileNames{1}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Function: phase2_eeg_eog_integration ------------------------------
function [processedEOG, features] = phase2_eeg_eog_integration(eogSignal, fs)
%PHASE2_EEG_EOG_INTEGRATION Integrates EEG+EOG processing for Phase 2.
%
%   [processedEOG, features] = phase2_eeg_eog_integration(eogSignal, fs)
%
%   INPUTS:
%       eogSignal - Raw EOG signal (vector) extracted with edfread.m
%       fs        - Sampling frequency (Hz)
%
%   OUTPUTS:
%       processedEOG - EOG signal after baseline correction and noise filtering
%       features     - Struct containing extracted EOG features:
%                        .eyeMovement           - Envelope representing sleep eye movements
%                        .blinks                - Indices of detected blinks
%                        .movementDensity       - Variance per 30-sec epoch (proxy for movement density)
%                        .blinkRate             - Blinks per minute
%                        .blinkCharacteristics  - Amplitude and duration of detected blinks
%                        .slowRapidPatterns     - Spectral ratio of slow (<3Hz) to rapid (3–30Hz) eye movements
%
%   This function applies:
%       1. Baseline correction using a high-pass Butterworth filter.
%       2. Noise filtering using a low-pass Butterworth filter.
%
%   Then, it extracts optional features:
%       a. Eye movement envelope via a custom Hilbert transform.
%       b. Blink detection using a threshold-based method.
%       c. Movement density computed as variance over 30-sec epochs.
%       d. Blink rate and blink characteristic estimation.
%       e. Slow/rapid eye movement spectral ratio.
%
%   Example:
%       [hdr, record] = edfread('R1.edf');
%       % Assume left EOG channel is record(1,:) and fs = 50 Hz
%       [procEOG, feat] = phase2_eeg_eog_integration(record(1,:), 50);
%

%% --- EOG Preprocessing ---
% Baseline Correction: Remove drift.
hpCutoff = 0.5; % High-pass cutoff frequency in Hz
[b_hp, a_hp] = butter(2, hpCutoff/(fs/2), 'high');
eog_baseline_corrected = filtfilt(b_hp, a_hp, eogSignal);

% Noise Filtering: Remove high-frequency noise.
lpCutoff = 30; % Low-pass cutoff frequency in Hz
[b_lp, a_lp] = butter(4, lpCutoff/(fs/2), 'low');
processedEOG = filtfilt(b_lp, a_lp, eog_baseline_corrected);

%% --- Optional EOG Feature Extraction ---
features = struct();

% 1. Eye Movement Detection: Compute the envelope using a custom Hilbert transform.
envEOG = abs(customHilbert(processedEOG));
features.eyeMovement = envEOG;

% 2. Blink Detection: Simple threshold-based detection.
blinkThreshold = mean(processedEOG) + 2*std(processedEOG);
[~, blinkLocs] = findpeaks(processedEOG, 'MinPeakHeight', blinkThreshold, 'MinPeakDistance', round(0.3*fs));
features.blinks = blinkLocs;

% 3. Movement Density: Compute variance in 30-second epochs.
epochLength_sec = 30;
winLength = round(epochLength_sec * fs);
numEpochs = floor(length(processedEOG) / winLength);
movementDensity = zeros(1, numEpochs);
for iEpoch = 1:numEpochs
    epochData = processedEOG((iEpoch-1)*winLength+1 : iEpoch*winLength);
    movementDensity(iEpoch) = var(epochData);
end
features.movementDensity = movementDensity;

% 4. Blink Rate and Blink Characteristics:
if ~isempty(blinkLocs)
    totalTimeMinutes = length(processedEOG) / fs / 60;
    features.blinkRate = length(blinkLocs) / totalTimeMinutes;
    features.blinkCharacteristics.amplitudes = processedEOG(blinkLocs);
    % Placeholder: Assume fixed blink duration (e.g., 0.2 sec per blink)
    features.blinkCharacteristics.durations = repmat(0.2, size(blinkLocs));
else
    features.blinkRate = 0;
    features.blinkCharacteristics.amplitudes = [];
    features.blinkCharacteristics.durations = [];
end

% 5. Slow/Rapid Eye Movement Patterns: Compute spectral ratio.
nfft = 2^nextpow2(length(processedEOG));
EOG_FFT = fft(processedEOG, nfft);
freqAxis = fs/2 * linspace(0, 1, nfft/2+1);
powerSpectrum = abs(EOG_FFT(1:nfft/2+1)).^2;
slowIdx = freqAxis < 3;
rapidIdx = (freqAxis >= 3) & (freqAxis < 30);
slowPower = sum(powerSpectrum(slowIdx));
rapidPower = sum(powerSpectrum(rapidIdx));
features.slowRapidPatterns.slowPower = slowPower;
features.slowRapidPatterns.rapidPower = rapidPower;
if rapidPower ~= 0
    features.slowRapidPatterns.ratio = slowPower / rapidPower;
else
    features.slowRapidPatterns.ratio = Inf;
end

%% --- Custom Hilbert Transform Implementation ---
function x_analytic = customHilbert(x)
    % customHilbert computes the analytic signal of x using the FFT method.
    n = length(x);
    X = fft(x);
    H = zeros(n,1);
    if rem(n,2)==0
        % Even-length signal
        H(1) = 1;
        H(2:n/2) = 2;
        H(n/2+1) = 1;
    else
        % Odd-length signal
        H(1) = 1;
        H(2:(n+1)/2) = 2;
    end
    x_analytic = ifft(X .* H);
end

%% --- Integration with Phase 1 (EEG Preprocessing) ---
% If you have already processed EEG signals in Phase 1, you may merge the EEG and EOG
% features here. For example:
%
%   [eegProcessed, eegFeatures] = phase1_eeg_preprocessing(eegSignal, fs_eeg);
%   combinedFeatures = merge_features(eegFeatures, features);
%
% This section is left for further development.

end