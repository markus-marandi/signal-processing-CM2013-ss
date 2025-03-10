% ---------------------------
% process_all_EOG.m
% Comprehensive EOG Processing for 10 EDF Files
% ---------------------------
clc;
close all;
clear all;

%% 1) Locate and Load All EDF and XML Files
% Adjust dataFolder if needed
dataFolder = fullfile(pwd, 'Data');
edfFiles = dir(fullfile(dataFolder, 'R*.edf'));

if isempty(edfFiles)
    error('No EDF files matching "R*.edf" found in folder: %s', dataFolder);
end

% Preallocate a structure array to store summary features from each file
allResults = struct('fileName', {}, 'EOGChannel', {}, ...
    'numBlinks', {}, 'blinkRatePerMin', {}, ...
    'movementDensityMean', {}, 'slowPower', {}, ...
    'rapidPower', {}, 'powerRatio', {});

% Set processing parameters (adjust as needed)
blinkThreshold   = 100;  % amplitude threshold for blink detection
minPeakDistSec   = 0.3;  % minimum peak distance (s)
hpCutoff         = 0.5;  % high-pass cutoff (Hz) for baseline correction
lpCutoff         = 30;   % low-pass cutoff (Hz) for noise filtering

%% 2) Loop Over Each EDF File
for fIdx = 1:length(edfFiles)
    edfName = edfFiles(fIdx).name;
    edfPath = fullfile(dataFolder, edfName);
    
    % Construct matching XML file name (assuming same base name)
    [~, baseName] = fileparts(edfName);
    xmlPath = fullfile(dataFolder, [baseName, '.xml']);
    
    fprintf('\n=== Processing file: %s ===\n', edfName);
    
    % Load EDF file and (optionally) XML file
    [hdr, record] = edfread(edfPath);
    if exist(xmlPath, 'file')
        [events, stages, epochLength, annotation] = readXML(xmlPath);
    else
        events = []; stages = []; epochLength = 30; annotation = [];
        fprintf('XML file not found for %s. Proceeding without annotations.\n', edfName);
    end
    
    % Display number of 30-sec epochs (using channel 3 as an example)
    numberOfEpochs = length(record(3,:)) / (30 * hdr.samples(3));
    fprintf('Number of 30-sec epochs (based on channel #3): %.0f\n', numberOfEpochs);
    
    %% 3) Identify EOG Channels in This File
    eogChannelIndices = [];
    for chIdx = 1:length(hdr.label)
        if contains(upper(hdr.label{chIdx}), 'EOG')
            eogChannelIndices(end+1) = chIdx; %#ok<AGROW>
        end
    end
    
    if isempty(eogChannelIndices)
        warning('No EOG channels found in file %s. Skipping EOG processing.', edfName);
        continue;
    end
    
    fprintf('EOG channels found in %s: %s\n', edfName, mat2str(eogChannelIndices));
    
    %% 4) Process Each EOG Channel in This File
    for idx = 1:length(eogChannelIndices)
        channelIdx = eogChannelIndices(idx);
        fsEOG = hdr.samples(channelIdx);
        rawEOG = record(channelIdx, :);
        
        % Preprocess: Baseline correction and noise filtering using Butterworth filters
        preprocessedEOG = eog_preprocessing_butter(rawEOG, fsEOG, hpCutoff, lpCutoff);
        
        % Extract EOG features using pwelch for PSD estimation and findpeaks for blink detection
        featuresEOG = eog_feature_extraction(preprocessedEOG, fsEOG, blinkThreshold, minPeakDistSec);
        
        % Store results in a structure (one row per channel per file)
        resultStruct.fileName            = edfName;
        resultStruct.EOGChannel          = hdr.label{channelIdx};
        resultStruct.numBlinks           = featuresEOG.numBlinks;
        resultStruct.blinkRatePerMin     = featuresEOG.blinkRatePerMin;
        resultStruct.movementDensityMean = featuresEOG.movementDensityMean;
        resultStruct.slowPower           = featuresEOG.slowPower;
        resultStruct.rapidPower          = featuresEOG.rapidPower;
        resultStruct.powerRatio          = featuresEOG.powerRatio;
        
        allResults(end+1) = resultStruct; %#ok<AGROW>
        
        % Optional: Plot a 30-sec segment for visual inspection
        doPlotSegment = false;  % Set true to visualize each channel
        if doPlotSegment
            segDurationSec = 30;
            samplesToPlot  = min(segDurationSec * fsEOG, length(rawEOG));
            tAxis = (0:samplesToPlot-1) / fsEOG;
            figure('Name', ['EOG Compare: ' hdr.label{channelIdx} ' - ' edfName], 'NumberTitle', 'off');
            subplot(2,1,1);
            plot(tAxis, rawEOG(1:samplesToPlot));
            title(['Raw EOG (first ' num2str(segDurationSec) ' s) - ' hdr.label{channelIdx}]);
            ylabel('Amplitude'); xlabel('Time (s)');
            subplot(2,1,2);
            plot(tAxis, preprocessedEOG(1:samplesToPlot));
            hold on;
            blinkIdx = featuresEOG.blinkIndices;
            blinkIdxSegment = blinkIdx(blinkIdx <= samplesToPlot);
            plot((blinkIdxSegment-1)/fsEOG, preprocessedEOG(blinkIdxSegment), 'ro');
            title('Preprocessed EOG (with Blink Markers)');
            ylabel('Amplitude'); xlabel('Time (s)');
            set(gcf, 'color', 'w');
        end
    end
end

%% 5) Summarize and Save the Results
if isempty(allResults)
    warning('No results to display.');
else
    % Convert structure array to table for easy comparison and analysis
    T = struct2table(allResults);
    disp('=== Summary of EOG Feature Extraction Across All Files ===');
    disp(T);
    
    % Optionally, save results to a MAT file or CSV
    % save('EOG_Results.mat', 'allResults');
    % writetable(T, 'EOG_Results.csv');
end

%% -------------------------
%  END OF SCRIPT
% -------------------------


%% ----------------------------------------------------------------------
%  FUNCTIONS
% ----------------------------------------------------------------------

function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
%EOG_PREPROCESSING_BUTTER Preprocess EOG signal using Butterworth filters.
%   1) High-pass filter at hpCutoff (e.g., 0.5 Hz) to remove baseline drift.
%   2) Low-pass filter at lpCutoff (e.g., 30 Hz) to remove high-frequency noise.
%   Cutoffs are clamped to be within (0, fs/2).

nyquistFreq = fs / 2;
if hpCutoff < 0
    hpCutoff = 0.01;
elseif hpCutoff >= nyquistFreq
    hpCutoff = nyquistFreq - 0.01;
end
if lpCutoff >= nyquistFreq
    lpCutoff = nyquistFreq - 0.01;
elseif lpCutoff <= 0
    lpCutoff = 0.5;
end

hpNorm = hpCutoff / nyquistFreq;
lpNorm = lpCutoff / nyquistFreq;

[b_hp, a_hp] = butter(2, hpNorm, 'high');
signalHP = filtfilt(b_hp, a_hp, eogSignal);

[b_lp, a_lp] = butter(4, lpNorm, 'low');
processedEOG = filtfilt(b_lp, a_lp, signalHP);
end


function features = eog_feature_extraction(eogSignal, fs, blinkThreshold, minPeakDistSec)
%EOG_FEATURE_EXTRACTION Extract basic EOG features:
%   - Blink detection (number of blinks, blink rate, blink indices).
%   - Movement density (standard deviation in 5-sec windows).
%   - Slow vs. rapid eye movement power ratio (using pwelch for PSD estimation).
%
% Requires Signal Processing Toolbox.

if nargin < 3, blinkThreshold = 100; end
if nargin < 4, minPeakDistSec = 0.3; end

features = struct();
minPeakDistance = round(minPeakDistSec * fs);

[~, blinkIndices] = findpeaks(abs(eogSignal), ...
    'MinPeakHeight', blinkThreshold, ...
    'MinPeakDistance', minPeakDistance);

features.numBlinks = length(blinkIndices);
features.blinkRatePerMin = features.numBlinks / (length(eogSignal) / fs / 60);
features.blinkIndices = blinkIndices;

% Movement Density in 5-second windows
windowSec = 5;
windowSamples = windowSec * fs;
numWindows = floor(length(eogSignal) / windowSamples);
mdVals = zeros(1, numWindows);
for w = 1:numWindows
    seg = eogSignal((w-1)*windowSamples+1 : w*windowSamples);
    mdVals(w) = std(seg);
end
features.movementDensityMean = mean(mdVals);
features.movementDensityStd  = std(mdVals);

% Slow vs. Rapid Eye Movement Power Ratio using Welch's method
windowLength = 2 * fs;  % e.g., 2-second window
[pxx, f] = pwelch(eogSignal, windowLength, [], [], fs);

slowBand = [0.5 3];
rapidBand = [3 30];
slowIdx = (f >= slowBand(1)) & (f < slowBand(2));
rapidIdx = (f >= rapidBand(1)) & (f <= rapidBand(2));
slowPower = trapz(f(slowIdx), pxx(slowIdx));
rapidPower = trapz(f(rapidIdx), pxx(rapidIdx));
features.slowPower = slowPower;
features.rapidPower = rapidPower;
if rapidPower > 0
    features.powerRatio = slowPower / rapidPower;
else
    features.powerRatio = Inf;
end
end