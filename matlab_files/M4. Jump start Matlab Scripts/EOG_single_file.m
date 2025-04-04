% ---------------------------
% EOG Script
% ---------------------------
clc;
close all;
clear all;


%% ----------------------------------------------------------------------
% 1) Load EDF and XML Files
% -----------------------------------------------------------------------
edfFilename = 'Data/R4.edf';
xmlFilename = 'Data/R4.xml';
[hdr, record] = edfread(edfFilename);
[events, stages, epochLength, annotation] = readXML(xmlFilename);

% check how many 30-sec epochs we have for channel #3
numberOfEpochs = length(record(3,:)) / (30 * hdr.samples(3));
disp(['Number of 30-sec epochs (based on channel #3): ' num2str(numberOfEpochs)]);


%% ----------------------------------------------------------------------
% 2) Plot 1 epoch (30 sec) of each signal
% -----------------------------------------------------------------------
figure('Name','30-sec Epoch of Each Signal','NumberTitle','off');
epochNumber = 1;  % choose which 30-sec epoch to plot
for i = 1:size(record,1)
    Fs = hdr.samples(i);
    epochStart = (epochNumber-1)*Fs*30 + 1;
    epochEnd   = epochStart + 30*Fs - 1;
    if epochEnd > size(record,2)
        error('Epoch %d exceeds available data in channel %d', epochNumber, i);
    end
    sigEpoch = record(i, epochStart:epochEnd);

    subplot(size(record,1), 1, i);
    plot((1:length(sigEpoch))/Fs, sigEpoch);
    ylabel(hdr.label(i));
    xlim([0 30]);
end
sgtitle(['30-second Epoch #' num2str(epochNumber)]);
set(gcf, 'color', 'w');


%% ----------------------------------------------------------------------
% 3) Plot Hypnogram (sleep stages over time)
% -----------------------------------------------------------------------
figure('Name','Hypnogram','NumberTitle','off');
plot(((1:length(stages))*30)./60, stages);
ylim([0 6]);
set(gca, 'YTick', 0:6, 'YTickLabel', {'REM','','N3','N2','N1','Wake',''});
xlabel('Time (Minutes)');
ylabel('Sleep Stage');
box off;
title('Hypnogram');
set(gcf, 'color', 'w');


%% ----------------------------------------------------------------------
% 4) Compute and Plot Correlation Matrix for a Specific 30-sec Epoch
% -----------------------------------------------------------------------
epochNumber = 1;  % any epoch
numChannels = size(record,1);

% Allocate a matrix to hold signals for that epoch
signalsEpoch = zeros(numChannels, 30*min(hdr.samples));
for i = 1:numChannels
    Fs = hdr.samples(i);
    epochStart = (epochNumber-1)*30*Fs + 1;
    epochEnd   = epochStart + 30*Fs - 1;
    if epochEnd > length(record(i,:))
        error('Epoch %d exceeds available data in channel %d', epochNumber, i);
    end
    signalsEpoch(i,1:30*Fs) = record(i, epochStart:epochEnd);
end

corrMatrix = corrcoef(signalsEpoch');
figure('Name','Correlation Matrix','NumberTitle','off');
imagesc(corrMatrix);
colorbar;
title(['Correlation Matrix for 30-second Epoch #' num2str(epochNumber)]);
xlabel('Channels');
ylabel('Channels');
set(gca, 'XTick', 1:numChannels, 'XTickLabel', hdr.label, ...
         'YTick', 1:numChannels, 'YTickLabel', hdr.label);
axis square;


%% ----------------------------------------------------------------------
% 5) EOG Processing and Feature Extraction
% -----------------------------------------------------------------------
% Identify EOG channels by name
eogChannelIndices = [];
for chIdx = 1:length(hdr.label)
    if contains(upper(hdr.label{chIdx}), 'EOG')
        eogChannelIndices(end+1) = chIdx; %#ok<AGROW>
    end
end

if isempty(eogChannelIndices)
    warning('No EOG channels found. Please confirm channel labels.');
else
    disp('EOG channels found at indices:');
    disp(eogChannelIndices);

    % Example parameters
    blinkThreshold   = 100;  % amplitude threshold for blink detection
    minPeakDistSec   = 0.3;  % 0.3 s between peaks
    hpCutoff         = 0.5;  % high-pass at 0.5 Hz
    lpCutoff         = 30;   % low-pass at 30 Hz

    for idx = 1:length(eogChannelIndices)
        channelIdx = eogChannelIndices(idx);
        fsEOG      = hdr.samples(channelIdx);
        rawEOG     = record(channelIdx, :);

        % --- EOG Preprocessing: Baseline correction + noise filtering ---
        preprocessedEOG = eog_preprocessing_butter(rawEOG, fsEOG, hpCutoff, lpCutoff);

        % --- EOG Feature Extraction ---
        featuresEOG = eog_feature_extraction(preprocessedEOG, fsEOG, ...
            blinkThreshold, minPeakDistSec);

        % Display results
        fprintf('\n--- EOG Channel: %s ---\n', hdr.label{channelIdx});
        disp(featuresEOG);

        % Optional: Plot raw vs. preprocessed EOG for a 30-sec segment
        figure('Name',['EOG Compare: ' hdr.label{channelIdx}], 'NumberTitle','off');
        segDurationSec = 30;
        samplesToPlot  = segDurationSec * fsEOG;
        tAxis          = (0:samplesToPlot-1) / fsEOG;

        subplot(2,1,1);
        plot(tAxis, rawEOG(1:samplesToPlot));
        title(['Raw EOG (first ' num2str(segDurationSec) ' s) - ' hdr.label{channelIdx}]);
        ylabel('Amplitude');
        xlabel('Time (s)');

        subplot(2,1,2);
        plot(tAxis, preprocessedEOG(1:samplesToPlot));
        hold on;
        blinkIdx = featuresEOG.blinkIndices;
        blinkIdxSegment = blinkIdx(blinkIdx <= samplesToPlot);
        plot((blinkIdxSegment-1)/fsEOG, preprocessedEOG(blinkIdxSegment), 'ro');
        title('Preprocessed EOG (with Blink Markers)');
        ylabel('Amplitude');
        xlabel('Time (s)');
        set(gcf, 'color', 'w');
    end
end


%% ----------------------------------------------------------------------
%  Helper functions
% ----------------------------------------------------------------------

function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
%EOG_PREPROCESSING_BUTTER Preprocess EOG signal using Butterworth filters
%
% Steps:
%   1) High-pass filter at hpCutoff (e.g., 0.5 Hz) to remove baseline drift
%   2) Low-pass filter at lpCutoff (e.g., 30 Hz) to remove high-frequency noise
%
% If the cutoff frequencies exceed Nyquist (fs/2), they are clamped automatically.

% --- Ensure cutoff frequencies are valid ---
nyquistFreq = fs/2;

% If hpCutoff is below 0 or above nyquistFreq, clamp it
if hpCutoff < 0
    hpCutoff = 0.01;  % or 0.1, any small positive freq
elseif hpCutoff >= nyquistFreq
    hpCutoff = nyquistFreq - 0.01; 
end

% If lpCutoff is above nyquistFreq, clamp it
if lpCutoff >= nyquistFreq
    lpCutoff = nyquistFreq - 0.01;
elseif lpCutoff <= 0
    lpCutoff = 0.5; % fallback if user gave invalid number
end

% Normalise the cutoff frequencies for butter()
hpNorm = hpCutoff / nyquistFreq;
lpNorm = lpCutoff / nyquistFreq;

% --- High-pass filter ---
[b_hp, a_hp] = butter(2, hpNorm, 'high');
signalHP = filtfilt(b_hp, a_hp, eogSignal);

% --- Low-pass filter ---
[b_lp, a_lp] = butter(4, lpNorm, 'low');
processedEOG = filtfilt(b_lp, a_lp, signalHP);
end


function features = eog_feature_extraction(eogSignal, fs, blinkThreshold, minPeakDistSec)
%EOG_FEATURE_EXTRACTION Extract basic EOG features:
%   - Blink detection (# of blinks, blink rate, blink indices)
%   - Movement density (std dev in 5-sec windows)
%   - Slow vs. Rapid eye movement power ratio (via pwelch)
%

if nargin < 3, blinkThreshold = 100; end
if nargin < 4, minPeakDistSec = 0.3; end

features = struct();

%% 1) Blink Detection
minPeakDistance = round(minPeakDistSec * fs);

% Use findpeaks on the absolute value of the signal
[~, blinkIndices] = findpeaks(abs(eogSignal), ...
    'MinPeakHeight', blinkThreshold, ...
    'MinPeakDistance', minPeakDistance);

features.numBlinks = length(blinkIndices);
features.blinkRatePerMin = features.numBlinks / (length(eogSignal)/fs/60);
features.blinkIndices = blinkIndices;

%% 2) Movement Density
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

%% 3) Slow vs. Rapid Eye Movement Power
% Using Welch's method for PSD
windowLength = 2 * fs;  % e.g., 2-second window
[pxx, f] = pwelch(eogSignal, windowLength, [], [], fs);

% Define frequency bands
slowBand = [0.5 3];
rapidBand = [3 30];

slowIdx  = (f >= slowBand(1)) & (f < slowBand(2));
rapidIdx = (f >= rapidBand(1)) & (f <= rapidBand(2));

slowPower  = trapz(f(slowIdx), pxx(slowIdx));
rapidPower = trapz(f(rapidIdx), pxx(rapidIdx));

features.slowPower  = slowPower;
features.rapidPower = rapidPower;
if rapidPower > 0
    features.powerRatio = slowPower / rapidPower;
else
    features.powerRatio = Inf;
end
end