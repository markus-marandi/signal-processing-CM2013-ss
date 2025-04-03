clc; clear; close all;

%% ----------------------------------------------------------------------
% 1) Load EDF Files & Identify EOG Channels
% ----------------------------------------------------------------------
projectRoot = fileparts(mfilename('fullpath'));
dataFolder = fullfile(projectRoot, 'Data');
edfFiles = dir(fullfile(dataFolder, 'R*.edf'));

if isempty(edfFiles)
    error('No EDF files found. Ensure they are in the Data folder.');
end

% Process only the first file for noise analysis
edfName = edfFiles(1).name;
edfPath = fullfile(dataFolder, edfName);
fprintf('\nAnalyzing powerline noise in: %s\n', edfName);

% Load EDF
[hdr, record] = edfread(edfPath);

% Identify EOG channels
eogChannelIndices = find(contains(upper(hdr.label), 'EOG'));
if isempty(eogChannelIndices)
    error('No EOG channels found in file: %s', edfName);
end

%% ----------------------------------------------------------------------
% 2) Compute & Plot Power Spectrum for EOG Channels
% ----------------------------------------------------------------------
figure('Name', 'Power Spectral Density - EOG Channels', 'NumberTitle', 'off');
notchRequired = false;
fsEOG = hdr.samples(eogChannelIndices(1)); % Assume all EOG channels have the same Fs
hold on;

for idx = 1:length(eogChannelIndices)
    channelIdx = eogChannelIndices(idx);
    eogSignal = record(channelIdx, :);
    
    % Compute PSD using Welch’s method
    [pxx, f] = pwelch(eogSignal, fsEOG, [], [], fsEOG);
    
    % Check for powerline interference (50 Hz or 60 Hz peaks)
    power50Hz = max(pxx(abs(f - 50) < 0.5));
    power60Hz = max(pxx(abs(f - 60) < 0.5));
    maxPower = max(pxx); % Ensure maxPower is a scalar
    
    % Ensure power50Hz and power60Hz are scalars
    power50Hz = max(power50Hz(:));
    power60Hz = max(power60Hz(:));
    
    % Determine if notch filter is needed (threshold: 10% of max power)
    if any(power50Hz > 0.1 * maxPower) || any(power60Hz > 0.1 * maxPower)
        notchRequired = true;
    end
    
    % Plot PSD
    plot(f, 10*log10(pxx), 'DisplayName', hdr.label{channelIdx});
end

xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density of EOG Channels');
legend show;
grid on;
hold off;

%% ----------------------------------------------------------------------
% 3) Display Notch Filter Recommendation
% ----------------------------------------------------------------------
if notchRequired
    fprintf('Powerline noise detected! Applying Notch Filter is recommended.\n');
else
    fprintf('No significant powerline noise detected. Notch Filter is not needed.\n');
end