% --------------------------- 
% Add your custom Scripts folder (where edfread.m is located) to the beginning of the path
%addpath(fullfile(pwd, 'Scripts'), '-begin');

% Refresh the function cache
%rehash toolboxcache;

% Verify that MATLAB now sees your custom edfread.m first
% which edfread -all
% ---------------------------
% ---------------------------
%  Batch EOG Processing and Summary Graphs
% ---------------------------
clc;
close all;
clear all;

%% ----------------------------------------------------------------------
% 1) Locate and load all EDF and XML files in the Data folder
% ----------------------------------------------------------------------
 % --- STEP 0: CLEAN PATH & ADD CUSTOM SCRIPTS ---
    % 1) Restore the default MATLAB path to ensure built-in functions (e.g. butter) are found
    restoredefaultpath;
    rehash toolboxcache;

    % 2) Determine the project root (the folder containing this .m file)
    projectRoot = fileparts(mfilename('fullpath'));

    % 3) Add your custom Scripts folder, so custom edfread overshadows built-in edfread
    addpath(fullfile(projectRoot, 'Scripts'), '-begin');
    rehash toolboxcache;

    % (Optional) Display which versions of edfread and butter are used
    disp('Using edfread from:');
    which edfread -all
    disp('Using butter from:');
    which butter -all

    % --- STEP 1: SETUP & FIND EDF FILES ---
    clc;
    close all;
    clearvars -except projectRoot;  % keep projectRoot if you want

    % Locate the Data folder relative to this script
    dataFolder = fullfile(projectRoot, 'Data');
    edfFiles = dir(fullfile(dataFolder, 'R*.edf'));

    if isempty(edfFiles)
        error('No EDF files matching "R*.edf" found in folder: %s', dataFolder);
    end

% Preallocate a structure array to store summary features
allResults = struct('fileName', {}, 'EOGChannel', {}, ...
    'numBlinks', {}, 'blinkRatePerMin', {}, ...
    'movementDensityMean', {}, 'slowPower', {}, ...
    'rapidPower', {}, 'powerRatio', {});

% Set processing parameters (adjust as needed)
blinkThreshold   = 100;  % amplitude threshold for blink detection
minPeakDistSec   = 0.3;  % minimum peak distance (in seconds)
hpCutoff         = 0.5;  % high-pass cutoff (Hz)
lpCutoff         = 30;   % low-pass cutoff (Hz)

%% ----------------------------------------------------------------------
% 2) Loop over each EDF file and process its EOG channels
% ----------------------------------------------------------------------
for fIdx = 1:length(edfFiles)
    edfName = edfFiles(fIdx).name;
    edfPath = fullfile(dataFolder, edfName);
    
    % Construct matching XML filename (if available)
    [~, baseName] = fileparts(edfName);
    xmlPath = fullfile(dataFolder, [baseName, '.xml']);
    
    fprintf('\n=== Processing file: %s ===\n', edfName);
    
    % Load EDF (and XML if available)
    [hdr, record] = edfread(edfPath);
    if exist(xmlPath, 'file')
        [events, stages, epochLength, annotation] = readXML(xmlPath);
    else
        events = []; stages = []; epochLength = 30; annotation = [];
        fprintf('XML file not found for %s. Proceeding without annotations.\n', edfName);
    end
    
    % (Optional) Display number of 30-sec epochs based on channel 3
    numberOfEpochs = length(record(3,:)) / (30 * hdr.samples(3));
    fprintf('Number of 30-sec epochs (based on channel #3): %.0f\n', numberOfEpochs);
    
    % Identify EOG channels by checking if the label contains 'EOG'
    eogChannelIndices = [];
    for chIdx = 1:length(hdr.label)
        % Use strfind (if strfind returns non-empty, it's a match)
        if ~isempty(strfind(upper(hdr.label{chIdx}), 'EOG'))
            eogChannelIndices(end+1) = chIdx; %#ok<AGROW>
        end
    end
    if isempty(eogChannelIndices)
        warning('No EOG channels found in file %s. Skipping EOG processing.', edfName);
        continue;
    end
    fprintf('EOG channels found in %s: %s\n', edfName, mat2str(eogChannelIndices));
    
    % Process each EOG channel in the file
    for idx = 1:length(eogChannelIndices)
        channelIdx = eogChannelIndices(idx);
        fsEOG = hdr.samples(channelIdx);
        rawEOG = record(channelIdx, :);
        
        % --- EOG Preprocessing: Baseline correction and noise filtering ---
        processedEOG = eog_preprocessing_butter(rawEOG, fsEOG, hpCutoff, lpCutoff);
        
        % --- EOG Feature Extraction ---
        featuresEOG = eog_feature_extraction(processedEOG, fsEOG, blinkThreshold, minPeakDistSec);
        
        % Store results in the structure array
        result.fileName            = edfName;
        result.EOGChannel          = hdr.label{channelIdx};
        result.numBlinks           = featuresEOG.numBlinks;
        result.blinkRatePerMin     = featuresEOG.blinkRatePerMin;
        result.movementDensityMean = featuresEOG.movementDensityMean;
        result.slowPower           = featuresEOG.slowPower;
        result.rapidPower          = featuresEOG.rapidPower;
        result.powerRatio          = featuresEOG.powerRatio;
        allResults(end+1) = result;  %#ok<AGROW>
        
        % Uncomment the following block to visualize a 30-sec segment
        %{
        segDurationSec = 30;
        samplesToPlot  = min(segDurationSec * fsEOG, length(rawEOG));
        tAxis = (0:samplesToPlot-1) / fsEOG;
        figure('Name', ['EOG Compare: ' hdr.label{channelIdx} ' - ' edfName], 'NumberTitle', 'off');
        subplot(2,1,1);
        plot(tAxis, rawEOG(1:samplesToPlot));
        title(['Raw EOG (first ' num2str(segDurationSec) ' s) - ' hdr.label{channelIdx}]);
        ylabel('Amplitude'); xlabel('Time (s)');
        subplot(2,1,2);
        plot(tAxis, processedEOG(1:samplesToPlot));
        hold on;
        blinkIdx = featuresEOG.blinkIndices;
        blinkIdxSegment = blinkIdx(blinkIdx <= samplesToPlot);
        plot((blinkIdxSegment-1)/fsEOG, processedEOG(blinkIdxSegment), 'ro');
        title('Processed EOG (with Blink Markers)');
        ylabel('Amplitude'); xlabel('Time (s)');
        set(gcf, 'color', 'w');
        %}
    end
end

%% ----------------------------------------------------------------------
% 3) Summarise and save results
% ----------------------------------------------------------------------
if isempty(allResults)
    warning('No EOG results to display.');
else
    T = struct2table(allResults);
    disp('=== Summary of EOG Feature Extraction Across All Files ===');
    disp(T);
    
    % Uncomment to save the results to a MAT file or CSV
    % save('EOG_Results.mat', 'allResults');
    % writetable(T, 'EOG_Results.csv');
end

%% ----------------------------------------------------------------------
% 4) Produce Summary Graphs
% ----------------------------------------------------------------------
if exist('T','var') && ~isempty(T)
    % Convert fileName to categorical for plotting
    T.fileName = categorical(T.fileName);
    channels = unique(T.EOGChannel);
    files = categories(T.fileName);
    
    % 4.1: Blink Rate per Minute across Files
    dataBlink = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataBlink(i,j) = mean(T.blinkRatePerMin(idx));
        end
    end
    figure('Name','Blink Rate per Minute','NumberTitle','off');
    bar(dataBlink);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location', 'best');
    xlabel('File');
    ylabel('Blink Rate (per min)');
    title('Blink Rate per Minute Across Files');
    
    % 4.2: Movement Density Mean across Files
    dataMD = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataMD(i,j) = mean(T.movementDensityMean(idx));
        end
    end
    figure('Name','Movement Density Mean','NumberTitle','off');
    bar(dataMD);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location', 'best');
    xlabel('File');
    ylabel('Movement Density Mean');
    title('Movement Density Mean Across Files');
    
    % 4.3: Power Ratio (Slow/Rapid) across Files
    dataPR = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataPR(i,j) = mean(T.powerRatio(idx));
        end
    end
    figure('Name','Power Ratio (Slow/Rapid)','NumberTitle','off');
    bar(dataPR);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location', 'best');
    xlabel('File');
    ylabel('Power Ratio (Slow/Rapid)');
    title('Power Ratio Across Files');
    
    % 4.4: Scatter Plot: Slow Power vs. Rapid Power (points colored by file)
    figure('Name','Slow vs. Rapid Power','NumberTitle','off');
    gscatter(T.slowPower, T.rapidPower, T.fileName);
    xlabel('Slow Power');
    ylabel('Rapid Power');
    title('Slow Power vs. Rapid Power by File');
    legend('Location', 'best');
end



%% ----------------------------------------------------------------------
% 5) Helper functions 
% ----------------------------------------------------------------------

function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
%EOG_PREPROCESSING_BUTTER Preprocess EOG signal using Butterworth filters.
%   1) High-pass filter at hpCutoff to remove baseline drift.
%   2) Low-pass filter at lpCutoff to remove high-frequency noise.
%   Cutoff frequencies are clamped to be within (0, fs/2).
    
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
%   - Slow vs. Rapid eye movement power ratio (via pwelch for PSD estimation).
%
    
    if nargin < 3, blinkThreshold = 100; end
    if nargin < 4, minPeakDistSec = 0.3; end
    
    features = struct();
    minPeakDistance = round(minPeakDistSec * fs);
    
    % Blink Detection
    [~, blinkIndices] = findpeaks(abs(eogSignal), ...
        'MinPeakHeight', blinkThreshold, ...
        'MinPeakDistance', minPeakDistance);
    features.numBlinks = length(blinkIndices);
    features.blinkRatePerMin = features.numBlinks / (length(eogSignal) / fs / 60);
    features.blinkIndices = blinkIndices;
    
    % Movement Density: standard deviation in 5-sec windows
    windowSec = 5;
    windowSamples = windowSec * fs;
    numWindows = floor(length(eogSignal) / windowSamples);
    mdVals = zeros(1, numWindows);
    for w = 1:numWindows
        seg = eogSignal((w-1)*windowSamples+1 : w*windowSamples);
        mdVals(w) = std(seg);
    end
    features.movementDensityMean = mean(mdVals);
    features.movementDensityStd = std(mdVals);
    
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