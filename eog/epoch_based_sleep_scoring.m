function epoch_based_sleep_scoring_enhanced
%EPOCH_BASED_SLEEP_SCORING_ENHANCED
%
% Here we have:
%   1) Exclude zero-epoch periods (skip if EOG is all-zero).
%   2) Use an adaptive blink threshold with an amplitude cap (to avoid 
%      "Invalid MinPeakHeight" warnings).
%   3) Optionally remove wavelet-based artifacts from the EOG.
%   4) Compute wavelet-based EOG features as well as classic features.
%   5) Standardize features for classification (especially SVM).
%   6) Use cost-based multi-class SVM if desired.
%   7) Present confusion matrices & accuracy for both Random Forest & support vector machine.
%
% If you see repeated warnings about 'Invalid MinPeakHeight', 
% you can disable them with:
%   warning('off','signal:findpeaks:largeMinPeakHeight');
%
% ---------------------------------------------------------------

    %% User Toggles
    doBalanceRF     = true;    % use 'Prior','uniform' in Random Forest
    doBalanceSVM    = true;    % use cost matrix in SVM
    doWaveletDenoise= true;    % wavelet-based artifact removal for EOG
    doWaveletFeats  = true;    % wavelet-based sub-band features
    doMuteWarnings  = false;   % if true => disable repeated findpeaks warnings

    if doMuteWarnings
        warning('off','signal:findpeaks:largeMinPeakHeight');
    end

    %% STEP 0: PATHS, CLEANUP
    clc; close all;

    restoredefaultpath;
    rehash toolboxcache;

    projectRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(projectRoot, 'Scripts'), '-begin');
    rehash toolboxcache;

    dataFolder = fullfile(projectRoot, 'Data');
    figFolder  = fullfile(projectRoot, 'Figures');
    if ~exist(figFolder, 'dir'), mkdir(figFolder); end

    disp('Using edfread from:'); which edfread -all
    disp('Using butter from:');  which butter  -all

    %% STEP 1: Collect EDF files
    edfFiles = dir(fullfile(dataFolder, 'R*.edf'));
    if isempty(edfFiles)
        error('No EDF files matching R*.edf found in "%s".', dataFolder);
    end
    
    % We'll store epoch-level features & labels from all files:
    allFeatures = [];
    allLabels   = [];

    for fIdx = 1:numel(edfFiles)
        edfName = edfFiles(fIdx).name;
        edfPath = fullfile(dataFolder, edfName);
        [~, baseName] = fileparts(edfName);
        xmlPath = fullfile(dataFolder, [baseName '.xml']);

        fprintf('\n=== Processing %s ===\n', edfName);

        [hdr, record] = edfread(edfPath);
        
        % Attempt to load stage labels from XML
        if exist(xmlPath, 'file')
            [events, stages, epochLength, annotation] = readXML(xmlPath);
        else
            warning('XML for %s not found => using stage = -1 for unknown.', edfName);
            stages = [];
            epochLength = 30;  % fallback
        end

        %% Identify EOG channel(s)
        eogIdx = find(contains(upper(hdr.label), 'EOG'));
        if isempty(eogIdx)
            warning('No EOG channel found in file %s => skipping.', edfName);
            continue;
        end

        % In principle, we could do ICA if we had consistent Fs + fastica, 
        % but user says channels differ => skipping ICA in practice.
        eogCh = eogIdx(1);
        fsEOG = hdr.samples(eogCh);
        eogSignal = double(record(eogCh,:));
        
        totalSamples = numel(eogSignal);
        estEpochs    = floor(totalSamples / (30*fsEOG));
        if isempty(stages)
            stages = -1 * ones(estEpochs,1);
        end
        nEpochs = min(estEpochs, length(stages));

        % Debug arrays
        epochMins  = zeros(nEpochs,1);
        epochMaxs  = zeros(nEpochs,1);
        epochMeans = zeros(nEpochs,1);
        filtMins   = zeros(nEpochs,1);
        filtMaxs   = zeros(nEpochs,1);
        filtMeans  = zeros(nEpochs,1);

        for eIdx = 1:nEpochs
            s1 = (eIdx-1)*30*fsEOG + 1;
            s2 = eIdx*30*fsEOG;
            epochSig = eogSignal(s1:s2);

            % Exclude zero epochs
            if all(epochSig==0)
                continue;
            end

            epochMins(eIdx)  = min(epochSig);
            epochMaxs(eIdx)  = max(epochSig);
            epochMeans(eIdx) = mean(epochSig);

            % Basic filtering
            eogFilt = eog_preprocessing_butter(epochSig, fsEOG, 0.5, 30);

            % Optional wavelet-based artifact removal
            if doWaveletDenoise
                eogFilt = wavelet_denoise_eog(eogFilt, fsEOG);
            end

            filtMins(eIdx)  = min(eogFilt);
            filtMaxs(eIdx)  = max(eogFilt);
            filtMeans(eIdx) = mean(eogFilt);

            % Extract standard EOG features
            feats = eog_feature_extraction(eogFilt, fsEOG);

            % (Optional) wavelet sub-band
            if doWaveletFeats
                featsWV = wavelet_features_eog(eogFilt, fsEOG);
                feats.waveletSlowPower = featsWV.slowPower;
                feats.waveletFastPower = featsWV.fastPower;
                feats.waveletRatio     = featsWV.slowPowerRatio;
            end

            % Build final feature vector
            if doWaveletFeats
                fvec = [feats.blinkRatePerMin, feats.movementDensityMean, ...
                        feats.powerRatio, ...
                        feats.waveletSlowPower, feats.waveletFastPower, ...
                        feats.waveletRatio];
            else
                fvec = [feats.blinkRatePerMin, feats.movementDensityMean, ...
                        feats.powerRatio];
            end

            allFeatures = [allFeatures; fvec]; %#ok<AGROW>
            allLabels   = [allLabels; stages(eIdx)]; %#ok<AGROW>

            if mod(eIdx,50)==0
                fprintf('   [Epoch %3d/%3d] rawMean=%.2f rawRange=[%.2f %.2f], filtMean=%.2f\n',...
                    eIdx, nEpochs, epochMeans(eIdx), epochMins(eIdx), epochMaxs(eIdx), filtMeans(eIdx));
            end
        end

        %% Print debug summary
        if nEpochs>0
            fprintf('\n--- Debug Summary for file: %s ---\n', edfName);
            fprintf('  # of epochs: %d\n', nEpochs);

            rawMin  = min(epochMins);
            rawMax  = max(epochMaxs);
            rawMean = mean(epochMeans);
            fprintf('  RAW EOG overall: min=%.2f, max=%.2f, avgMean=%.2f\n',...
                rawMin, rawMax, rawMean);

            fm = mean(filtMeans);
            fprintf('  FILTERED EOG overall: min=%.2f, max=%.2f, avgMean=%.2f\n',...
                min(filtMins), max(filtMaxs), fm);

            zE = sum(epochMaxs==0 & epochMins==0);
            if zE>0
                fprintf('  ** # of epochs with ALL-ZERO data: %d\n', zE);
            end
            fprintf('-------------------------------\n');
        end
    end

    %% STEP 2: Build dataset & skip unknown or zero-labeled epochs
    if isempty(allFeatures)
        warning('No features to classify => aborting.');
        return;
    end
    
    % If you want to skip unknown stages (like -1):
    %  keepIdx = (allLabels>=0 & allLabels<=5);
    %  X = allFeatures(keepIdx,:);
    %  Ynum = allLabels(keepIdx);
    X = allFeatures;
    Ynum = allLabels;

    Ycat = categorical(Ynum);

    % If there's <2 classes, nothing to do
    uC = unique(Ycat);
    if numel(uC)<2
        warning('Not enough distinct classes => cannot classify.');
        return;
    end

    %% STEP 3: Classification (5-fold CV) with standardization

    nTotal = size(X,1);
    KF = 5; 
    cvo = cvpartition(Ycat,'KFold',KF);

    predictedRF  = cell(nTotal,1);
    predictedSVM = cell(nTotal,1);

    % We define cost matrix of ones off-diagonal if doBalanceSVM
    unqClasses = categories(Ycat);
    nClasses   = numel(unqClasses);
    costMat    = ones(nClasses) - eye(nClasses);

    for fold = 1:KF
        trainIdx = training(cvo,fold);
        testIdx  = test(cvo,fold);

        Xtrain = X(trainIdx,:);
        Ytrain = Ycat(trainIdx);
        Xtest  = X(testIdx,:);
        Ytest  = Ycat(testIdx);

        % ---- Feature standardization ----
        mu = mean(Xtrain,1);
        sd = std(Xtrain,1)+1e-8;   % avoid division by zero
        XtrainZ = (Xtrain - mu)./ sd;
        XtestZ  = (Xtest  - mu)./ sd;

        %% 3.1 Random Forest
        if doBalanceRF
            RFmodel = TreeBagger(50, XtrainZ, Ytrain, ...
                'OOBPrediction','off','Prior','uniform');
        else
            RFmodel = TreeBagger(50, XtrainZ, Ytrain, 'OOBPrediction','off');
        end

        YhatRF = predict(RFmodel, XtestZ);
        predictedRF(testIdx) = YhatRF;

        %% 3.2 SVM
        % We'll do multi-class SVM with auto kernel scale
        template = templateSVM('KernelFunction','rbf',...
            'KernelScale','auto','Standardize',false); 

        if doBalanceSVM
            SVMmodel = fitcecoc(XtrainZ, Ytrain, 'Learners', template, ...
                'Coding','onevsone','ClassNames',unqClasses, 'Cost',costMat);
        else
            SVMmodel = fitcecoc(XtrainZ, Ytrain, 'Learners', template, ...
                'Coding','onevsone','ClassNames',unqClasses);
        end

        YhatSVM = predict(SVMmodel, XtestZ);
        predictedSVM(testIdx) = YhatSVM;
    end

    predictedRF  = categorical(predictedRF);
    predictedSVM = categorical(predictedSVM);

    accRF  = mean(predictedRF==Ycat);
    accSVM = mean(predictedSVM==Ycat);

    cmRF  = confusionmat(Ycat, predictedRF);
    cmSVM = confusionmat(Ycat, predictedSVM);

    %% Final results
    fprintf('\n=== EPOCH-BASED CLASSIFICATION RESULTS ===\n');
    fprintf('Random Forest Accuracy: %.2f%%\n', 100*accRF);
    fprintf('SVM Accuracy:          %.2f%%\n', 100*accSVM);
    disp('Confusion Matrix (RF):');  disp(cmRF);
    disp('Confusion Matrix (SVM):'); disp(cmSVM);

    figC = figure('Name','Confusion Matrices','NumberTitle','off');
    subplot(1,2,1);
    imagesc(cmRF);
    title(sprintf('RF (Acc=%.2f%%)',100*accRF));
    xlabel('Predicted'); ylabel('True'); colorbar;

    subplot(1,2,2);
    imagesc(cmSVM);
    title(sprintf('SVM (Acc=%.2f%%)',100*accSVM));
    xlabel('Predicted'); ylabel('True'); colorbar;
    set(gcf,'Color','w');

    saveas(figC, fullfile(figFolder,'EpochBased_ConfusionMatrices.png'));
    close(figC);

    %% Re-enable warnings, if you turned them off
    if doMuteWarnings
        warning('on','signal:findpeaks:largeMinPeakHeight');
    end
end

%% ------------------------------------------------------------------------
% HELPER FUNCTIONS
%% ------------------------------------------------------------------------
function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
% EOG_PREPROCESSING_BUTTER - Basic Butterworth filtering for EOG
%
% 1) High-pass filter at hpCutoff (e.g. 0.5 Hz) to remove baseline drift.
% 2) Low-pass filter at lpCutoff (e.g. 30 Hz) to remove high-frequency noise.
%
% If the user-specified cutoff frequencies are invalid or too close to Nyquist, 
% they are clamped automatically. The filtering is done with zero-phase 
% filtfilt() to avoid introducing a phase shift.
%
% INPUTS:
%   eogSignal - raw EOG data (1D array)
%   fs        - sampling frequency (Hz)
%   hpCutoff  - high-pass cutoff in Hz
%   lpCutoff  - low-pass cutoff in Hz
%
% OUTPUT:
%   processedEOG - filtered signal


% Basic Butterworth HP/LP
    nyq = fs/2;
    hpCutoff = max(0.01, min(hpCutoff, nyq-0.01));
    lpCutoff = max(0.5,  min(lpCutoff, nyq-0.01));

    [b_hp,a_hp] = butter(2, hpCutoff/nyq, 'high');
    tmp = filtfilt(b_hp, a_hp, double(eogSignal));

    [b_lp,a_lp] = butter(4, lpCutoff/nyq, 'low');
    processedEOG = filtfilt(b_lp, a_lp, tmp);
end

function eogClean = wavelet_denoise_eog(eogSignal, ~)
% WAVELET_DENOISE_EOG  Simple wavelet-based artifact removal
%
% We use wdenoise() with a particular wavelet (e.g. 'db4'), universal threshold,
% and soft thresholding. This helps remove large spikes or abrupt jumps 
% that are typical artifacts in EOG signals (due to movement or electrode pop).
%
% Adjust wavelet name, level, threshold method as needed for your data.
% Simple wavelet-based artifact removal
    level = 4;
    wname = 'db4';
    eogClean = wdenoise(eogSignal, level, 'Wavelet', wname, ...
        'DenoisingMethod','UniversalThreshold',...
        'ThresholdRule','Soft');
end

function feats = eog_feature_extraction(eogSignal, fs)
% EOG_FEATURE_EXTRACTION standard + adaptive blink threshold
% EOG_FEATURE_EXTRACTION - Extract classic EOG features from a 30s epoch
%
% This function includes:
%   1) Blink detection & blink rate
%      - Uses an adaptive threshold based on the mean & std of the EOG 
%        plus a capping fraction of the maximum amplitude.
%      - We then call findpeaks() on the absolute EOG with 'MinPeakHeight'
%        = blinkThreshold, and 'MinPeakDistance' to avoid counting the same
%        blink multiple times if it has multiple local maxima.
%
%   2) Movement density
%      - The EOG epoch (30s) is broken into smaller 5s windows. The standard
%        deviation of each 5s segment is computed. The average of these 
%        standard deviations is the "movementDensityMean".
%      - Higher movement density indicates more saccades or "busy" ocular activity.
%
%   3) Slow vs. Rapid eye movement power ratio
%      - We compute the PSD (pwelch) with a typical 2s window.
%      - Define "slow" band as ~0.5-3 Hz, "rapid" band as 3-30 Hz.
%      - Integrate power (via trapz) in those freq bands.
%      - The ratio slowPower/rapidPower can help distinguish REM vs. non-REM,
%        or highlight big slow deflections vs. fast saccades.
%
% INPUTS:
%   eogSignal - 1D EOG data (already filtered for baseline drift & noise)
%   fs        - sampling frequency (Hz)
%
% OUTPUT (struct):
%   .numBlinks            (# of blink events found)
%   .blinkRatePerMin      (blinks per minute in this epoch)
%   .blinkIndices         (sample indices where the peaks occurred)
%   .movementDensityMean  (avg std dev across 5s windows)
%   .movementDensityStd   (std dev of those window-wise stds)
%   .slowPower            (area under PSD in [0.5, 3) Hz)
%   .rapidPower           (area under PSD in [3,   30] Hz)
%   .powerRatio           (slowPower/rapidPower, or Inf if rapid=0)

    feats = struct();

    % 1) Adaptive threshold
    peakStdFactor = 3;
    sigStd  = std(eogSignal);
    sigMean = mean(abs(eogSignal));
    rawThresh = sigMean + peakStdFactor*sigStd;

    mx = max(abs(eogSignal));
    capVal = 0.9*mx;
    blinkThreshold = min(rawThresh, capVal);
    blinkThreshold = max(blinkThreshold, 20); 
    if blinkThreshold<1
        blinkThreshold=1;
    end

    minPeakDistSec = 0.3;
    minPD = round(fs*minPeakDistSec);

    if max(abs(eogSignal))<blinkThreshold
        blinkIdx = [];
    else
        [~,blinkIdx] = findpeaks(abs(eogSignal), ...
            'MinPeakHeight',blinkThreshold, ...
            'MinPeakDistance',minPD);
    end
    feats.numBlinks = numel(blinkIdx);
    feats.blinkRatePerMin = feats.numBlinks / (numel(eogSignal)/fs/60);

    % ------------------
    % 2) Movement Density
    % ------------------
    % We subdivide the 30s epoch into 5s windows => 6 sub-windows if 30s.
    % For each sub-window, compute std. Then average them => movementDensityMean.
    wSec = 5;
    wSamp= wSec*fs;
    nWin = floor(numel(eogSignal)/wSamp);
    mdVals= zeros(nWin,1);
    for w=1:nWin
        seg = eogSignal((w-1)*wSamp+1:w*wSamp);
        mdVals(w)= std(seg);
    end
    feats.movementDensityMean = mean(mdVals);

    % ------------------
    % 3) Slow vs. Rapid Eye Movement Power Ratio
    % ------------------
    % Use pwelch with a 2s window. Then define:
    %   slow band  ~ 0.5-3 Hz
    %   rapid band ~ 3-30 Hz
    % Integrate the PSD in those bands, ratio = slowPower/rapidPower
    [pxx,f] = pwelch(eogSignal,2*fs,[],[],fs);
    slowIdx  = (f>=0.5 & f<3);
    rapidIdx = (f>=3   & f<=30);

    slowP  = trapz(f(slowIdx),  pxx(slowIdx));
    rapidP = trapz(f(rapidIdx), pxx(rapidIdx));

    feats.slowPower  = slowP;
    feats.rapidPower = rapidP;
    if rapidP>0
        feats.powerRatio = slowP/rapidP;
    else
        feats.powerRatio = Inf;
    end
end

function featsWV = wavelet_features_eog(eogSignal, fs)
% WAVELET_FEATURES_EOG approximate slow (<3 Hz) vs. fast (3-30 Hz)
% WAVELET_FEATURES_EOG - Extract wavelet-based slow & fast band energies
%
% This example uses a Discrete Wavelet Packet Transform (DWPT) to 
% reconstruct approximate "slow" (< 3 Hz) vs. "fast" (3-30 Hz) sub-bands.
% We then compute the mean-squared amplitude (energy) in each sub-band.
%
% Depending on your wavelet decomposition tree, you can refine how you
% pick "slow" and "fast" nodes. Here, we simply choose the first 1 or 2
% nodes for "slow" and the rest for "fast". This is approximate, but can
% be refined if you want more precise frequency partitions.
%
% INPUTS:
%   eogSignal - a 30s EOG segment (filtered) 
%   fs        - sampling frequency
%
% OUTPUT (struct):
%   .slowPower        - mean of squared amplitude in slow band
%   .fastPower        - mean of squared amplitude in fast band
%   .slowPowerRatio   - slowPower / fastPower
    featsWV = struct('slowPower',0,'fastPower',0,'slowPowerRatio',0);

    level = 4;
    wname = 'sym3';

    [wpt,bk,~,~] = dwpt(eogSignal, wname, 'Level',level);

     % We'll define "slowNodes" to approximate <3 Hz sub-bands,
    % and "fastNodes" for the remainder, typically 3–30 Hz or so.
    slowNodes = [1 2];
    fastNodes = 3:16;

    % Reconstruct slow band
    wpt_slow = wpt;
    for i=1:numel(wpt)
        if ~ismember(i, slowNodes)
            wpt_slow{i} = zeros(size(wpt_slow{i}));
        end
    end
    slowSig  = idwpt(wpt_slow, bk, wname);
    slowPow  = mean(slowSig.^2);

    % Reconstruct fast band
    wpt_fast = wpt;
    for i=1:numel(wpt)
        if ~ismember(i, fastNodes)
            wpt_fast{i} = zeros(size(wpt_fast{i}));
        end
    end
    fastSig  = idwpt(wpt_fast, bk, wname);
    fastPow  = mean(fastSig.^2);

    featsWV.slowPower     = slowPow;
    featsWV.fastPower     = fastPow;
    featsWV.slowPowerRatio= Inf;
    if fastPow>0
        featsWV.slowPowerRatio = slowPow / fastPow;
    end
end
