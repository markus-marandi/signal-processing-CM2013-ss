function main_EEG_EOG_ICA_Classification
% MAIN_EEG_EOG_ICA_CLASSIFICATION
%   1) ICA for EEG+EOG
%   2) Single-label classification per file (Random Forest & SVM)
%   3) Subplots & auto-save figures to PNG
%

    %% ---------------------------
    %  STEP 0: PATH SETUP
    % ---------------------------
    clc; close all; clear;

    % Restore default path so built-in functions are accessible
    restoredefaultpath;
    rehash toolboxcache;

    projectRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(projectRoot, 'Scripts'), '-begin');
    rehash toolboxcache;

    % Optional debug info
    disp('Using edfread from:'); which edfread -all
    disp('Using butter from:'); which butter -all

    % Data folder & figure output folder
    dataFolder = fullfile(projectRoot, 'Data');
    figFolder  = fullfile('/Users','markus','university','signal-processing-CM2013-ss','eog','figures');
    if ~exist(figFolder, 'dir')
        mkdir(figFolder);
    end

    % Find all EDF files
    edfFiles = dir(fullfile(dataFolder, 'R*.edf'));
    if isempty(edfFiles)
        error('No EDF files matching R*.edf found in %s', dataFolder);
    end

    % Parameters for EOG processing
    blinkThreshold   = 100;
    minPeakDistSec   = 0.3;   % min blink distance in seconds
    hpCutoff         = 0.5;   % HP filter cutoff
    lpCutoff         = 30;    % LP filter cutoff

    %% ---------------------------
    %  STEP 1: PROCESS EACH FILE
    % ---------------------------
    allResults = [];   % Holds EOG feature results
    allLabels  = [];   % Holds numeric label for classification
    labelNames = {};   % Holds text label (Wake, N1, etc.), for reference

    for fIdx = 1:length(edfFiles)
        edfName = edfFiles(fIdx).name;
        edfPath = fullfile(dataFolder, edfName);

        [~, baseName] = fileparts(edfName);
        xmlPath = fullfile(dataFolder, [baseName '.xml']);

        fprintf('\n=== Processing file: %s ===\n', edfName);
        [hdr, record] = edfread(edfPath);

        % (Optional) Attempt to load XML (sleep stages)
        if exist(xmlPath, 'file')
            [events, stages, _Length, annotation] = readXML(xmlPath);
        else
            events = []; stages = []; epochLength = 30; annotation = [];
            fprintf('XML file not found for %s. Proceeding without annotations.\n', edfName);
        end

        % 1. Identify EEG channels
        %    (assuming channel labels contain 'EEG' or a known pattern, adapt as needed)
        eegChannelIdx = find(contains(upper(hdr.label), 'EEG'));
        % 2. Identify EOG channels
        eogChannelIdx = find(contains(upper(hdr.label), 'EOG'));

        if isempty(eogChannelIdx)
            warning('No EOG channel in %s. Skipping EOG processing.', edfName);
            continue;
        end

        % Show # of 30s epochs using channel 3 as reference (optional)
        try
            nEpochs = length(record(3,:)) / (30*hdr.samples(3));
            fprintf('Approx. # of 30s epochs (ch3): %d\n', nEpochs);
        catch
            % If channel 3 doesn't exist, skip quietly
        end

        %% ----------------------------------------------------------
        %  STEP 2: (Optional) ICA to remove cross-talk from EEG + EOG
        % ----------------------------------------------------------
        % Combine all EEG + EOG into one matrix for ICA, if they share same Fs
        % If different Fs exist, you'd have to resample them. Here, we assume
        % the same or you adapt code to handle differences.
        chanIndices = [eegChannelIdx(:); eogChannelIdx(:)];
        if ~isempty(chanIndices)
            % Figure out sample rates
            fsList = hdr.samples(chanIndices);
            if length(unique(fsList)) == 1
                fs = fsList(1);  % single sampling freq
            else
                warning('Channels have different Fs. Skipping ICA step.');
                % We won't do ICA if sampling rates differ. (Or resample as needed.)
                fs = hdr.samples(eogChannelIdx(1)); % fallback to EOG
            end

            % Extract data
            dataMat = double(record(chanIndices, :));

            % Run fastICA
            try
                fprintf('Running ICA on EEG+EOG channels...\n');
                % You may need: addpath('fastICA') if not on path
                [icasig, A, W] = fastica(dataMat, 'numOfIC', size(dataMat,1), ...
                                         'verbose','off','displayMode','off');
                % Reconstruct signals. Typically you'd identify artifact comps
                % and zero them out in icasig. Because we want to keep EOG blink,
                % we won't remove those. But you could remove EOG from EEG.
                % For demonstration, we do not remove any comp here:
                dataClean = A * icasig;  %#ok<NASGU> 
                % dataClean is the same as dataMat if you haven't removed comps.

                % If you wanted to remove blink from EEG, you'd identify comps
                % that are strongly correlated with EOG channels & zero them out.

            catch me
                warning('ICA failed: %s', me.message);
            end

        else
            fs = hdr.samples(eogChannelIdx(1)); % fallback
        end

        %% ----------------------------------------------------------
        %  STEP 3: PROCESS EOG (Filtering, Feature Extraction)
        % ----------------------------------------------------------
        for idxEOG = 1:length(eogChannelIdx)
            chEOG = eogChannelIdx(idxEOG);
            fsEOG = hdr.samples(chEOG);
            rawEOG = record(chEOG, :);

            % Preprocessing
            processedEOG = eog_preprocessing_butter(rawEOG, fsEOG, hpCutoff, lpCutoff);

            % EOG Feature Extraction
            feats = eog_feature_extraction(processedEOG, fsEOG, blinkThreshold, minPeakDistSec);

            % Store in allResults
            rowData.fileName            = edfName;
            rowData.EOGChannel          = hdr.label{chEOG};
            rowData.numBlinks           = feats.numBlinks;
            rowData.blinkRatePerMin     = feats.blinkRatePerMin;
            rowData.movementDensityMean = feats.movementDensityMean;
            rowData.slowPower           = feats.slowPower;
            rowData.rapidPower          = feats.rapidPower;
            rowData.powerRatio          = feats.powerRatio;
            allResults = [allResults; rowData]; %#ok<AGROW>
        end

        %% ----------------------------------------------------------
        %  STEP 4: Determine Single Label for the Entire File
        % ----------------------------------------------------------
        % If 'stages' is a vector of numeric codes (0,1,2,3,4,5 for REM,N1...), or
        % if it's textual, adapt code here.
        if ~isempty(stages)
            % Example approach: choose the most frequent stage in the entire recording
            % (typical codes: 0=Wake,1=N1,2=N2,3=N3,5=REM or similar)
            uniqueStages = unique(stages);
            modeStage = mode(stages);
            % Save numeric label
            allLabels(end+1) = modeStage; %#ok<AGROW>
            labelText = stage_code_to_text(modeStage); % local function below
            labelNames{end+1} = labelText; %#ok<AGROW>
            fprintf('Dominant stage for %s is: %s\n', edfName, labelText);
        else
            % If no annotation, store -1
            allLabels(end+1) = -1; %#ok<AGROW>
            labelNames{end+1} = 'Unknown'; %#ok<AGROW>
        end

        %% ----------------------------------------------------------
        % OPTIONAL: Plot an example segment with EOG & Save PNG
        % ----------------------------------------------------------
        if fIdx==1 && ~isempty(eogChannelIdx)
            segSec = 30; 
            N = min(segSec*fsEOG, length(rawEOG));
            tAxis = (0:N-1)/fsEOG;

            figH = figure('Name',['EOG Example - ' edfName],'NumberTitle','off');
            subplot(2,1,1);
            plot(tAxis, rawEOG(1:N));
            title(['Raw EOG (' rowData.EOGChannel ') - first ' num2str(segSec) ' s']);
            ylabel('Amplitude'); xlabel('Time (s)');

            subplot(2,1,2); hold on;
            plot(tAxis, processedEOG(1:N));
            plot((feats.blinkIndices(feats.blinkIndices<=N)-1)/fsEOG, ...
                 processedEOG(feats.blinkIndices(feats.blinkIndices<=N)), 'ro');
            title('Processed EOG (with Blink Markers)');
            ylabel('Amplitude'); xlabel('Time (s)');
            set(gcf, 'color','w');

            saveas(figH, fullfile(figFolder, ['EOG_Sample_' edfName '.png']));
            close(figH);
        end

    end  % end for each file

    %% ---------------------------
    %  STEP 5: Summarize + Classification
    % ---------------------------
    if isempty(allResults)
        warning('No EOG results. Aborting classification step.');
        return;
    end

    % Convert struct array to table
    T = struct2table(allResults);
    disp('=== Summary of EOG Feature Extraction ===');
    disp(T);

    % Combine results by file: we want 1 feature vector per file
    % Because you have multiple EOG channels per file, we can either
    % (A) average across EOG channels or
    % (B) just pick one EOG channel.
    % For demonstration, let's do (A) average:

    [uniqueFiles,~,idxFiles] = unique(T.fileName);
    nFiles = length(uniqueFiles);
    featureMatrix = [];
    for iF = 1:nFiles
        theseRows = (idxFiles==iF);
        blinkRate = mean(T.blinkRatePerMin(theseRows));
        moveDens = mean(T.movementDensityMean(theseRows));
        powerRat = mean(T.powerRatio(theseRows));
        % Add more if you want: slowPower, rapidPower, etc.
        featureMatrix(iF,:) = [blinkRate, moveDens, powerRat]; %#ok<AGROW>
    end

    % Get numeric labels from allLabels, matching uniqueFiles order
    numericLabels = [];
    textLabels    = {};
    for iF = 1:nFiles
        fileMask = strcmp(uniqueFiles{iF}, {allResults.fileName});
        % The label is the same for each row in that file, so:
        numericLabels(iF,1) = allLabels(find(fileMask,1)); %#ok<AGROW>
        textLabels{iF,1}    = labelNames{find(fileMask,1)}; %#ok<AGROW>
    end

    % Remove unknown-labeled files (if any had no XML)
    validIdx = (numericLabels>=0);
    featureMatrix = featureMatrix(validIdx,:);
    numericLabels = numericLabels(validIdx);
    textLabels    = textLabels(validIdx);
    validFiles    = uniqueFiles(validIdx);

    % If all are unknown or only 1 class, no classification can be done
    if numel(unique(numericLabels))<2
        warning('Not enough classes to train classifier. Aborting...');
        return;
    end

    % Map numericLabels to a categorical or keep numeric
    Y = categorical(numericLabels);

    %% ---------------------------
    %  TRAIN + COMPARE CLASSIFIERS
    % ---------------------------
    % We'll do a simple cross-validation over the set of files (leave-one-out).
    nValidFiles = sum(validIdx);
    predictedRF  = nan(nValidFiles,1);
    predictedSVM = nan(nValidFiles,1);

    for testIdx = 1:nValidFiles
        % Indices
        trainSet = true(nValidFiles,1);
        trainSet(testIdx) = false;
        testSet = ~trainSet;

        Xtrain = featureMatrix(trainSet,:);
        Ytrain = Y(trainSet);
        Xtest  = featureMatrix(testSet,:);
        Ytest  = Y(testSet);

        % 1) Train Random Forest
        RFmodel = TreeBagger(50, Xtrain, Ytrain, 'OOBPrediction','off');
        YhatRF  = predict(RFmodel, Xtest);
        predictedRF(testIdx) = double(YhatRF{1}); % convert cell->char->double if numeric

        % 2) Train SVM
        SVMmodel = fitcsvm(Xtrain, Ytrain, 'KernelFunction','rbf');
        YhatSVM  = predict(SVMmodel, Xtest);
        predictedSVM(testIdx) = double(YhatSVM);
    end

    % Evaluate
    trueLabelsNum = double(Y);
    [accRF, cmRF]   = getAccuracyCM(trueLabelsNum, predictedRF);
    [accSVM, cmSVM] = getAccuracyCM(trueLabelsNum, predictedSVM);

    % Print results
    fprintf('\n=== Classification Results (Entire Signal) ===\n');
    fprintf('Random Forest Accuracy: %.2f %%\n', accRF*100);
    fprintf('SVM Accuracy:           %.2f %%\n', accSVM*100);

    disp('Confusion Matrix (RF):');
    disp(cmRF);
    disp('Confusion Matrix (SVM):');
    disp(cmSVM);

    %% --- Visualize Confusion Matrices ---
    figH = figure('Name','Confusion Matrices','NumberTitle','off');
    subplot(1,2,1);
    imagesc(cmRF);
    title(sprintf('RF (Acc=%.2f%%)',accRF*100));
    ylabel('True Class'); xlabel('Predicted Class');
    colorbar;

    subplot(1,2,2);
    imagesc(cmSVM);
    title(sprintf('SVM (Acc=%.2f%%)',accSVM*100));
    ylabel('True Class'); xlabel('Predicted Class');
    colorbar;
    set(gcf, 'color','w');

    saveas(figH, fullfile(figFolder,'ConfusionMatrices.png'));
    close(figH);

    %% --- Final Summaries and Figures ---
    % (Similar to your original bar plots, plus auto-save)
    if ~isempty(T)
        visualize_and_save_barplots(T, figFolder);
    end

end % end main function

%% ------------------------------------------------------------------------
%  Local Functions
%% ------------------------------------------------------------------------

function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
    % EOG_PREPROCESSING_BUTTER
    nyquistFreq = fs/2;
    hpCutoff = max(0.01, min(hpCutoff, nyquistFreq-0.01));
    lpCutoff = max(0.5,  min(lpCutoff, nyquistFreq-0.01));

    hpNorm = hpCutoff/nyquistFreq;
    lpNorm = lpCutoff/nyquistFreq;

    [b_hp, a_hp] = butter(2, hpNorm, 'high');
    sigHP = filtfilt(b_hp, a_hp, double(eogSignal));

    [b_lp, a_lp] = butter(4, lpNorm, 'low');
    processedEOG = filtfilt(b_lp, a_lp, sigHP);
end

function features = eog_feature_extraction(eogSignal, fs, blinkThreshold, minPeakDistSec)
    if nargin<3, blinkThreshold=100; end
    if nargin<4, minPeakDistSec=0.3; end

    features = struct();
    minPeakDistance = round(minPeakDistSec * fs);

    % Blinks
    [~, blinkIndices] = findpeaks(abs(eogSignal), 'MinPeakHeight', blinkThreshold,...
                                  'MinPeakDistance', minPeakDistance);
    features.numBlinks = length(blinkIndices);
    features.blinkRatePerMin = length(blinkIndices)/(length(eogSignal)/fs/60);
    features.blinkIndices = blinkIndices;

    % Movement density
    windowSec = 5; 
    winSamples = windowSec*fs;
    nWins = floor(length(eogSignal)/winSamples);
    mdVals = zeros(nWins,1);
    for w=1:nWins
        seg = eogSignal((w-1)*winSamples+1 : w*winSamples);
        mdVals(w) = std(seg);
    end
    features.movementDensityMean = mean(mdVals);
    features.movementDensityStd  = std(mdVals);

    % Slow vs Rapid Eye Movement (PSD)
    [pxx, f] = pwelch(eogSignal, 2*fs, [], [], fs);
    slowBand  = [0.5 3];
    rapidBand = [3   30];
    slowIdx  = (f>=slowBand(1)) & (f<slowBand(2));
    rapidIdx = (f>=rapidBand(1)) & (f<=rapidBand(2));
    slowPower  = trapz(f(slowIdx), pxx(slowIdx));
    rapidPower = trapz(f(rapidIdx), pxx(rapidIdx));
    features.slowPower  = slowPower;
    features.rapidPower = rapidPower;
    if rapidPower>0
        features.powerRatio = slowPower/rapidPower;
    else
        features.powerRatio = Inf;
    end
end

function stageText = stage_code_to_text(stageCode)
    % STAGE_CODE_TO_TEXT Convert numeric stage code to text
    % Common coding: 0=Wake,1=N1,2=N2,3=N3,5=REM
    switch stageCode
        case 0
            stageText = 'Wake';
        case 1
            stageText = 'N1';
        case 2
            stageText = 'N2';
        case 3
            stageText = 'N3';
        case 4
            stageText = 'N3';  % Some datasets use 4 for deep sleep
        case 5
            stageText = 'REM';
        otherwise
            stageText = 'Unknown';
    end
end

function [acc, cm] = getAccuracyCM(yTrue, yPred)
    % GETACCURACYCM  Return accuracy and confusion matrix
    uniqueClasses = unique([yTrue; yPred]);
    cm = zeros(length(uniqueClasses));
    for i = 1:length(yTrue)
        row = find(uniqueClasses==yTrue(i));
        col = find(uniqueClasses==yPred(i));
        cm(row,col) = cm(row,col)+1;
    end
    acc = sum(diag(cm))/sum(cm(:));
end

function visualize_and_save_barplots(T, figFolder)
    % VISUALIZE_AND_SAVE_BARPLOTS  Recreates the bar plots from your original script,
    % plus auto-save to PNG.
    T.fileName = categorical(T.fileName);
    channels = unique(T.EOGChannel);
    files = categories(T.fileName);

    % Blink Rate
    dataBlink = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataBlink(i,j) = mean(T.blinkRatePerMin(idx));
        end
    end
    fig1 = figure('Name','Blink Rate per Minute','NumberTitle','off');
    bar(dataBlink);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location','best');
    xlabel('File'); ylabel('Blink Rate (per min)');
    title('Blink Rate per Minute Across Files');
    saveas(fig1, fullfile(figFolder,'BlinkRatePerMin.png'));
    close(fig1);

    % Movement Density
    dataMD = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataMD(i,j) = mean(T.movementDensityMean(idx));
        end
    end
    fig2 = figure('Name','Movement Density Mean','NumberTitle','off');
    bar(dataMD);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location','best');
    xlabel('File'); ylabel('Movement Density Mean');
    title('Movement Density Mean Across Files');
    saveas(fig2, fullfile(figFolder,'MovementDensityMean.png'));
    close(fig2);

    % Power Ratio
    dataPR = zeros(length(files), length(channels));
    for i = 1:length(files)
        for j = 1:length(channels)
            idx = (T.fileName == files{i}) & strcmp(T.EOGChannel, channels{j});
            dataPR(i,j) = mean(T.powerRatio(idx));
        end
    end
    fig3 = figure('Name','Power Ratio (Slow/Rapid)','NumberTitle','off');
    bar(dataPR);
    set(gca, 'XTickLabel', files);
    legend(channels, 'Location','best');
    xlabel('File'); ylabel('Power Ratio (Slow/Rapid)');
    title('Power Ratio Across Files');
    saveas(fig3, fullfile(figFolder,'PowerRatio.png'));
    close(fig3);

    % Slow vs Rapid Scatter
    fig4 = figure('Name','Slow vs. Rapid Power','NumberTitle','off');
    gscatter(T.slowPower, T.rapidPower, T.fileName);
    xlabel('Slow Power'); ylabel('Rapid Power');
    title('Slow Power vs. Rapid Power by File');
    legend('Location','best');
    saveas(fig4, fullfile(figFolder,'SlowVsRapidScatter.png'));
    close(fig4);
end
