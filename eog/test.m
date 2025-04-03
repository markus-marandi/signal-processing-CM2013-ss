function comprehensive_sleep_scoring
% COMPREHENSIVE_SLEEP_SCORING
% Demonstrates:
%   1) Zero-epoch exclusion
%   2) Multi-signal placeholders (EEG, EOG, EMG, etc.)
%   3) Class balancing or weighting
%   4) Checking sleep-stage labels + improved hypnogram
%   5) Optional resampling for ICA if sampling rates differ
%
% Author: [Your Name]
% Date:   [Today's Date]

    %% 0) PATHS & SETUP
    clc; close all; clear;
    restoredefaultpath; rehash toolboxcache;

    projectRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(projectRoot, 'Scripts'), '-begin');
    rehash toolboxcache;

    dataFolder = fullfile(projectRoot, 'Data');
    figFolder  = fullfile(projectRoot, 'Figures');  % or your path
    if ~exist(figFolder,'dir'), mkdir(figFolder); end

    % Debug info
    disp('Using edfread from:'); which edfread -all
    disp('Using butter from:');  which butter -all

    %% 1) GATHER EDF FILES
    edfFiles = dir(fullfile(dataFolder,'R*.edf'));
    if isempty(edfFiles)
        error('No EDF files found in %s', dataFolder);
    end

    % Store epoch-level features + labels across ALL files
    allFeatures = [];
    allLabels   = [];
    labelText   = {};

    for fIdx = 1:numel(edfFiles)
        edfName = edfFiles(fIdx).name;
        edfPath = fullfile(dataFolder, edfName);
        [~, baseName] = fileparts(edfName);
        xmlPath = fullfile(dataFolder, [baseName '.xml']);

        fprintf('\n=== Processing %s ===\n', edfName);
        [hdr, record] = edfread(edfPath);

        % Attempt to load staging info
        if exist(xmlPath,'file')
            [events, stages, epochLen, annotation] = readXML(xmlPath);
        else
            warning('No XML for %s => no official labels.', edfName);
            stages = []; epochLen=30;
        end

        %% 1.1) Identify EOG + EEG + EMG, etc.
        eogIdx = find(contains(upper(hdr.label),'EOG'));
        eegIdx = find(contains(upper(hdr.label),'EEG'));
        emgIdx = find(contains(upper(hdr.label),'EMG'));
        % etc. for ECG, respiration, etc.

        if isempty(eogIdx)
            warning('No EOG found in %s => skipping.', edfName);
            continue;
        end

        % Check if sampling rates differ. If so, we can resample for ICA:
        fsList = hdr.samples([eogIdx(:); eegIdx(:); emgIdx(:)]); 
        sameFs = (numel(unique(fsList))==1);
        haveFastICA = (exist('fastica','file')==2);

        %% (Optional) Perform ICA with resampling (if you want)
        doICA = false;  % set true to demonstrate
        if doICA
            if ~sameFs
                % Example of resampling EOG + EEG to 200 Hz:
                desiredFs = 200;
                disp('Resampling EOG+EEG+EMG to do ICA...');
                % Example for EOG
                for iEOG = 1:numel(eogIdx)
                    oldFs = hdr.samples(eogIdx(iEOG));
                    raw = double(record(eogIdx(iEOG),:));
                    eogResampled = resample(raw, desiredFs, oldFs);
                    % store eogResampled somewhere
                end
                % Similarly for EEG, EMG, etc.
            end

            if haveFastICA
                fprintf('Running ICA...\n');
                % dataMat = [ eogResampled; eegResampled; emgResampled ];
                % [icasig,A,W] = fastica(dataMat, 'numOfIC',size(dataMat,1),...
                %    'verbose','off','displayMode','off');
            else
                disp('Skipping ICA => fastica() not found.');
            end
        else
            if ~sameFs
                disp('Warning: channels differ in Fs => skipping ICA entirely.');
            end
        end

        %% 2) EXTRACT EOG EPOCHS, SKIP ZERO EPOCHS
        eogChan = eogIdx(1);  % choose the first EOG channel or loop over both
        fsEOG = hdr.samples(eogChan);
        eogSignal = double(record(eogChan,:));

        totalSamps = numel(eogSignal);
        estEpochs  = floor(totalSamps/(30*fsEOG));
        if isempty(stages)
            stages = -1 * ones(estEpochs,1);  % no labels => -1
        end
        nEpochs = min(estEpochs, length(stages));

        for eIdx = 1:nEpochs
            sIdx = (eIdx-1)*30*fsEOG + 1;
            eIdx_ = eIdx*30*fsEOG;
            epochSig = eogSignal(sIdx : eIdx_);

            % SKIP zero epochs
            if all(epochSig==0)
                continue;  % do not train or test on these
            end

            % Filter + feature extraction
            epochEOG = eog_preprocessing_butter(epochSig, fsEOG, 0.5, 30);
            featsEOG = eog_feature_extraction(epochEOG, fsEOG, 100, 0.3);

            % (Placeholder) Also extract EEG + EMG features if you want:
            % featsEEG = ...
            % featsEMG = ...

            % Combine all signal features into one vector
            % e.g. [ EOG-blinkRate, EOG-powerRatio, EEG-deltaPower, ...]
            featVec = [ featsEOG.blinkRatePerMin, ...
                        featsEOG.movementDensityMean, ...
                        featsEOG.powerRatio ];
            % e.g. add featsEEG, featsEMG

            allFeatures = [allFeatures; featVec]; %#ok<AGROW>

            stageLabel = stages(eIdx);
            allLabels  = [allLabels; stageLabel]; %#ok<AGROW>
            labelText{end+1,1} = stage_code_to_text(stageLabel); %#ok<AGROW>
        end

        %% 2.1) (Optional) Plot the hypnogram if you want
        if fIdx==1
            figure('Name','Hypnogram','NumberTitle','off');
            % Each 'stage' is for a 30s epoch => time in minutes:
            tAxis = ((1:length(stages))*epochLen)/60;  
            plot(tAxis, stages, 'LineWidth',1);
            xlim([0, tAxis(end)]);
            xlabel('Time (Minutes)');
            ylabel('Sleep Stage');
            set(gca, 'YTick', [0 1 2 3 4 5], ...
                'YTickLabel', {'Wake','N1','N2','N3','(4?)','REM'} );
            title('Hypnogram');
            box off; grid on;
            set(gcf,'color','w');
            saveas(gcf, fullfile(figFolder,'HypnogramExample.png'));
            close(gcf);
        end
    end  % end file loop

    %% 3) CLASSIFICATION
    if isempty(allFeatures)
        warning('No features => cannot classify. Exiting...');
        return;
    end

    Ynum = allLabels;  % numeric code for stages
    Ycat = categorical(Ynum);
    if numel(unique(Ycat))<2
        warning('Only 1 class => no classification possible');
        return;
    end

    X = allFeatures;
    N = size(X,1);

    % 3.1) BALANCING / WEIGHTING
    doBalance = true;
    if doBalance
        % For Random Forest, we can do 'Prior','uniform'
        rfParams = {'OOBPrediction','off','Prior','uniform'};
        % For SVM, we can define a cost matrix or use 'Weights'
        % Example for 4-class: costMatrix = [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0];
        % We'll do a simpler approach: no cost => reliant on 'fitcecoc' defaults
        costMatrix = []; % or define your cost
    else
        rfParams = {'OOBPrediction','off'};
        costMatrix = [];
    end

    % 3.2) Cross-Validation
    KF = 5;
    cvo = cvpartition(Ycat,'KFold',KF);
    YhatRF  = cell(N,1);
    YhatSVM = cell(N,1);

    for fold=1:KF
        trIdx = training(cvo, fold);
        teIdx = test(cvo, fold);

        Xtrain = X(trIdx,:);
        Ytrain = Ycat(trIdx);
        Xtest  = X(teIdx,:);
        Ytest  = Ycat(teIdx);

        % --- Random Forest with uniform prior
        RFmodel = TreeBagger(50, Xtrain, Ytrain, rfParams{:});
        predRF  = predict(RFmodel, Xtest);
        YhatRF(teIdx) = predRF;

        % --- SVM with optional cost
        tpl = templateSVM('KernelFunction','rbf','Standardize',true);
        if ~isempty(costMatrix)
            SVMmodel = fitcecoc(Xtrain, Ytrain, 'Learners', tpl,...
                'Coding','onevsone','ClassNames',categories(Ytrain),...
                'Cost', costMatrix);
        else
            SVMmodel = fitcecoc(Xtrain, Ytrain, 'Learners', tpl,...
                'Coding','onevsone','ClassNames',categories(Ytrain));
        end
        predSVM = predict(SVMmodel, Xtest);
        YhatSVM(teIdx) = predSVM;
    end

    YhatRF  = categorical(YhatRF);
    YhatSVM = categorical(YhatSVM);
    accRF  = mean(YhatRF==Ycat);
    accSVM = mean(YhatSVM==Ycat);

    cmRF  = confusionmat(Ycat, YhatRF);
    cmSVM = confusionmat(Ycat, YhatSVM);

    fprintf('\n=== Classification Results (EOG-based, Zero-Epoch Excluded) ===\n');
    fprintf('Random Forest Accuracy: %.2f%%\n', accRF*100);
    fprintf('SVM Accuracy:          %.2f%%\n', accSVM*100);

    disp('Confusion Matrix (RF):');   disp(cmRF);
    disp('Confusion Matrix (SVM):');  disp(cmSVM);

    figure('Name','Confusion Matrices','NumberTitle','off');
    subplot(1,2,1);
    imagesc(cmRF); colorbar;
    title(sprintf('RF (Acc=%.2f%%)',accRF*100));
    xlabel('Predicted'); ylabel('True');

    subplot(1,2,2);
    imagesc(cmSVM); colorbar;
    title(sprintf('SVM (Acc=%.2f%%)',accSVM*100));
    xlabel('Predicted'); ylabel('True');
    set(gcf,'color','w');

    saveas(gcf, fullfile(figFolder, 'ConfMatrices.png'));
    close(gcf);

end % end main function

%% ------------------------------------------------------------------------
% HELPER FUNCTIONS
%% ------------------------------------------------------------------------
function processedEOG = eog_preprocessing_butter(eogSignal, fs, hpCutoff, lpCutoff)
    nyq = fs/2;
    hpCutoff = max(0.01, min(hpCutoff, nyq-0.01));
    lpCutoff = max(0.5,  min(lpCutoff, nyq-0.01));

    [b_hp,a_hp] = butter(2, hpCutoff/nyq, 'high');
    sigHP = filtfilt(b_hp,a_hp, double(eogSignal));
    [b_lp,a_lp] = butter(4, lpCutoff/nyq, 'low');
    processedEOG = filtfilt(b_lp,a_lp, sigHP);
end

function feats = eog_feature_extraction(eogSignal, fs, blinkThreshold, minPeakDistSec)
    if nargin<3 || isempty(blinkThreshold), blinkThreshold=[]; end
    if nargin<4 || isempty(minPeakDistSec), minPeakDistSec=0.3; end

    feats = struct();
    minPD = round(fs*minPeakDistSec);

    if isempty(blinkThreshold)
        factor    = 3;
        sigStd    = std(eogSignal);
        sigMean   = mean(abs(eogSignal));
        blinkThreshold = max(20, sigMean + factor*sigStd);
    end

    if max(abs(eogSignal)) < blinkThreshold
        blinkIdx = [];
    else
        [~,blinkIdx] = findpeaks(abs(eogSignal),...
            'MinPeakHeight', blinkThreshold,...
            'MinPeakDistance', minPD);
    end
    feats.numBlinks = length(blinkIdx);
    feats.blinkRatePerMin = feats.numBlinks / (length(eogSignal)/fs/60);

    % Movement density in 5s windows
    wSec   = 5;
    wSamps = wSec*fs;
    nWin   = floor(length(eogSignal)/wSamps);
    mdVals = zeros(nWin,1);
    for w=1:nWin
        seg = eogSignal((w-1)*wSamps+1 : w*wSamps);
        mdVals(w) = std(seg);
    end
    feats.movementDensityMean = mean(mdVals);

    % Slow vs. rapid power ratio
    [pxx,f] = pwelch(eogSignal, 2*fs, [], [], fs);
    slowIdx  = (f>=0.5 & f<3);
    rapidIdx = (f>=3   & f<=30);
    sPow = trapz(f(slowIdx),   pxx(slowIdx));
    rPow = trapz(f(rapidIdx),  pxx(rapidIdx));
    if rPow>0
        feats.powerRatio = sPow / rPow;
    else
        feats.powerRatio = Inf;
    end
end

function txt = stage_code_to_text(numCode)
    % typical: 0=Wake,1=N1,2=N2,3=N3,5=REM
    switch numCode
        case 0, txt='Wake';
        case 1, txt='N1';
        case 2, txt='N2';
        case 3, txt='N3'; % or combine N3+4
        case 4, txt='N3'; % some have 4
        case 5, txt='REM';
        otherwise, txt='Unknown';
    end
end
