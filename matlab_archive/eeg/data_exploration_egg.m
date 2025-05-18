%% Iteration #1 apply an SVM on a raw signal 

%%
clc;
close all;
clear all;

%% load edf and xml files of several files
addpath("Scripts\")
edfFilename = 'data/R4.edf';
xmlFilename = 'data/R4.xml';
[hdr, record] = edfread(edfFilename);
[events, stages, epochLength,annotation] = readXML(xmlFilename);

%% Structure and preprocess the data

% Preprocessing 
% `record(8,:)` contains the EEG signal
fs = hdr.samples(8); % Sampling frequency from the EDF header
eeg_signal = record(8,:);

%% 1. Baseline Drift Removal (High-Pass Filter)
hp_cutoff = 0.5; % 0.5 Hz cutoff for drift removal
[b_hp, a_hp] = butter(2, hp_cutoff/(fs/2), 'high');
eeg_no_drift = filtfilt(b_hp, a_hp, eeg_signal);

%% 2. Muscle Noise Filtering (Band-Pass Filter)
bp_low = 0.5;   % Low cutoff to remove slow drifts
bp_high = 40;   % High cutoff to remove muscle artifacts
[b_bp, a_bp] = butter(4, [bp_low bp_high]/(fs/2), 'bandpass');
eeg_filtered = filtfilt(b_bp, a_bp, eeg_no_drift);

%% 3. Power Line Interference Removal (Notch Filter)
notch_freq = 60; % 50 or 60 Hz depending on region
Q = 35; % Quality factor - adjust to control notch sharpness
wo = notch_freq/(fs/2);
bw = wo/Q;
[b_notch, a_notch] = butter(2, [wo - bw, wo + bw], 'stop');

eeg_clean = filtfilt(b_notch, a_notch, eeg_filtered);


%% Plot the signals
t = (0:length(eeg_signal)-1)/fs;
figure;
subplot(4,1,1);
plot(t, eeg_signal);
title('Raw EEG Signal');

subplot(4,1,2);
plot(t, eeg_no_drift);
title('After Baseline Drift Removal');

subplot(4,1,3);
plot(t, eeg_filtered);
title('After Muscle Noise Filtering');

subplot(4,1,4);
plot(t, eeg_clean);
title('After Power Line Interference Removal');

xlabel('Time (s)');


%% Features extraction

% Start with basic features
fs = hdr.samples(8); % Sampling frequency
epoch_duration = 30; % 30 seconds per epoch
epoch_samples = fs * epoch_duration; % Samples per epoch

% EEG Signal
eeg_signal = eeg_clean; % Preprocessed signal from previous step
num_epochs = floor(length(eeg_signal) / epoch_samples);

disp('Number of epochs in the edf record');
num_epochs


% Initialize feature matrix
features = zeros(num_epochs, 8); % 4 time-domain + 4 frequency-domain features

% Feature Extraction Loop
for i = 1:num_epochs
    % Extract epoch
    start_idx = (i - 1) * epoch_samples + 1;
    end_idx = i * epoch_samples;
    epoch = eeg_signal(start_idx:end_idx);

    % Normalize epoch
    epoch = (epoch - mean(epoch)) / std(epoch);

    % Time-Domain Features
    features(i, 1) = mean(epoch);         % Mean
    features(i, 2) = var(epoch);          % Variance

    % Manual skewness
    mu = mean(epoch);
    sigma = std(epoch);
    features(i, 3) = mean((epoch - mu).^3) / (sigma^3); % Skewness

    % Manual kurtosis
    features(i, 4) = mean((epoch - mu).^4) / (sigma^4); % Kurtosis

    % Frequency-Domain Features
    Y = fft(epoch);
    P2 = abs(Y/epoch_samples);
    P1 = P2(1:epoch_samples/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = fs*(0:(epoch_samples/2))/epoch_samples;

    % Frequency bands
    delta = bandpower(epoch, fs, [0.5 4]);
    theta = bandpower(epoch, fs, [4 8]);
    alpha = bandpower(epoch, fs, [8 13]);
    beta = bandpower(epoch, fs, [13 30]);

    % Store frequency features
    features(i, 5) = delta;
    features(i, 6) = theta;
    features(i, 7) = alpha;
    features(i, 8) = beta;
end

% Save features
csvwrite('EEG_features.csv', features);

disp('Feature extraction completed and saved to EEG_features.csv');

%% Classification using a SVM

% Load features and corresponding labels
features = csvread('EEG_features.csv');
% Exploring the xml file
disp('Number of data point in the stages column');
length(stages)

disp('length_stages/epoch_duration(30s)');
length(stages)/30

disp('Each second is labelled with a sleeping score');

labels = stages(15:30:length(stages)); % We consider the label halfway through the 30s epochs (label(epoch(i)+15secondes)

if length(labels) > size(features, 1)
    labels = labels(1:size(features, 1));
elseif length(labels) < size(features, 1)
    features = features(1:length(labels), :);
end

for i = 1:length(labels)
    if labels(i) == 2
        labels(i) = 3;
    end
end

% Split dataset into train (80%) and test (20%)
n = size(features, 1);
idx = randperm(n);
train_size = round(0.8 * n);

X_train = features(idx(1:train_size), :);
y_train = labels(idx(1:train_size));
X_test = features(idx(train_size+1:end), :);
y_test = labels(idx(train_size+1:end));
y_test = y_test(:);

% Train SVM classifier using fitcecoc for multi-class classification
svm_model = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf', 'Standardize', true));

% Make predictions on the test set
y_pred = predict(svm_model, X_test);

% Evaluate the model
accuracy = mean(y_pred == y_test) * 100;  % Ensures a single scalar value
conf_matrix = confusionmat(y_test, y_pred);

% Display results
fprintf('Test Accuracy: %.2f%%\n', accuracy);
disp('Confusion Matrix:');
disp(conf_matrix);

% Visualize results with custom class labels
classLabels = {'REM','N3','N2','N1','Wake'};
figure;
confusionchart(conf_matrix, classLabels);
title('SVM Confusion Matrix for EEG Sleep Stage Classification');

%% Classification using a Random Forest

% Load features and corresponding labels
features = csvread('EEG_features.csv');
labels = stages(15:30:length(stages)); % We consider the label halfway through the 30s epochs (label(epoch(i)+15secondes)

if length(labels) > size(features, 1)
    labels = labels(1:size(features, 1));
elseif length(labels) < size(features, 1)
    features = features(1:length(labels), :);
end

for i = 1:length(labels)
    if labels(i) == 2
        labels(i) = 3;
    end
end

% Split dataset into train (80%) and test (20%)
n = size(features, 1);
idx = randperm(n);
train_size = round(0.8 * n);

X_train = features(idx(1:train_size), :);
y_train = labels(idx(1:train_size));
X_test = features(idx(train_size+1:end), :);
y_test = labels(idx(train_size+1:end));
y_test = y_test(:);

% Train Random Forest classifier
rf_model = TreeBagger(50, X_train, y_train, 'Method', 'classification');

% Make predictions on the test set
y_pred = predict(rf_model, X_test);
y_pred = str2double(y_pred);

% Evaluate the model
accuracy = mean(y_pred == y_test) * 100;  % Ensures a single scalar value
conf_matrix = confusionmat(y_test, y_pred);

% Display results
fprintf('Test Accuracy: %.2f%%\n', accuracy);
disp('Confusion Matrix:');
disp(conf_matrix);

% Visualize results with custom class labels
classLabels = {'REM','N3','N2','N1','Wake'};
figure;
confusionchart(conf_matrix, classLabels);
title('Random Forest Confusion Matrix for EEG Sleep Stage Classification');

%% Classification using a Deep Learning Model

% Load features and corresponding labels
features = csvread('EEG_features.csv');
labels = stages(15:30:length(stages)); % We consider the label halfway through the 30s epochs (label(epoch(i)+15secondes)

if length(labels) > size(features, 1)
    labels = labels(1:size(features, 1));
elseif length(labels) < size(features, 1)
    features = features(1:length(labels), :);
end

for i = 1:length(labels)
    if labels(i) == 2
        labels(i) = 3;
    end
end

% Split dataset into train (80%) and test (20%)
n = size(features, 1);
idx = randperm(n);
train_size = round(0.8 * n);

X_train = features(idx(1:train_size), :);
y_train = labels(idx(1:train_size));
X_test = features(idx(train_size+1:end), :);
y_test = labels(idx(train_size+1:end));
y_test = y_test(:);

% Convert labels to categorical
y_train = categorical(y_train);
y_test = categorical(y_test);

% Define the neural network architecture
layers = [
    featureInputLayer(8)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numel(unique(y_train)))
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', 'MaxEpochs', 30, 'MiniBatchSize', 32, 'ValidationData', {X_test, y_test}, 'Verbose', false, 'Plots', 'training-progress');

% Train the neural network
dl_model = trainNetwork(X_train, y_train, layers, options);

% Make predictions on the test set
y_pred = classify(dl_model, X_test);

% Evaluate the model
accuracy = mean(y_pred == y_test) * 100;  % Ensures a single scalar value
conf_matrix = confusionmat(y_test, y_pred);

% Display results
fprintf('Test Accuracy: %.2f%%\n', accuracy);
disp('Confusion Matrix:');
disp(conf_matrix);

% Visualize results with custom class labels
classLabels = {'REM','N3','N2','N1','Wake'};
figure;
confusionchart(conf_matrix, classLabels);
title('Deep Learning Confusion Matrix for EEG Sleep Stage Classification');
