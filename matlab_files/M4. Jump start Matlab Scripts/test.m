%%
clc;
close all;
clear all;

%% load edf and xml files
addpath("Scripts/")
edfFilename = 'data/R4.edf';
xmlFilename = 'data/R4.xml';
[hdr, record] = edfread(edfFilename);
[events, stages, epochLength,annotation] = readXML(xmlFilename);
%%
numberOfEpochs = length(record(3,:)')/(30*hdr.samples(3))

%% plot 1 30 sec epoch of each signal
figure(1);
for i=1:size(record,1)
    Fs = hdr.samples(i);
    (length(record(i,:)')/Fs);
    epochNumber = 1; % plot nth epoch of 30 seconds
    epochStart = (epochNumber*Fs*30);
    epochEnd = epochStart + 30*Fs;
    subplot(size(record,1),1,i);
    signal = record(i,epochStart:epochEnd);
    plot((1:length(signal))/Fs,signal);
    ylabel(hdr.label(i));
    xlim([1 30]);
end
sgtitle(['30 seconds epoch #' num2str(epochNumber)]);
set(gcf,'color','w');


%% plot Hypnogram (sleep stages over time)
figure(2);
plot(((1:length(stages))*30)./60,stages); %sleep stages are for 30 seconds epochs
ylim([0 6]);
set(gca,'ytick',[0:6],'yticklabel',{'REM','','N3','N2','N1','Wake',''});
xlabel('Time (Minutes)');
ylabel('Sleep Stage');
box off;
title('Hypnogram');
set(gcf,'color','w');


%% Compute and Plot Correlation Matrix for a Specific Epoch
% Choose the epoch number to analyze (e.g., epoch 1)
epochNumber = 1;

% Preallocate matrix to hold the epoch signals (each row corresponds to a channel)
numChannels = size(record,1);
signalsEpoch = zeros(numChannels, 30*min(hdr.samples)); % use minimum sampling rate to be safe

for i = 1:numChannels
    Fs = hdr.samples(i);
    % Calculate start and end indices for the chosen epoch
    epochStart = (epochNumber-1)*30*Fs + 1;
    epochEnd   = epochStart + 30*Fs - 1;
    
    % In case different channels have different lengths, use min(length, epochEnd)
    if epochEnd > length(record(i,:))
        error('Epoch %d exceeds available data in channel %d', epochNumber, i);
    end
    
    % Extract the 30-sec epoch for the channel
    signalsEpoch(i,1:30*Fs) = record(i, epochStart:epochEnd);
end

% Since channels may have different sampling rates, it's common to 
% resample or work with a common time vector. Here, we assume channels 
% are roughly aligned in time and use their extracted segments as-is.
% For the correlation, we compute across channels.
% If channels have different numbers of samples, you might need to
% resample them to a common sampling rate first.

% Compute the correlation matrix. Each entry (i,j) is the correlation
% coefficient between channel i and channel j.
% Since each row of signalsEpoch is a signal from a channel,
% we compute correlation across columns by transposing.
corrMatrix = corrcoef(signalsEpoch');

% Plot the correlation matrix
figure(3);
imagesc(corrMatrix);
colorbar;
title(['Correlation Matrix for 30-second Epoch #' num2str(epochNumber)]);
xlabel('Channels');
ylabel('Channels');

% Label ticks with channel names from hdr.label (if available)
set(gca,'XTick',1:numChannels, 'XTickLabel', hdr.label, ...
        'YTick',1:numChannels, 'YTickLabel', hdr.label);
axis square;