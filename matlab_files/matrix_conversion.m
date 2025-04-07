%% ---------------------------
%  Batch EDF Loading and Raw Data Conversion
%
%  This script:
%   - Restores the default MATLAB path and adds your custom Scripts folder
%     (which contains edfread.m) to the beginning of the path.
%   - Locates all EDF files (matching "R*.edf") in the Data folder.
%   - Loads the raw data and header from each file.
%   - Saves the data into a MAT file:
%       • "allData" is a structure array with fields: fileName, hdr, record.
%       • If possible, the raw data are also stacked into a 3D matrix
%         (channels x samples x files) as "bigMatrix".
%
%  Note: This script does not perform any filtering or channel-specific processing.
%% ---------------------------

clc;
close all;
clear all;

%% --- STEP 0: CLEAN PATH & ADD CUSTOM SCRIPTS ---
restoredefaultpath;
rehash toolboxcache;

% Determine the project root (the folder containing this .m file)
projectRoot = fileparts(mfilename('fullpath'));

% Add your custom Scripts folder (with edfread.m) to the beginning of the path
addpath(fullfile(projectRoot, 'Scripts'), '-begin');
rehash toolboxcache;

% (Optional) Display which versions of edfread and butter are used
disp('Using edfread from:');
which edfread -all
disp('Using butter from:');
which butter -all

%% --- STEP 1: SETUP & FIND EDF FILES ---
% Set the correct Data folder path
dataFolder = '/Users/markus/university/signal-processing-CM2013-ss/M4. Jump start Matlab Scripts/Data';
edfFiles = dir(fullfile(dataFolder, 'R*.edf'));

if isempty(edfFiles)
    error('No EDF files matching "R*.edf" found in folder: %s', dataFolder);
end

%% --- STEP 2: LOAD ALL EDF FILES ---
% Initialize a structure array to store raw data and header information
allData = struct('fileName', {}, 'hdr', {}, 'record', {});

for fIdx = 1:length(edfFiles)
    edfName = edfFiles(fIdx).name;
    edfPath = fullfile(dataFolder, edfName);
    
    fprintf('Processing file: %s\n', edfName);
    
    % Load the EDF file using custom edfread; no filtering or processing is done.
    [hdr, record] = edfread(edfPath);
    
    % Save the file name, header info, and raw data (record)
    allData(end+1).fileName = edfName; %#ok<SAGROW>
    allData(end).hdr = hdr;
    allData(end).record = record;
end

%% --- STEP 3: OPTIONAL 3D MATRIX STACKING ---
% Check if all EDF files have the same dimensions (channels and samples)
allSame = true;
nChannels = [];
nSamples = [];
for i = 1:length(allData)
    [c, s] = size(allData(i).record);
    if isempty(nChannels)
        nChannels = c;
        nSamples = s;
    else
        if c ~= nChannels || s ~= nSamples
            allSame = false;
            break;
        end
    end
end

% If dimensions match, stack the raw data into a 3D matrix.
if allSame
    % Dimensions: channels x samples x files
    bigMatrix = zeros(nChannels, nSamples, length(allData));
    for i = 1:length(allData)
        bigMatrix(:, :, i) = allData(i).record;
    end
    % Save both the structure array and the 3D matrix using MAT-file version 7.3
    save(fullfile(projectRoot, 'EDF_RawData.mat'), 'allData', 'bigMatrix', '-v7.3');
    fprintf('All raw data saved to EDF_RawData.mat with a 3D matrix (channels x samples x files).\n');
else
    % If dimensions differ, only save the structure array.
    save(fullfile(projectRoot, 'EDF_RawData.mat'), 'allData', '-v7.3');
    fprintf('Raw data saved to EDF_RawData.mat as a structure array (files with varying dimensions).\n');
end
