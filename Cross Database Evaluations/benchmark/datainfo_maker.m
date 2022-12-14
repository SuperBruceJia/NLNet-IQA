% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear,clc;

%% CSIQ

data_path = '/home/shuyuej/Desktop/Dataset/CSIQ/csiq.txt';
data = readtable(data_path);

names = data.image;
dst_type = data.dst_type;
dst_level = data.dst_lev;
dmos = data.dmos;
dmos_std = data.dmos_std;

% im_names
[rows, cols] = size(data);
im_names = cell(rows, 1);

for i = 1 : rows
    if strcmp(dst_type{i}, 'noise')
        temp_name =strcat( 'awgn/', names{i}, '.', 'AWGN', '.', num2str(dst_level(i)), '.png');
        
    elseif strcmp(dst_type{i}, 'jpeg')
        temp_name = strcat('jpeg/', names{i}, '.', 'JPEG', '.', num2str(dst_level(i)), '.png');
        
    elseif strcmp(dst_type{i}, 'jpeg 2000')
        temp_name = strcat('jpeg2000/', names{i}, '.', 'jpeg2000', '.', num2str(dst_level(i)), '.png');
        
    elseif strcmp(dst_type{i}, 'fnoise')
        temp_name = strcat('fnoise/', names{i}, '.', 'fnoise', '.', num2str(dst_level(i)), '.png');
    
    elseif strcmp(dst_type{i}, 'blur')
        temp_name = strcat('blur/', names{i}, '.', 'BLUR', '.', num2str(dst_level(i)), '.png');
    end
    
    im_names{i} = temp_name;
end

% ref_ids
ref_ids = [];
id = 1.0;
temp = names{1};
for i = 1 : rows
    if strcmp(temp, names{i})
        ref_ids = [ref_ids, id];
    else
        id = id + 1;
        ref_ids = [ref_ids, id];
        temp = names{i};
    end
end
ref_ids = ref_ids';

% ref_names
ref_names = union(names, names);

for i = 1 : 30
    ref_names{i} =strcat(ref_names{i}, '.png');
end

% subjective_scores
subjective_scores = dmos;

% subjective_scoresSTD
subjective_scoresSTD = dmos_std;

% index: `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(ref_names)), 1:1000, 'UniformOutput', false)'); 

clear cols data data_path dmos dmos_std dst_level dst_type i id names rows temp temp_name
save('CSIQfullinfo', '-v7.3')

















% data_path = '/home/shuyuej/Desktop/Dataset/CSIQ/csiq.DMOS.xlsx';
% data = xlsread(data_path);

% video_names = data.file_name; % video names
% scores = data.MOS; % subjective scores
% clear data_path data
% 
% height = 540; % video height
% width = 960; % video width
% max_len = 240; % maximum video length in the dataset
% video_format = 'RGB'; % video format
% ref_ids = [1:length(scores)]'; % video content ids

% % `random` train-val-test split index, 1000 runs
% index = cell2mat(arrayfun(@(i)randperm(length(scores)), 1:1000,'UniformOutput', false)'); 

% save('KoNViD-1kinfo','-v7.3')
























% %% KoNViD-1k
% data_path = '/media/ldq/Research/Data/KoNViD-1k/KoNViD_1k_attributes.csv';
% data = readtable(data_path);
% video_names = data.file_name; % video names
% scores = data.MOS; % subjective scores
% clear data_path data
% 
% height = 540; % video height
% width = 960; % video width
% max_len = 240; % maximum video length in the dataset
% video_format = 'RGB'; % video format
% ref_ids = [1:length(scores)]'; % video content ids
% % `random` train-val-test split index, 1000 runs
% index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
%     1:1000,'UniformOutput', false)'); 
% save('KoNViD-1kinfo','-v7.3')
% 
% %% CVD2014
% data_path = '/media/ldq/Research/Data/CVD2014/CVD2014_ratings/Realignment_MOS.csv';
% data = readtable(data_path);
% video_names = arrayfun(@(i) ['Test' data.File_name{i}(6) '/' ...
%     data.Content{i} '/' data.File_name{i} '.avi'], 1:234, ...
%     'UniformOutput', false)';  % video names, remove '', add dir
% scores = arrayfun(@(i) str2double(data.RealignmentMOS{i})/100, 1:234)'; % subjective scores
% clear data_path data
% 
% height = [720 480];
% width = [1280 640];
% max_len = 830;
% video_format = 'RGB';
% ref_ids = [1:length(scores)]';
% % `random` train-val-test split index, 1000 runs
% index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
%     1:1000,'UniformOutput', false)'); 
% save('CVD2014info','-v7.3')
% % LIVE-Qualcomm
% data_path = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/qualcommSubjectiveData.mat';
% data = load(data_path);
% scores = data.qualcommSubjectiveData.unBiasedMOS; % subjective scores
% video_names = data.qualcommVideoData;
% video_names = arrayfun(@(i) [video_names.distortionNames{video_names.distortionType(i)} ...
%     '/' video_names.vidNames{i}], 1:length(scores), ...
%     'UniformOutput', false)'; % video names
% clear data_path data
% 
% height = 1080;
% width = 1920;
% max_len = 526;
% video_format = 'YUV420';
% ref_ids = [1:length(scores)]';
% % `random` train-val-test split index, 1000 runs
% index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
%     1:1000,'UniformOutput', false)'); 
% save('LIVE-Qualcomminfo','-v7.3')