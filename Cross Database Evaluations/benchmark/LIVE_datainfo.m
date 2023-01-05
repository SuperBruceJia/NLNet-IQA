% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear all;
clc;

%% LIVE

data_path = '/home/shuyuej/Desktop/Dataset/live/final.csv';
data = readtable(data_path);

names = data.Var1;
im_names = data.Var2;

realigned_dmos = load('/home/shuyuej/Desktop/Dataset/live/dmos_realigned.mat');
dmos = realigned_dmos.dmos_new;
dmos = dmos';
dmos_std = realigned_dmos.dmos_std;
dmos_std = dmos_std';

% ref_names
ref_names = union(names, names);

% subjective_scores
subjective_scores = dmos;

% subjective_scoresSTD
subjective_scoresSTD = dmos_std;

% ref_ids
ref_ids = [];
for i = 1 : length(names)
    if strcmp(names{i}, 'bikes.bmp')
        ref_ids = [ref_ids, 1];
    elseif strcmp(names{i}, 'building2.bmp')
        ref_ids = [ref_ids, 2];
    elseif strcmp(names{i}, 'buildings.bmp')
        ref_ids = [ref_ids, 3];
    elseif strcmp(names{i}, 'caps.bmp')
        ref_ids = [ref_ids, 4];
    elseif strcmp(names{i}, 'carnivaldolls.bmp')
        ref_ids = [ref_ids, 5];
    elseif strcmp(names{i}, 'cemetry.bmp')
        ref_ids = [ref_ids, 6];
    elseif strcmp(names{i}, 'churchandcapitol.bmp')
        ref_ids = [ref_ids, 7];
    elseif strcmp(names{i}, 'coinsinfountain.bmp')
        ref_ids = [ref_ids, 8];
    elseif strcmp(names{i}, 'dancers.bmp')
        ref_ids = [ref_ids, 9];
    elseif strcmp(names{i}, 'flowersonih35.bmp')
        ref_ids = [ref_ids, 10];
    elseif strcmp(names{i}, 'house.bmp')
        ref_ids = [ref_ids, 11];
    elseif strcmp(names{i}, 'lighthouse.bmp')
        ref_ids = [ref_ids, 12];
    elseif strcmp(names{i}, 'lighthouse2.bmp')
        ref_ids = [ref_ids, 13];
    elseif strcmp(names{i}, 'manfishing.bmp')
        ref_ids = [ref_ids, 14];
    elseif strcmp(names{i}, 'monarch.bmp')
        ref_ids = [ref_ids, 15];
    elseif strcmp(names{i}, 'ocean.bmp')
        ref_ids = [ref_ids, 16];
    elseif strcmp(names{i}, 'paintedhouse.bmp')
        ref_ids = [ref_ids, 17];
    elseif strcmp(names{i}, 'parrots.bmp')
        ref_ids = [ref_ids, 18];
    elseif strcmp(names{i}, 'plane.bmp')
        ref_ids = [ref_ids, 19];
    elseif strcmp(names{i}, 'rapids.bmp')
        ref_ids = [ref_ids, 20];
    elseif strcmp(names{i}, 'sailing1.bmp')
        ref_ids = [ref_ids, 21];
    elseif strcmp(names{i}, 'sailing2.bmp')
        ref_ids = [ref_ids, 22];
    elseif strcmp(names{i}, 'sailing3.bmp')
        ref_ids = [ref_ids, 23];
    elseif strcmp(names{i}, 'sailing4.bmp')
        ref_ids = [ref_ids, 24];
    elseif strcmp(names{i}, 'statue.bmp')
        ref_ids = [ref_ids, 25];
    elseif strcmp(names{i}, 'stream.bmp')
        ref_ids = [ref_ids, 26];
    elseif strcmp(names{i}, 'studentsculpture.bmp')
        ref_ids = [ref_ids, 27];
    elseif strcmp(names{i}, 'woman.bmp')
        ref_ids = [ref_ids, 28];
    elseif strcmp(names{i}, 'womanhat.bmp')
        ref_ids = [ref_ids, 29];
    end
end
ref_ids = ref_ids';

% index: `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(ref_names)), 1:1000, 'UniformOutput', false)'); 

%%
clear data data_path i names dmos realigned_dmos orgs dmos_new dmos_std
save('LIVE982info', '-v7.3')
