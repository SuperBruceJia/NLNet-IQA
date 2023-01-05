% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear,clc;

format long

% SCID

data_path = '/home/shuyuej/Desktop/Dataset/SCID/MOS_SCID.txt';
data = readtable(data_path);
[rows, cols] = size(data);

%% dmos, im_names, ref_names
ref_names = data.Var1;
ref_names = union(ref_names, ref_names);

im_names = data.Var2;
dmos = data.Var3;

%% im_names
for i = 1: rows
    cur_data = cell2mat(im_names(i));
    im_names{i} = strcat(cur_data, '.bmp');
end


%% ref_ids
ref_ids = [];

for i = 1 : rows
    ref_id = cell2mat(im_names(i));
    ref_id = str2double(ref_id(4:5));
    ref_ids = [ref_ids; ref_id];
end

% index
index = cell2mat(arrayfun(@(i)randperm(length(ref_names)), 1:1000, 'UniformOutput', false)');

clear cols cur_data data data_path dmos i ref_id ref_names rows
save('SCIDinfo', '-v7.3')

% %% dmos, im_names, ref_names
% dmos = [];
% var1 = data.Var1;
% var2 = data.Var2;
% var3 = data.Var3;
% 
% for i = 1: rows
%     cur_data = cell2mat(var1(i));
%     find_space = find(cur_data == ' ');
%     
%     % dmos
%     cur_mos = str2double(cur_data(1, 1:find_space-1));
%     dmos = [dmos; cur_mos];
%     
%     % im_names
%     before = cur_data(1, find_space+1:end);
%     middle = num2str(var2(i));
%     after = cell2mat(var3(i));
%     im_names{i} = strcat(before, '_', middle, '_', after);
%     ref_names{i} = strcat(before, '.bmp');
% end
% im_names = im_names';
% ref_names = ref_names';
% ref_names = union(ref_names, ref_names);
% 
% %% ref_ids
% ref_ids = [];
% 
% for i = 1 : rows
%     ref_id = cell2mat(im_names(i));
%     
%     if i <= 441
%         ref_id = str2double(ref_id(4));
%     else
%         ref_id = str2double(ref_id(4:5));
%     end
%     
%     ref_ids = [ref_ids; ref_id];
% end
% 
% %% index
%  index = cell2mat(arrayfun(@(i)randperm(length(ref_names)), 1:1000, 'UniformOutput', false)');
%  
% % % subjective_scores
% subjective_scores = dmos;
% 
% clear before after middle cols cur_data cur_mos data data_path dmos find_space i ref_id ref_names var1 var2 var3 rows
% % save('SIQADinfo', '-v7.3')
