% converts tidigits mat files to wav files
clc
clear all
close all

data = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/tidigits_test_mfccs.mat';
file_name = 'test.h5';

load(data)
samples = length(mfccs_third);
labels = [];
for i=1:1:samples
    curr_digs = digs{i}(1:end-1);
    
    for ii=1:1:length(curr_digs)
        curr_dig = curr_digs(ii);
        if strcmp(curr_dig, 'O')==1
            curr_labels(ii)= 10;
        elseif strcmp(curr_dig, 'Z')==1
            curr_labels(ii)= 11;
        else
            curr_labels(ii) = str2num(curr_dig);
        end
    end
    
    labels = horzcat(labels, curr_labels);   
    
    label_lens(i) = length(curr_digs);
    keys(i) = i;
end

%% Create hdf5
if exist(file_name, 'file')==2
  delete(file_name);
end
h5create(file_name, '/features', [39 Inf], 'ChunkSize', [39 8192], 'Datatype', 'single');
h5create(file_name, '/feature_lens', samples)
h5create(file_name, '/labels', length(labels));
h5create(file_name, '/label_lens', samples);
h5create(file_name, '/keys', samples);

h5write(file_name, '/labels', labels);
h5write(file_name, '/label_lens', label_lens);
h5write(file_name, '/keys', keys);

start=1;
for i=1:1:samples
    logf_mat=mfccs_third{i}';
    h5write(file_name, '/features', logf_mat, [1 start], size(logf_mat)) % write features
    h5write(file_name, '/feature_lens', length(logf_mat(1,:)), i,1) % write lenghts of samples
    % update hdf5 indices for features
    start = start+length(logf_mat(1,:));
end