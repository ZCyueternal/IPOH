%% init the workspace
close all; clear; clc; warning off;

%% load dataset
train_param.ds_name='MIRFLICKR';
% train_param.N=16000;
train_param.normalizeX = 1;
train_param.kernel = 0;
train_param.unsupervised=0;
train_param.hbits=512;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_param.incre_paramiter=5;
train_param.mu_incre=10;
train_param.sita_incre=0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_param.plugin=1;
train_param.nus4w=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbits_set=[16 32 64 96 128];
for i=1:length(nbits_set)
    train_param.current_bits = nbits_set(i);
    [train_param,XTrain,LTrain,XQuery,LQuery,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate] = incre_load_dataset(train_param);
    [eva(i,:),t] = incre_evaluate_HMOH_test(train_param,XTrain,LTrain,XQuery,LQuery,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate);
end




