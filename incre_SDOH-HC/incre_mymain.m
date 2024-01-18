clear;
addpath(genpath('./'));

nbits_set=[16 32 64 96 128];
% nbits_set=[16 32 64];
%% load dataset
fprintf('loading dataset...\n')

%% load dataset
param.ds_name='MIRFLICKR'; 

param.load_type='second_setting';

param.nus4w = 1;
[param,XTrain,LTrain,XQuery,LQuery,anchor,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate] = load_dataset(param);


%% initialization
fprintf('initializing...\n')
set = param.ds_name;
trainset_size = param.trainset_size;
label_size = param.label_size;


if strcmp(set,'MIRFLICKR')
% MIR
    param.alpha = 1; param.gama = param.alpha;
    
    param.beta = 1;
    param.delta = 1;
    param.sita = 10;
    param.yita = 1;
    param.epsilon = 10;


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.mu_incre = 10;
param.sita_incre = 0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.datasets = set;

param.paramiter = 10;
param.incre_paramiter = 5;
if strcmp(set,'MIRFLICKR')
    param.nq = 200; 
    param.n1 = 100;
    param.chunk = 2000;
    param.nmax = 500;  % 1000 for SDOH-HC
end

%% loading 


%% model training
for bit=1:length(nbits_set)
    nbits=nbits_set(bit);
    Binit = sign(randn(trainset_size, nbits));
    Vinit = randn(trainset_size, nbits);
    Pinit = randn(1000, nbits);
    Sinit = zeros(label_size,label_size)-1;
    param.nbits=nbits;
    
    % randomly generate Teacher codebook
    if strcmp(param.datasets,'MIRFLICKR')
        load('../data/MIRFLICKR.mat');
        h = hadamard(512); % 404tags/ 24label
        h = h(randperm(size(L_tr,2)),randperm(nbits)); % 404*nbits
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param.plugin = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ MAP(bit,:),training_time(bit,:)] = incre_train_twostep(XTrain,LTrain,XQuery,LQuery,param,anchor,Binit,Vinit,Pinit,Sinit,h,seperate);

end 


