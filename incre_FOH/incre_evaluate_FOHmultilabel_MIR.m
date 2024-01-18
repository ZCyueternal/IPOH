clear;
param.unsupervised = 0;
param.normalizeX = 1;
param.K = 500;
param.numq = 200;

set = 'MIRFlickr';

fprintf('loading dataset...\n');
tic;
if strcmp(set,'MIRFlickr')
%     load('../../Datasets/supervised/MIRFLICKR.mat');
    load('../data/MIRFlickr_preprocessing_for_FOHmultilabel.mat');
end

%% preprocessing
% I_tr: 18015*512 
% I_te: 2000 *512
% L_tr: 18015*512
% L_te: 2000 *512

% fprintf('Normalizing features...\n');
% % normalize features
% if param.normalizeX
%     I_tr = bsxfun(@minus, I_tr, mean(I_tr,1));  % first center at 0
%     I_tr = normalize(double(I_tr));  % then scale to unit length
%     I_te = bsxfun(@minus, I_te, mean(I_te,1));  % first center at 0
%     I_te = normalize(double(I_te));  % then scale to unit length
% end
% 
% % mapped into a sphere space
% test = I_te ./ sqrt(sum(I_te .* I_te, 2));  
% testLabel = L_te;  % n x c
% train = I_tr ./ sqrt(sum(I_tr .* I_tr, 2));   
% trainLabel = L_tr; % n x c
% 
% test = test';   %d x n
% train = train'; 
% testLabel = testLabel';  %c x n   
% trainLabel = trainLabel';
% 
% save('MIRFlickr_preprocessing_for_FOHmultilabel.mat','train','test','trainLabel','testLabel');
load('../data/S.mat');
loadingTime = toc;
fprintf('Successfully loaded %s dataset!!!\n', set);
fprintf('loading dataset costs %f seconds \n',loadingTime);

%% get dimention
dX = size(train,1); % 512
param.dX = dX;
dL = size(trainLabel,1); % 24
param.dL = dL;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = [train';test'];
% L = [trainLabel';testLabel'];
param.ds_name = "MIRFLICKR";
[param,XTrain,LTrain,XQuery,LQuery,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate] = load_dataset(param,train',test',trainLabel',testLabel');

%%%%%%%%%%%%  parameters depicted in the paper %%%%%%%%%%%%%%%%
param.lambda = 0.5;   
param.sigma = 0.8;    
param.etad = 0.11;     
param.etas = 1;    
param.eta = 0.1;
param.theta = 1.5;
param.mu = 0.5;
param.tau = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.incre_paramiter = 5;

param.mu_incre = 1;
param.sita_incre = 1;
param.plugin = 1;

%% Trainning Start
nbits_set=[16 32 48 64 128];
% 
for bit=1:length(nbits_set)
    param.nbits = nbits_set(bit);

    W_t = randn(dX, param.nbits);
    W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', dX, 1);   % 512 * 16

    P_t = randn(param.nbits, dL);   %k x c
    P_t = P_t ./ repmat(diag(sqrt(P_t' * P_t))', param.nbits, 1);


    knei = [];
    kneil = [];
    % s: streaming
    Xs_t = [];    Bs_t = [];    ls_t = [];    vs_t = [];
    % e: existing
    Be_t = [];    Xe_t = [];    le_t = [];    ve_t = [];

    S_t = [];
    now_L = [];

    h = hadamard(512); % mir
    h = h(randperm(size(trainLabel,1)),randperm(param.nbits)); % 5000*nbits
    
    
    for chunki=1:param.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        
        train_t = XTrain{chunki,:}; %  n_t x 512   (need ' later)
        trainLabel_t = LTrain{chunki,:}; % n_t x 24 (need ' later)
        
        test_t = XQuery{chunki,:}; % n_t x 24 (need ' later)
        testLabel_t = LQuery{chunki,:}; % n_t x 24 (need ' later)
        
        param.current_index_start = size(cell2mat(XTrain(1:chunki-1)),1)+1; % 588+1=589
        
        tic;
        
        if chunki == 1
            [Xe_t,Be_t,ve_t,q,qlabel,now_X,now_B,now_L,tmp_W,dex,LTrain_only_incre,BB] = incre_train_FOHmulti0(param,chunki,train_t',trainLabel_t',W_t,seperate);
        else
            [Xe_t,Be_t,le_t,ve_t,Xs_t,now_X,Bs_t,ls_t,vs_t,now_L,W_t,q,qlabel,now_B,tmp_W,S,LTrain_only_incre,BB] = incre_train_FOHmulti(S,chunki,tmp_W,q,qlabel,param,Xe_t,Xs_t,Be_t,Bs_t,le_t,ls_t,ve_t,vs_t,train_t',now_X,W_t,trainLabel_t',now_L,P_t,h,seperate,LTrain_only_incre,BB);
        end
        
        
        
        
        training_time(bit,chunki) = toc;
        fprintf('Training time: %f second\n',training_time(bit,chunki));
        fprintf('test beginning\n');

        W_t = tmp_W ./ 9;
        tic;

        Htest = single(W_t' * test_t' >= 0); % r x ***
        Hq = single(W_t' * q >= 0);

        edex = knnsearch(Hq',Htest','K',10,'Distance','hamming');
        eud = unique(edex(:));
        prenei = dex(eud,:);
        up = unique(prenei(:));
        now_BB = now_B(:,up); % r x ***
        now_L = now_L(:,up);

        Aff = affinity([], [], now_L', testLabel_t, param);
        param.metric = 'mAP';

        res = cal_precision_multi_label_batch(now_BB', Htest', now_L', testLabel_t);
        Testing_time(bit,chunki) = toc;
        fprintf('Testing time: %f second\n',Testing_time(bit,chunki));
        fprintf('[%s][%d bit][chunk %d]\n',param.ds_name,param.nbits,chunki);
        MAP(bit,chunki) = res;
        logInfo(['mAP = ' num2str(res)]);
        
        
    end
    

end


