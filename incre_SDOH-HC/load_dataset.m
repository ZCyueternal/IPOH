function [train_param,XTrain,LTrain,XQuery,LQuery,anchor,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate] = load_dataset(train_param)
    fprintf(['-------load dataset------', '\n']);
%     load(['datasets/',train_param.ds_name,'_deep.mat']);
%     load(['datasets/',train_param.ds_name,'_Groundtrue_Vec.mat']);
    
    if strcmp(train_param.ds_name, 'MIRFLICKR')
        
        load('../data/MIRFLICKR.mat');
        train_param.image_feature_size = 512;  % 4096
%         train_param.text_feature_size = 1386;
        train_param.trainset_size = size(I_tr, 1);
        train_param.label_size = size(L_tr, 2);
        
        if strcmp(train_param.load_type, 'second_setting')
            X = [I_tr; I_te];
            L = [L_tr; L_te];
            
            anchor=I_tr(randsample(2000,1000),:); %% random select 1000 sample from XTrain (1000*4096)
            
            [~,L_idx]=sort(sum(L),'descend');  % the quantity of different labels, and sort from to high to low 
            L=L(:,L_idx); % change orders of labels
            
            train_param.nchunks=10;
            
            labels=linspace(1,24,24);
            seperate=cell(train_param.nchunks,1);
            seperate{1,1}=[1];
            seperate{2,1}=[2,3,4];
            seperate{3,1}=[5,6];
            seperate{4,1}=[7,8];
            seperate{5,1}=[9,10];
            seperate{6,1}=[11,12];
            seperate{7,1}=[13,14];
            seperate{8,1}=[15,16,17];
            seperate{9,1}=[18,19,20];
            seperate{10,1}=[21,22,23,24];
            

            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_only_incre_labels = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_only_incre_labels = cell(train_param.nchunks,1);

            
            label_allow=[];
            last_found_idx=[];
            
            for l=1:train_param.nchunks
                label_allow=[seperate{l,1} label_allow];
                label_notallow=setdiff(labels,label_allow);
                idx_find_all=find(sum(L(:,label_notallow),2)==0); 
                idx_find=setdiff(idx_find_all,last_found_idx);  % exclude the data at previous round to avoid duplication
                last_found_idx=idx_find_all;
                
                R = randperm(size(idx_find,1)); % random generate numbers
                queryInds = R(1,1:floor(size(idx_find,1)*0.1));  % 0.1 query
                sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end); % 0.9 train
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_allow);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
%                 LTrain{l,1}=L_tmp(sampleInds,:);
                LTrain{l,1}=L_all_tmp(sampleInds,:);
                LTrain_only_incre_labels{l,1}=L_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
%                 LQuery{l,1}=L_tmp(queryInds,:);
                LQuery{l,1}=L_all_tmp(queryInds,:);
                LQuery_only_incre_labels{l,1}=L_tmp(queryInds,:);
                
                
                train_param.chunksize{l,1}=size(sampleInds,2);
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end

            
            
        end
        
        clear X L subi queryInds sampleInds R Image Tag Label

    elseif strcmp(train_param.ds_name, 'NUSWIDE')   
        
        % NUSWIDE 10
        load('../data/NUSWIDE10.mat');
        train_param.image_feature_size=500;
        
        % NUSWIDE 21
%         load('Datasets/NUSWIDE21_deep.mat');
%         train_param.text_feature_size=5018;

        train_param.trainset_size = size(I_tr, 1);
        train_param.label_size = size(L_tr, 2);
        
        if strcmp(train_param.load_type, 'second_setting')
            X = [I_tr; I_te];
            L = [L_tr; L_te];
            
            if train_param.nus4w == 1
                fprintf("Now is NUS4w mode.....\n");
                X=X(1:40000,:);
                L=L(1:40000,:);
            end
            anchor=I_tr(randsample(2000,1000),:);
            
            if size(L_tr,2) == 21
                
                train_param.nchunks=21;
                
                labels=linspace(1,21,21);
                seperate=cell(train_param.nchunks,1);
                
                seperate{1,1}=[1];
                seperate{2,1}=[2];
                seperate{3,1}=[3];
                seperate{4,1}=[4];
                seperate{5,1}=[5];
                seperate{6,1}=[6];
                seperate{7,1}=[7];
                seperate{8,1}=[8];
                seperate{9,1}=[9];
                seperate{10,1}=[10];
                seperate{11,1}=[11];
                seperate{12,1}=[12];
                seperate{13,1}=[13];
                seperate{14,1}=[14];
                seperate{15,1}=[15];
                seperate{16,1}=[16];
                seperate{17,1}=[17];
                seperate{18,1}=[18];
                seperate{19,1}=[19];
                seperate{20,1}=[20,21];
            elseif size(L_tr,2) == 10
                
                train_param.nchunks=10;
                
                labels=linspace(1,10,10);
                seperate=cell(train_param.nchunks,1);
                
                seperate{1,1}=[1];
                seperate{2,1}=[2];
                seperate{3,1}=[3];
                seperate{4,1}=[4];
                seperate{5,1}=[5];
                seperate{6,1}=[6];
                seperate{7,1}=[7];
                seperate{8,1}=[8];
                seperate{9,1}=[9];
                seperate{10,1}=[10];

            end
            
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_only_incre_labels = cell(train_param.nchunks,1);
            
            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_only_incre_labels = cell(train_param.nchunks,1);
            
            label_allow=[];
            last_found_idx=[];
            
            for l=1:train_param.nchunks
                label_allow=[seperate{l,1} label_allow];
                label_notallow=setdiff(labels,label_allow);
                idx_find_all=find(sum(L(:,label_notallow),2)==0);
                idx_find=setdiff(idx_find_all,last_found_idx);
                last_found_idx=idx_find_all;
                
                R = randperm(size(idx_find,1));
                queryInds = R(1,1:floor(size(idx_find,1)*0.1));
                sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end);
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_allow);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
                LTrain{l,1}=L_all_tmp(sampleInds,:);
                LTrain_only_incre_labels{l,1}=L_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
                LQuery{l,1}=L_all_tmp(queryInds,:);
                LQuery_only_incre_labels{l,1}=L_tmp(queryInds,:);
                
                train_param.chunksize{l,1}=size(sampleInds,2);
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end
            
        

        clear X L subi queryInds sampleInds R
    

    
        end
    
    fprintf('-------load data finished-------\n');
    clear I_tr I_te L_tr L_te
end

