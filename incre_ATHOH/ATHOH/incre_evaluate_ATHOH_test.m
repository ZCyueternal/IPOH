function [eva,train_time_round] = incre_evaluate_ATHOH_test(train_param,XTrain,LTrain,XQuery,LQuery,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate)
%     eva=zeros(1,train_param.nchunks);
    train_time_round=zeros(1,train_param.nchunks);
    ds_name = train_param.ds_name;
    
    train_param.eta=0.2;
    train_param.lambda=0.1;
    train_param.T=0.3;
    
    d = train_param.image_feature_size; % 512
    
    W_tea = randn(d, train_param.current_bits);  % 512*16bits
    W_tea = W_tea ./ repmat(diag(sqrt(W_tea' * W_tea))', d, 1);
    
    
    W_stu = randn(d, train_param.current_bits);
    W_stu = W_stu ./ repmat(diag(sqrt(W_stu' * W_stu))', d, 1);
    
    train_param.Flag=1;
    
    c=train_param.label_size;  % 24
    m=max(train_param.current_bits,c);
    M=randn(m,m);
    M=normr(M);
    [U,~,~] = svd(M);
    D=sign(U(:,1:train_param.current_bits));
    D_stu=D(randperm(m),:);
    D_tea=D(randperm(m),:);
    
    
    
    h = hadamard(train_param.hbits);
    
    Htrain=[];

    BB = cell(train_param.nchunks,1);
    LTrain_only_incre = cell(train_param.nchunks,1);
    
    for chunki=1:train_param.nchunks
        fprintf('--chunk---%3d\n',chunki);
        
        XTrain_t=XTrain{chunki,:};
        XQuery_t=XQuery{chunki,:};
        LQuery_t=LQuery{chunki,:};
        LTrain_tmp=LTrain{chunki,:};
        
        Nt=size(XTrain_t,1); % 588
        
        LTrain_t=cell(Nt,1);
        trainlabel_stu=cell(Nt,1);
        trainlabel_tea=cell(Nt,1);
        for i=1:Nt
            LTrain_t{i,1}=find(LTrain_tmp(i,:)==1)';
            trainlabel_stu{i,1}=D_stu(LTrain_tmp(i,:)==1,:);
            trainlabel_tea{i,1}=D_tea(LTrain_tmp(i,:)==1,:);
        end
        
        train_param.M_stu=zeros(c,train_param.current_bits);
        train_param.M_tea=zeros(c,train_param.current_bits);
        train_param.S_stu=zeros(c,train_param.current_bits);
        train_param.S_tea=zeros(c,train_param.current_bits);
        train_param.A_stu=zeros(1,train_param.current_bits);
        train_param.A_tea=zeros(1,train_param.current_bits);
        train_param.Ny=zeros(1,c);
        
        tic
        if chunki==1
            [train_param,W_stu,W_tea,BB,LTrain_only_incre] = incre_train_ATHOH0(chunki,XTrain_t,LTrain_t,LTrain_tmp,trainlabel_stu,trainlabel_tea,W_stu,W_tea,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre);

        else
            [B_arrow_incre,train_param,W_stu,W_tea,BB,LTrain_only_incre] = incre_train_ATHOH(chunki,XTrain_t,LTrain_t,LTrain_tmp,trainlabel_stu,trainlabel_tea,W_stu,W_tea,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre);
            
        end
        train_time_round(1,chunki)=toc;
        evaluation_info.trainT = train_time_round(1,chunki);
        
        tic;
        fprintf('test beginning\n');
        
        Htrain_current = single(W_stu' * XTrain_t'> 0);
        
        Htest = single(W_stu' * XQuery_t' > 0); % (512 * 16)' * (65 * 512)'
        LBase=cell2mat(LTrain(1:chunki,:));
        BxTest = compactbit(Htest');
        
        
%         Htrain=[Htrain Htrain_current];
        
        %%%%%%%%%%%%%here to fuse hash codes
        
        if chunki > 1 && train_param.plugin ==1
            abs_1 = abs(B_arrow_incre);
            abs_2 = abs(Htrain_current');
            lamda_incre = sum(abs_1(:))./sum(abs_2(:));
            Htrain_current_fusion = sign(lamda_incre*B_arrow_incre+Htrain_current');
            Htrain = [Htrain Htrain_current_fusion'];  % accumulate
            
        elseif chunki >1 && train_param.plugin ==0
            Htrain_current_fusion = sign(B_arrow_incre+Htrain_current');
            Htrain = [Htrain Htrain_current_fusion'];  % accumulate
            
        else
            Htrain=[Htrain Htrain_current];
        end
        


        BxTrain = compactbit(Htrain');
        DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
        [~, orderH] = sort(DHamm, 2); % each row, from low to high
        evaluation_info.Image_VS_Text_MAP = mAP(orderH', LBase, LQuery_t);
        evaluation_info.testT=toc;

        fprintf("[%s][%d bits][chunk %d]\n",ds_name, train_param.current_bits, chunki);
        fprintf('       : test time is %f\n',evaluation_info.testT);
        fprintf('       : evaluation ends, MAP_fusion is %f\n',evaluation_info.Image_VS_Text_MAP);
        MAP_result(1,chunki) = evaluation_info.Image_VS_Text_MAP;
%             [evaluation_info.Image_VS_Text_precision_f, evaluation_info.Image_VS_Text_recall_f] = precision_recall(orderH_f', LBase, LQuery_t);
        evaluation_info.param = train_param;
        eva{chunki} = evaluation_info;
        clear evaluation_info            
        
        
        
        
%         BxTrain = compactbit(Htrain');
%     
%         DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
%         [~, orderH] = sort(DHamm, 2); % each row, from low to high
%     
%     % my mAP
%         evaluation_info.Image_VS_Text_MAP = mAP(orderH', LBase, LQuery_t);
%         evaluation_info.testT=toc;
%         fprintf("[%s][%d bits][chunk %d]\n",ds_name, train_param.current_bits, chunki);
%         fprintf('       : test time is %f\n',evaluation_info.testT);
%         fprintf('       : evaluation ends, MAP is %f\n',evaluation_info.Image_VS_Text_MAP);
%         MAP_result(1,chunki) = evaluation_info.Image_VS_Text_MAP;
% 
% %         [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LBase, LQuery_t);
%     
%         evaluation_info.param = train_param;
%         
%         eva{chunki} = evaluation_info;
%         clear evaluation_info
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%         tic;
%         fprintf('test beginning\n');
%         Htrain_temp = single(W_stu' * XTrain_t'> 0);
%         
%         Htrain=[Htrain Htrain_temp];
%         
%         %%%%%%%%%%%%%here to fuse hash codes
%         
%         Htest = single(W_stu' * XQuery_t' > 0);
% 
%         LBase=cell2mat(LTrain(1:chunki,:));
% 
%         BxTest = compactbit(Htest');
%         BxTrain = compactbit(Htrain');
%     DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
%     [~, orderH] = sort(DHamm, 2); % each row, from low to high
%     
%     % my mAP
%     evaluation_info.Image_VS_Text_MAP = mAP(orderH', LBase, LQuery_t);
%     evaluation_info.testT=toc;
%     fprintf("[%s][%d bits][chunk %d]\n",ds_name, train_param.current_bits, chunki);
%     fprintf('       : test time is %f\n',evaluation_info.testT);
%     fprintf('       : evaluation ends, MAP is %f\n',evaluation_info.Image_VS_Text_MAP);
%     MAP_result(1,chunki) = evaluation_info.Image_VS_Text_MAP;
% 
% [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LBase, LQuery_t);
%     
%         evaluation_info.param = train_param;
%         
%         eva{chunki} = evaluation_info;
%         clear evaluation_info
        
        

        
    end
end
