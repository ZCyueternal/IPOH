function [eva,train_time_round] = incre_evaluate_HMOH_test(train_param,XTrain,LTrain,XQuery,LQuery,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate)

%     eva=zeros(1,train_param.nchunks);
    train_time_round=zeros(1,train_param.nchunks);
    ds_name = train_param.ds_name;
    
    train_param.lr = 0.1;  
    train_param.eta = 280; 
    kernel_size = 500; 

    if strcmp(ds_name,'CIFAR10')
        train_param.lr = 0.01; 
    end

    Dtrain = train_param.image_feature_size;  % 512
    lshW = randn(train_param.hbits, train_param.current_bits);  % 512*16bits
    lshW = lshW ./ repmat(diag(sqrt(lshW'*lshW))', train_param.hbits, 1); % 512*16bits


    W = randn(Dtrain, train_param.current_bits); % 512*16bits
    W = W ./ repmat(diag(sqrt(W'* W))', Dtrain, 1);


    W = zeros(Dtrain, train_param.current_bits);
    tmp_W = 0;


    train_param.cnt = zeros(1, train_param.current_bits);
    h = hadamard(train_param.hbits);
    
    Htrain=[];
    
    BB = cell(train_param.nchunks,1);
    LTrain_only_incre = cell(train_param.nchunks,1);
    for chunki=1:train_param.nchunks

        XTrain_t=XTrain{chunki,:};
        LTrain_t=LTrain{chunki,:};
        XQuery_t=XQuery{chunki,:};
        LQuery_t=LQuery{chunki,:};

        if size(LTrain_t, 2) == 1  % single-label
            train_label = h(LTrain_t, :);
        else   % multi-label
            label_tmp = [];
            for j = 1:size(LTrain_t,1) % 588
                tmp = sum(h(LTrain_t(j,:) == 1, :), 1);
                tmp(tmp > 0) = 1;
                cnt_pos = sum(tmp == 1);
                tmp(tmp < 0) = -1;
                cnt_neg = sum(tmp == -1);

                ind_zero = find(tmp == 0);
                ind_zero = ind_zero(randperm(length(ind_zero)));

                dif = abs(cnt_pos - cnt_neg);
                nzero = length(ind_zero);

                if nzero ~= 0
                    minj = min(dif, nzero);

                    if cnt_pos > cnt_neg
                        tmp(ind_zero(1:minj)) = -1;
                        ind_zero = ind_zero(minj + 1:end);
                    end
                    if cnt_pos < cnt_neg
                        tmp(ind_zero(1:minj)) = 1;
                        ind_zero = ind_zero(minj + 1:end);
                    end

                    if length(ind_zero) > 0
                        mid = round(length(ind_zero)/2);
                        tmp(1:mid) = 1;
                        tmp(mid+1:end) = -1;
                    end 
    %                sum(tmp) 
                end  
                label_tmp = [label_tmp; tmp];             
            end    
            train_label = label_tmp;  % 588*512  ???
       end

        tic
        [B_arrow_incre,tmp_W,W,train_param,BB,LTrain_only_incre]=incre_train_HMOH(chunki,XTrain_t,LTrain_t,train_label,lshW,W,tmp_W,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre);
        train_time_round(1,chunki)=toc;
        evaluation_info.trainT = train_time_round(1,chunki);
        
        tmp_W = tmp_W ./ repmat(train_param.cnt, Dtrain, 1);  % averaged


        tic;
        fprintf('test beginning\n');
        
        Htrain_current = single(XTrain_t * tmp_W > 0);  % X * nbits
        
        Htest = single(XQuery_t * tmp_W > 0);  % (66 * 512) * (512 * 16)
        LBase=cell2mat(LTrain(1:chunki,:));
        BxTest = compactbit(Htest);
        
        %%%%%%%%%%%%%here to fuse hash codes
        
        
        
        
        if chunki >1 && train_param.plugin ==1
            abs_1 = abs(B_arrow_incre);
            abs_2 = abs(Htrain_current);
            lamda_incre = sum(abs_1(:))./sum(abs_2(:));
            Htrain_current_fusion = sign(lamda_incre*B_arrow_incre+Htrain_current);          
            Htrain = [Htrain;Htrain_current_fusion];  % accumulate
        elseif chunki >1 && train_param.plugin ==0
            Htrain_current_fusion = sign(B_arrow_incre+Htrain_current);          
            Htrain = [Htrain;Htrain_current_fusion];  % accumulate
        else
            Htrain=[Htrain Htrain_current];
        end
            
        BxTrain = compactbit(Htrain);
        DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
        [~, orderH] = sort(DHamm, 2); % each row, from low to high
        evaluation_info.Image_VS_Text_MAP = mAP(orderH', LBase, LQuery_t);
        evaluation_info.testT=toc;

        fprintf("[%s][%d bits][chunk %d]\n",ds_name, train_param.current_bits, chunki);
        fprintf('       : test time is %f\n',evaluation_info.testT);
        fprintf('       : evaluation ends, MAP_direct is %f\n',evaluation_info.Image_VS_Text_MAP);
        MAP_result(1,chunki) = evaluation_info.Image_VS_Text_MAP;
%             [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LBase, LQuery_t);
        evaluation_info.param = train_param;
        eva{chunki} = evaluation_info;
        clear evaluation_info
            

        

    end
   
end
