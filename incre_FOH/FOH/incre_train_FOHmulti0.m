function [Xe_t,Be_t,ve_t,q,qlabel,now_X,now_B,now_L,tmp_W,dex,LTrain_only_incre,BB] = incre_train_FOHmulti0(param,chunki,train_t,trainLabel_t,W_t,seperate)
    

    seperate_t = seperate{chunki,:};  % the incremental labels of this chunki
    
    LTrain_new_only_incre = trainLabel_t(seperate_t,:);
    
    LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every




    Xe_t = train_t;  % 512 * n_t
    tmp = W_t' * Xe_t;  % 16 * n_t
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;
    Be_t = tmp; % 16 * n_t
    BB{1,1} = Be_t;
    now_X = Xe_t; % 512 * n_t
    now_B = single(W_t' * Xe_t >=0);    % 16 * n_t

    ve_t = trainLabel_t;
    [qpool, qind] = datasample(Xe_t',param.numq);
    q = qpool';
    qlabel = ve_t(:,qind);
    qlabel = qlabel';
    Hq = single(W_t' * q >= 0);
    dex = single(knnsearch(now_B',Hq','K',param.K,'Distance','hamming'));
    ud = unique(dex(:)); 
    seq = (1:size(now_B,2));
    now_B(:,setdiff(seq,ud)) = nan;
    now_X(:,setdiff(seq,ud)) = nan;
    now_L = ve_t; 
    now_L(:,setdiff(seq,ud)) = nan;
    tmp_W = W_t;
end

