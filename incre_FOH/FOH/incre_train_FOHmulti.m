function [Xe_t,Be_t,le_t,ve_t,Xs_t,now_X,Bs_t,ls_t,vs_t,now_L,W_t,q,qlabel,now_B,tmp_W,S,LTrain_only_incre,BB] = incre_train_FOHmulti(S,chunki,tmp_W,q,qlabel,param,Xe_t,Xs_t,Be_t,Bs_t,le_t,ls_t,ve_t,vs_t,train_t,now_X,W_t,trainLabel_t,now_L,P_t,h,seperate,LTrain_only_incre,BB)
    dL = param.dL; % 24
    dX = param.dX; % 512
    incre_paramiter = param.incre_paramiter;
    mu_incre = param.mu_incre;
    sita_incre = param.sita_incre;
    nbits = param.nbits;
    
    seperate_t = seperate{chunki,:};  % the incremental labels of this chunki
    
    LTrain_new_only_incre = trainLabel_t(seperate_t,:);
    
    LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every 
    
    if chunki == param.nchunks
%         Xe_last_round = Xs_t;
        Xe_t = [Xe_t, Xs_t];
        Be_t = [Be_t, Bs_t]; 
%         le_t = [le_t; ls_t];
        ve_t = [ve_t, vs_t];

        Xs_t = train_t;
        now_X = [now_X,Xs_t];

        tmp = W_t' * Xs_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;

        Bs_t = tmp;  

        vs_t = trainLabel_t;
        now_L = [now_L, vs_t];
        

        S_t = S(param.current_index_start:param.current_index_start+size(train_t,2)-1, 1:param.current_index_start-1) ;
        fprintf('size of S_t %d %d\n',size(S_t));
        
        for i = 1:size(train_t,2)
            if sum(S_t(i,:)) ~= 0
                ind = find(S_t(i,:) ==1);
                if(ind)
                    Bs_t(:, i) = Be_t(:, ind(1));
                end
            end
        end
        S_t(S_t == 0) = -param.etad;
        %S_t = S_t * opts.nbits;
        
        % update Bs
        G = param.nbits * Be_t * S_t' + param.sigma * W_t' * Xs_t + param.theta * P_t * vs_t;
        for r = 1:param.nbits
            be = Be_t(r, :);
            Be_r = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];
            
            bs = Bs_t(r, :);
            Bs_r = [Bs_t(1:(r-1),:); Bs_t((r+1):end, :)];
            
            g = G(r, :);
            G_r = [G(1:(r-1), :); G((r+1):end, :)];
            
            tmp = g - be * Be_r' * Bs_r;
            tmp(tmp >= 0) = 1;
            tmp(tmp < 0) = -1;
            
            Bs_t(r, :) = tmp;
        end
        
        %update Be
        Z = param.nbits * Bs_t * S_t - param.mu * P_t * ve_t;
        Be_t = 2 * Z - Bs_t * Bs_t' * Be_t;
        Be_t(Be_t >= 0) = 1;
        Be_t(Be_t < 0) =-1;
        
        %update P_t
        I_c = eye(dL);
        P_t = (param.mu * Be_t * ve_t' + param.theta * Bs_t * vs_t')/(param.theta * (vs_t * vs_t')+ param.mu * (ve_t * ve_t') + param.tau * I_c);
        
        % update W
        I = eye(dX);
        W_t = param.sigma * inv(param.sigma * (Xs_t * Xs_t') + param.lambda * I) * Xs_t * Bs_t';
              

    else
%         Xe_last_round = Xs_t;
        Xe_t = [Xe_t, Xs_t];
        Be_t = [Be_t, Bs_t]; 
        le_t = [le_t; ls_t];
        ve_t = [ve_t, vs_t];

        Xs_t = train_t;
        now_X = [now_X,Xs_t];

        tmp = W_t' * Xs_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;

        Bs_t = tmp;  

        vs_t = trainLabel_t;
        now_L = [now_L, vs_t];
        
        % s_t: current_chunk, chunks_before
        S_t = S(param.current_index_start:param.current_index_start+size(train_t,2)-1, 1:param.current_index_start-1) ;
        fprintf('size of S_t %d %d\n',size(S_t));

        for i = 1:size(train_t,2)
            if sum(S_t(i,:)) ~= 0
                ind = find(S_t(i,:) ==1);
                if(ind)
                    Bs_t(:, i) = Be_t(:, ind(1));
                end
            end
        end
    %     mulnum = floor(sum(S_t == 0) / sum(S_t ~=0));
    %     etas = mulnum * etad;
    % 
         S_t(S_t == 0) = -param.etad;
    %     S_t(S_t == 1) = etas;

        %S_t = S_t * opts.nbits;
        tag = 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update Bs
        G = param.nbits * Be_t * S_t' + param.sigma * W_t' * Xs_t + param.theta * P_t * vs_t;
        
        for r = 1:param.nbits
            be = Be_t(r, :);
            Be_r = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];

            bs = Bs_t(r, :);
            Bs_r = [Bs_t(1:(r-1),:); Bs_t((r+1):end, :)];

            g = G(r, :);
            G_r = [G(1:(r-1), :); G((r+1):end, :)];

            tmp = g - be * Be_r' * Bs_r;
            tmp(tmp >= 0) = 1;
            tmp(tmp < 0) = -1;

            Bs_t(r, :) = tmp;
        end 
        
        % update Be   existing B(old B)
        Z = param.nbits*Bs_t*S_t-param.mu*P_t*ve_t;
        
        Be_t = 2*Z-Bs_t * Bs_t' * Be_t;
        Be_t(Be_t >= 0) = 1;
        Be_t(Be_t < 0) =-1;

        % update P_t
        I_c = eye(dL);
        P_t = (param.mu * Be_t * ve_t' + param.theta * Bs_t * vs_t')/(param.theta * (vs_t * vs_t')+ param.mu * (ve_t * ve_t') + param.tau * I_c);

        % update W
        I = eye(dX);
        W_t = param.sigma * inv(param.sigma * (Xs_t * Xs_t') + param.lambda * I) * Xs_t * Bs_t';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

    
    end
    
    L_hat = trainLabel_t(seperate{chunki-1,1},:)';  % 603*1
    L_tilde = LTrain_only_incre{chunki-1,1}';  % 588*1
    S_incre = -ones(size(seperate{chunki-1,1},2),size(seperate_t,2));  % S:1*3
%         S_incre1 = -ones(size(L_hat,2),size(seperate_t,2));
    Y_arrow_incre = h(seperate_t,:); % current chunk incre label    Y arrow 3*16
    Y_tilde_incre = h(seperate{chunki-1,1},:); % last chunk incre label  Y tilde 1*16
    B_arrow_incre = Bs_t'; 
%     B_tilde_incre = Be_t';  
%     B_tilde_incre = Xe_last_round';
    B_tilde_incre = BB{chunki-1,1}';
%     L_arrow = trainLabel_t(seperate_t,:)' % use this or the one below 
    L_arrow = LTrain_new_only_incre';  % both are ok
    
    
    for incre_iter = 1:incre_paramiter
            
        % [incre]-Update Y_old

        G_incre = L_tilde'*B_tilde_incre + mu_incre*L_hat'*B_arrow_incre + sita_incre*S_incre*Y_arrow_incre;
        for rr=1:3
            for place=1:nbits
                bit=1:nbits;
                bit(place)=[];
                Y_tilde_incre(:,place) = sign(nbits*G_incre(:,place) - Y_tilde_incre(:,bit)*B_tilde_incre(:,bit)'*B_tilde_incre(:,place)-mu_incre*Y_tilde_incre(:,bit)*B_arrow_incre(:,bit)'*B_arrow_incre(:,place)-sita_incre*Y_tilde_incre(:,bit)*Y_arrow_incre(:,bit)'*Y_arrow_incre(:,place));
            end  
        end


        % [incre]-Update Y_arrow  3*16  increlabels * nbits

        U_incre = mu_incre*L_arrow'* B_arrow_incre + sita_incre*S_incre'*Y_tilde_incre;
        for rrr=1:3
            for place=1:nbits
                bit=1:nbits;
                bit(place)=[];
                Y_arrow_incre(:,place) = sign(nbits*U_incre(:,place) - mu_incre*Y_arrow_incre(:,bit)*B_arrow_incre(:,bit)'*B_arrow_incre(:,place)-sita_incre*Y_arrow_incre(:,bit)*Y_tilde_incre(:,bit)'*Y_tilde_incre(:,place));
            end  
        end


        % [incre]-Update B_arrow 

        Gb_incre = nbits*mu_incre*(L_arrow*Y_arrow_incre + L_hat*Y_tilde_incre);
        for rrrr=1:3
            for place=1:nbits
                bit=1:nbits;
                bit(place)=[];
                B_arrow_incre(:,place) = sign(Gb_incre(:,place) - mu_incre*B_arrow_incre(:,bit)*Y_arrow_incre(:,bit)'*Y_arrow_incre(:,place)-mu_incre*B_arrow_incre(:,bit)*Y_tilde_incre(:,bit)'*Y_tilde_incre(:,place));
           end  
        end
    
    end
    
    if param.plugin ==1
        abs_1 = abs(B_arrow_incre);
        abs_2 = abs(Bs_t);
        lamda_incre = sum(abs_1(:))./sum(abs_2(:));
        Bs_t = sign(lamda_incre*B_arrow_incre'+Bs_t);

    end
    
    BB{chunki,1} = Bs_t;
    

    
    [q,qlabel] = update_q(q, 50, Xs_t, vs_t',qlabel);
    Hq = single(W_t' * q > 0);
    now_B = single(W_t' * now_X >= 0);
    dex = knnsearch(now_B',Hq','K',param.K,'Distance','hamming');
    ud = unique(dex(:));
    seq = (1:size(now_B,2));
    now_B(:,setdiff(seq,ud)) = nan;
    now_X(:,setdiff(seq,ud)) = nan;
    now_L(:,setdiff(seq,ud)) = nan;

    tmp_W = tmp_W + W_t;
    
    

    
end

