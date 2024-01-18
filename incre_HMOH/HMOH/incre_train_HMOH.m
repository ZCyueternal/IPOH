function [B_arrow_incre,tmp_W,W,train_param,BB,LTrain_only_incre] = incre_train_HMOH(chunki,XTrain_t,LTrain_t,train_label,lshW,W,tmp_W,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre)
        

        incre_paramiter = train_param.incre_paramiter;
        mu_incre = train_param.mu_incre;
        sita_incre = train_param.sita_incre;
        nbits = train_param.current_bits;
        seperate_t = seperate{chunki,:};  % the incremental labels of this chunki
    
        LTrain_new_only_incre = LTrain_t(:,seperate_t);
    
        LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every 
        
        h = hadamard(512); % mir
        h = h(randperm(size(LTrain_t,2)),randperm(nbits)); % 5000*nbits
        
        
        Dtrain=size(XTrain_t,2);
        for t = 1:size(XTrain_t,1)  % 588
            
            train_class_hash = train_label(t, :);
            X = XTrain_t(t, :);

            f = X*W;
            F = sign(f);
            F(F == 0) = -1;

            if train_param.current_bits ~= train_param.hbits
                B1 = single(train_class_hash * lshW > 0);
                B1(B1<=0) = -1;     
            else
                B1 = train_class_hash;
            end
            % B1 is 1*nbits

        % Perceptual Algorithm
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xx = (F~=B1);
            T = single(xx); % false classification
            train_param.cnt(xx) = train_param.cnt(xx) + 1;
            T = T .* B1;
            T = reshape(T', [1, train_param.current_bits, 1]);
            T_mat = repmat(T, [Dtrain, 1, 1]);

            X_hat = reshape(X', [Dtrain, 1, 1]);
            X_mat = repmat(X_hat, [1, train_param.current_bits, 1]);
            tmp = T_mat .* X_mat;
            tmp = sum(tmp, 3)/1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            W = W + train_param.lr*tmp;
            tmp_W = tmp_W + W;

        end
        BB{chunki,1} = single(XTrain_t * tmp_W > 0);
        
        if chunki > 1
            fprintf("\n!!!Starting incremental calculation......\n\n");
            
            L_hat = LTrain_t(:, seperate{chunki-1,1});  % 603*1
            L_tilde = LTrain_only_incre{chunki-1,1};
            B_arrow_incre = single(XTrain_t * tmp_W > 0); % 603*16
            
            S_incre = -ones(size(seperate{chunki-1,1},2),size(seperate_t,2));  % S:1*3
%         S_incre1 = -ones(size(L_hat,2),size(seperate_t,2));
            Y_arrow_incre = h(seperate_t,:); % current chunk incre label    Y arrow 3*16
            Y_tilde_incre = h(seperate{chunki-1,1},:); % last chunk incre label  Y tilde 1*16
            B_tilde_incre = BB{chunki-1,1};
            L_arrow = LTrain_new_only_incre;
            
            
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
            
            
            
            
            
            
            
            
        else
            B_arrow_incre = [];
        end
        
        
        
        
        
        
end
