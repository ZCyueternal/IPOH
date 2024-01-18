function [B_arrow_incre,train_param,W_stu,W_tea,BB,LTrain_only_incre] = incre_train_ATHOH(chunki,XTrain_t,LTrain_t,LTrain_tmp,trainlabel_stu,trainlabel_tea,W_stu,W_tea,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre) 
    k=train_param.current_bits;
    eta=train_param.eta;
    
    incre_paramiter = train_param.incre_paramiter;
    mu_incre = train_param.mu_incre;
    sita_incre = train_param.sita_incre;
    nbits = train_param.current_bits;
    seperate_t = seperate{chunki,:};  % the incremental labels of this chunki

    LTrain_new_only_incre = LTrain_tmp(:,seperate_t);

    LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every 

    h = hadamard(512); % mir
    h = h(randperm(size(LTrain_tmp,2)),randperm(nbits)); % 5000*nbits
    
    
    for t=1:size(XTrain_t,1)
        x=XTrain_t(t,:)';   %d*1
        for j=1:size(LTrain_t{t,1},1)
            y=LTrain_t{t,1}(j,:);
            target_codes_stu=trainlabel_stu{t,1}(j,:)';
            target_codes_tea=trainlabel_tea{t,1}(j,:)';
        
            %%%caculate bit weight A
            train_param.Ny(1,y)=train_param.Ny(1,y)+1;
            for i=1:k
                train_param.M_stu(y,i)=target_codes_stu(i,1);
                train_param.M_tea(y,i)=target_codes_tea(i,1);
                temp_tea=(train_param.Ny(1,y)-1)*train_param.S_tea(y,i)+(W_tea(:,i)'*x-target_codes_tea(i,1)).^2;
                temp_stu=(train_param.Ny(1,y)-1)*train_param.S_stu(y,i)+(W_stu(:,i)'*x-target_codes_stu(i,1)).^2;
                train_param.S_tea(y,i)=temp_tea/train_param.Ny(1,y);
                train_param.S_stu(y,i)=temp_stu/train_param.Ny(1,y);
                temp_tea= pdf('Normal',W_tea(:,i)'*x,train_param.M_tea(y,i),train_param.S_tea(y,i));
                temp_stu= pdf('Normal',W_stu(:,i)'*x,train_param.M_stu(y,i),train_param.S_stu(y,i));
                train_param.A_tea(y,i)=max((train_param.T-temp_tea)./train_param.T,0);
                train_param.A_stu(y,i)=max((train_param.T-temp_stu)./train_param.T,0);
            end


             if train_param.Flag==1
                        %%%update W_tea by eq17
                for c=1:k
                    W_tea(:,c)=W_tea(:,c)-eta*train_param.A_tea(y,c)*(W_tea(:,c)'*x-target_codes_tea(c,:))*x;
                end
             end

            %%%update W_stu by eq18
            rS=(k-target_codes_stu'*W_stu'*x)/2*k;
            rT=(k-target_codes_tea'*W_tea'*x)/2*k;

            for c=1:k
                W_stu(:,c)=W_stu(:,c)-eta*train_param.A_stu(y,c)*(W_stu(:,c)'*x-target_codes_stu(c,:))*x;%-train_param.lambda*x*(rS-rT)*target_codes_stu(c,:);
            end
        end
    end
    train_param.Flag=-1*train_param.Flag;
    
    BB{chunki,1} = single(XTrain_t * W_stu > 0);
    
    B_arrow_incre = single(XTrain_t * W_stu > 0); % 603*16
    fprintf("\n!!!Starting incremental calculation......\n\n");
    
    L_hat = LTrain_tmp(:, seperate{chunki-1,1});  % 603*1
    L_tilde = LTrain_only_incre{chunki-1,1};
    B_arrow_incre = single(XTrain_t * W_stu > 0); % 603*16

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
    
    
    
    
    
end