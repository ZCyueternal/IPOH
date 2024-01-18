function [train_param,W_stu,W_tea,BB,LTrain_only_incre] = incre_train_ATHOH0(chunki,XTrain_t,LTrain_t,LTrain_tmp,trainlabel_stu,trainlabel_tea,W_stu,W_tea,train_param,LTrain_only_incre_labels,LQuery_only_incre_labels,seperate,BB,LTrain_only_incre) 
    k=train_param.current_bits;
    eta=train_param.eta;
    
    seperate_t = seperate{chunki,:};  % the incremental labels of this chunki

    LTrain_new_only_incre = LTrain_tmp(:,seperate_t);

    LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every 
    
    
    for t=1:size(XTrain_t,1)
        x=XTrain_t(t,:)';   
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
        
            %%%W = d*k    x=d*1   target_codes=k*1;
            %%%update W_stu and W_tea
            for c=1:k
                W_stu(:,c)=W_stu(:,c)-eta*(W_stu(:,c)'*x-target_codes_stu(c,:))*x;
                W_tea(:,c)=W_tea(:,c)-eta*(W_tea(:,c)'*x-target_codes_tea(c,:))*x;
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
end