function [MAP_result,training_time] = incre_train_twostep(XTrain, LTrain, XQuery, LQuery,param,anchor,Binit,Vinit,Pinit,Sinit,h,seperate)

%% get the dimensions of features
n = param.trainset_size;
dX = size(anchor,1);  % 1000 (1000*4096)
dY = param.label_size;  % 24

%% set the parameters
nbits = param.nbits; % length of the hash code
beta = param.beta;
alpha = param.alpha;
delta = param.delta;
% chunk = param.chunk; % 2000
gama = param.gama;

Sinit = Sinit +diag(2+zeros(1,dY)); % diag are all 1;

sita = param.sita;
yita = param.yita;
epsilon = param.epsilon;

paramiter = param.paramiter; % 10
incre_paramiter = param.incre_paramiter; % 5
nq = param.nq;

mu_incre = param.mu_incre;
sita_incre = param.sita_incre;

% dataset = param.datasets;

%% initialization
% Calculate n1,n2
% nq = 200 or 400
% n1 = floor((nq/chunk)*nq);
n1 = param.n1;
n2 = nq-n1; 

MAP_result=zeros(1,param.nchunks);  

nmax = param.nmax; % For select points   500 for incre  1000 for nus
% A = zeros(n,dY); % 16000*

myindex = zeros(param.nchunks,nmax);  % initialize ?*500   ? rounds' first 500 points

% copy this to the class-wise hash code (leibie haxi ma)
Y = h; % c*r

B = Binit;
V = Vinit;
P = Pinit;

S = Sinit; % This is c*c;


LTrain_only_incre = cell(param.nchunks,1);



for chunki = 1:param.nchunks
    fprintf('-----chunk----- %3d\n', chunki); 
    
    XTrain_new = XTrain{chunki,:};
    LTrain_new = LTrain{chunki,:};

    XQueryt = XQuery{chunki,:};
    LQueryt = LQuery{chunki,:};
    
    seperate_t = seperate{chunki,:};  % the incremental labels of this chunki
    
    LTrain_new_only_incre = LTrain_new(:,seperate_t);
    
    LTrain_only_incre{chunki,1} = LTrain_new_only_incre;   % save every 
%     LQueryt_new_only_incre = LQueryt(:,seperate_t);

    %% iterative optimization

    X = Kernelize(XTrain_new, anchor);

    tic;
    
    fprintf('[%s][%d bits][%d chunk]\n',param.ds_name,nbits,chunki);
        
    if chunki == 1
%%        simple Tag and low-level feature

        current_nt = size(XTrain_new, 1);
        B_new = sign(randn(current_nt, nbits));
        
        normytagA = ones(current_nt,1);% 588*1
        normdegree = zeros(1,current_nt); % 1*588   588 ge 2-norm value of (L-Y)
        
        L_minus_BY = LTrain_new-B_new*Y';
        degree = L_minus_BY.^2;
        
        for j = 1:current_nt %1:588
            normdegree(j) = norm(degree(j,:),2); % 2-norm of (Y) 1*2000
        end
        [~,index] = sort(normdegree); % get the index  (ascending,from low to high)
        myindex(chunki,:) = index(1:nmax); % first 1k / 500 index of the points add to this round

        % norm simple YTrain, norm XTrain(after kernel)
        for i =1:current_nt %1:588
            if norm(LTrain_new(i,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(LTrain_new(i,:));% -d column vector
            end
            if norm(X(i,:))~=0 % if current chunk's L's norm !=0
                normX(i,:)=norm(X(i,:));% -d column vector (*1)
            end
        end
        
        % This is ||L||
        normytagA = repmat(normytagA,1,dY); % 
        normX = repmat(normX,1,dX); % *1000
        
        % SA is G t arrow (Gt=Lt/||Lt||)
        SA_new = LTrain_new./normytagA; 
        % SX is X/|X|
        SX_new = X./normX;

        for iter = 1:paramiter

            % update V

            LTB = SA_new'*B_new; % V15 also use this part, please save this sentence.
            XTB = SX_new'*B_new;

        % V16 only use Sno and Sqn
            Qt =  beta*B_new;

%% eigenvalue decompositon(like DGH)         
%             Temp = Qt'*Qt-1/nsample*(Qt'*ones(nsample,1)*(ones(1,nsample)*Qt));
            Temp1 = Qt'*(eye(current_nt)-1/current_nt*ones(1,current_nt)*ones(current_nt,1))*Qt;
%             [~,Lmd,QQ] = svd(Temp);
            [~,Lmd,QQ] = svd(Temp1);
%             clear Temp
            clear Temp1
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            % The Qt of Temp and PP is opposite. shi xiang fan de
            PP = (Qt-1/current_nt*ones(current_nt,1)*(ones(1,current_nt)*Qt)) *  (Q / (sqrt(Lmd(idx,idx))));
            P_ = orth(randn(current_nt,nbits-length(find(idx==1))));
            V_new = sqrt(current_nt)*[PP P_]*[Q Q_]';
            
            % update Y arrow
            G = sita*S'*Y + yita*LTrain_new'*B_new+epsilon*h*(1./nbits); % G:c*r
            for k=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y(:,place) = sign(nbits*G(:,place)   -   Y(:,bit)*B_new(:, bit)'*B_new(:, place)-yita*Y(:,bit)*Y(:,bit)'*Y(:,place));
                end  
            end
            
            % UPDATE B my method

            LTV = SA_new'*V_new;
            XTV = SX_new'*V_new;


            % V16 only use Sno and Sqn
            U = beta*V_new + yita*nbits*LTrain_new*Y;
            for time=1:3
                for location=1:nbits
                    bite=1:nbits;
                    bite(location)=[];
                    B_new(:,location) = (sign(U(:,location)-yita*B_new(:,bite)*Y(:,bite)'*Y(:,location)))';
                end  
            end

            
        end
        %% save results
        
        C1 = X'*X;  % C1=X'*X
        C2 = X'*B_new;  % C2=X'*B
        Old_B = B_new;
        Old_V = V_new;
        % update P(hash funcation projector)
        P = pinv(C1+delta*eye(dX))*(C2);
        
        Qq = LTrain_new(myindex(1,1:nq),:);
        Xq = X(myindex(1,1:nq),:);
        
        Btemp = B_new;
        Bq = Btemp(myindex(1,1:nq),:);
        
        HH{1,1} = C1;
        HH{1,2} = C2;
        HH{1,3} = LTB;
        HH{1,4} = XTB;
        HH{1,5} = LTV;
        HH{1,6} = XTV;
        
        BB{1,1} = B_new;
        BB{1,2} = V_new;
        BB{1,3} = Bq;
        BB{1,4} = Qq;
        BB{1,5} = Xq;
        
        
    end
    
    if chunki >= 2
        
% 	    P_last = P;
        CC1 = C1;
        CC2 = C2;
        OOld_B = Old_B;
        OOld_V = Old_V;
        LLTB = LTB;
        XXTB = XTB;

        LLTV = LTV;
        XXTV = XTV;
        
        current_nt = size(XTrain_new, 1);
        B_new = sign(randn(current_nt, nbits));
        
        normdegree = zeros(1,current_nt);
        normytagA = ones(current_nt,1);
        
        L_minus_BY = LTrain_new-B_new*Y';
        degree = L_minus_BY.^2;
              
        for j = 1:current_nt
            normdegree(j) = norm(degree(j,:),2);
        end
        [~,index] = sort(normdegree);
        myindex(chunki,:) = index(:,1:nmax);
        clear normX;
        for i =1:current_nt %1:current_nt
            if norm(LTrain_new(i,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(LTrain_new(i,:));% 2000-d column vector
            end
            if norm(X(i,:))~=0 % if current chunk's L's norm !=0
                normX(i,:)=norm(X(i,:));% 2000-d column vector (2000*1)
            end
        end
        
        normytagA = repmat(normytagA,1,dY); % 2000*
        normX = repmat(normX,1,dX); % *1000
        
        SA_new = LTrain_new./normytagA; % Gt=Lt/||Lt||
        % SX is X/|X|
        SX = X./normX;
        
        for iter = 1:paramiter

             % update V
            % Notation C is Qt in paper
            
            LLqT = SA_new*Qq';
            iii = find(LLqT==0);
            
            S_mnT = SX*Xq';
            S_mnT(iii) = -1;
            
           
            % V16, only use Sno, Sqn,
            Qt = beta*B_new+gama*nbits*S_mnT*Bq;   % BB{1,3} is also Bq!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            Temp = Qt'*Qt-1/current_nt*(Qt'*ones(current_nt,1)*(ones(1,current_nt)*Qt));
            [~,Lmd,QQ] = svd(Temp); clear Temp  % Lmd is \sigma^2, which is 
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx)); % value of non-zero, value of zerp
            PP = (Qt-1/current_nt*ones(current_nt,1)*(ones(1,current_nt)*Qt)) *  (Q / (sqrt(Lmd(idx,idx))));
            P_ = orth(randn(current_nt,nbits-length(find(idx==1))));
            V_new = sqrt(current_nt)*[PP P_]*[Q Q_]';

            % update Y arrow
            G = sita*S'*Y + yita*LTrain_new'*B_new+epsilon*h; % G:c*r
            for k=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y(:,place) = sign(nbits*G(:,place) - Y(:,bit)*B_new(:,bit)'*B_new(:,place)-yita*Y(:,bit)*Y(:,bit)'*Y(:,place));
                end  
            end
            
            % UPDATE B my method

        % V16 
%         U = beta*V(e_id:n_id,:) + yita*nbits*A(e_id:n_id,:)*Y + alpha*nbits*SA(e_id:n_id,:)*LLTV;
            U = beta*V_new + yita*nbits*LTrain_new*Y + alpha*nbits*(SA_new*LLTV-ones(1,current_nt)'*(ones(1,size(cell2mat(XTrain(1:chunki-1,:)),1))*Old_V));
            
            for time=1:3
                for location=1:nbits
                    bite=1:nbits;
                    bite(location)=[];
                    B_new(:,location) = (sign(U(:,location)-yita*B_new(:,bite)*Y(:,bite)'*Y(:,location)))';
                end  
            end

        end
        
        
        L_hat = LTrain_new(:, seperate{chunki-1,1});  % 603*1
        S_incre = -ones(size(seperate{chunki-1,1},2),size(seperate_t,2));  % S:1*3
%         S_incre1 = -ones(size(L_hat,2),size(seperate_t,2));

%         Y_arrow_incre = h(seperate_t,:); % current chunk incre label    Y arrow 3*16
%         Y_tilde_incre = h(seperate{chunki-1,1},:); % last chunk incre label  Y tilde 1*16
        Y_arrow_incre = Y(seperate_t,:); % current chunk incre label    Y arrow 3*16
        Y_tilde_incre = Y(seperate{chunki-1,1},:); % last chunk incre label  Y tilde 1*16
        B_arrow_incre = B_new; 
        
         for incre_iter = 1:incre_paramiter
            
            % [incre]-Update Y_old
            
            G_incre = LTrain_only_incre{chunki-1,1}'*BB{chunki-1,1} + mu_incre*L_hat'*B_new + sita_incre*S_incre*Y_arrow_incre;
            for rr=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y_tilde_incre(:,place) = sign(nbits*G_incre(:,place) - Y_tilde_incre(:,bit)*BB{chunki-1,1}(:,bit)'*BB{chunki-1,1}(:,place)-mu_incre*Y_tilde_incre(:,bit)*B_new(:,bit)'*B_new(:,place)-sita_incre*Y_tilde_incre(:,bit)*Y_arrow_incre(:,bit)'*Y_arrow_incre(:,place));
                end  
            end
            
            
            % [incre]-Update Y_arrow  3*16  increlabels * nbits
            
            U_incre = mu_incre*LTrain_new(:,seperate_t)'* B_new + sita_incre*S_incre'*Y_tilde_incre;
            for rrr=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y_arrow_incre(:,place) = sign(nbits*U_incre(:,place) - mu_incre*Y_arrow_incre(:,bit)*B_new(:,bit)'*B_new(:,place)-sita_incre*Y_arrow_incre(:,bit)*Y_tilde_incre(:,bit)'*Y_tilde_incre(:,place));
                end  
            end
            
            
            % [incre]-Update B_arrow 
            
            Gb_incre = nbits*mu_incre*(LTrain_new(:,seperate_t)*Y_arrow_incre + L_hat*Y_tilde_incre);
            for rrrr=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    B_arrow_incre(:,place) = sign(Gb_incre(:,place) - mu_incre*B_arrow_incre(:,bit)*Y_arrow_incre(:,bit)'*Y_arrow_incre(:,place)-mu_incre*B_arrow_incre(:,bit)*Y_tilde_incre(:,bit)'*Y_tilde_incre(:,place));
               end  
            end


        end

        %% hash code fusion
        if param.plugin ==1
            abs_1 = abs(B_arrow_incre);
            abs_2 = abs(B_new);
            lamda_incre = sum(abs_1(:))./sum(abs_2(:));
            B_new = sign(lamda_incre*B_arrow_incre+B_new);

        end
        
%%
        % update Qq
        yindex1 = myindex(chunki,1:n2);

        neighbor = LTrain_new(yindex1,:);
        

        olddata = Qq(randsample(nq,n1),:);
        Qq = [olddata;neighbor];
        
        
        % update Xq
        xindex = myindex(chunki,1:n2);
        neighbor1 = X(xindex,:);
        olddata1 = Xq(randsample(nq,n1),:);
        Xq = [olddata1;neighbor1];
        
        % update Bq
        oldBq = Bq(randsample(nq,n1),:);  
        Btemp = B_new;
        Bq = [oldBq;Btemp(myindex(chunki,1:n2),:)]; 


%%        
        
        LTB_new = SA_new'*B_new;
        XTB_new = SX'*B_new;
        LTB = LLTB+LTB_new;
        XTB = XXTB+XTB_new;
        
        LTV_new = SA_new'*V_new;
        XTV_new = SX'*V_new;
        LTV = LLTV+LTV_new;
        XTV = XXTV+XTV_new;
        
        Old_B = [OOld_B ; B_new];
        Old_V = [OOld_V ; V_new];

        C1_new = X'*X;
        C2_new = X'*B_new;
        
        % update P
        C1 = CC1+C1_new;
        C2 = CC2+C2_new;
        
        P = pinv(C1+delta*eye(dX))*(C2);   
	    
        
       
        HH{1,1} = HH{1,1}+C1_new;
        HH{1,2} = HH{1,2}+C2_new;
        
        HH{1,3} = HH{1,3}+LTB_new;
        HH{1,4} = HH{1,4}+XTB_new;
        HH{1,5} = HH{1,5}+LTV_new;
        HH{1,6} = HH{1,6}+XTV_new;
        
        BB{end+1,1} = B_new;
        BB{end,2} = V_new;
        BB{end,3} = Bq;
        BB{end,4} = Qq;
        BB{end,5} = Xq;
        
    
    end
    training_time(1,chunki) = toc;
    
    fprintf('       : training ends, training time is %f,\nevaluation begins. \n',training_time(1,chunki));
    
    XKTest=Kernelize(XQueryt,anchor); % Da phi(XTest)
    
    BxTest = compactbit(XKTest*P >= 0);
    
    B = cell2mat(BB(1:chunki,1));
    
    BxTrain = compactbit(B >= 0);
    DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
    [~, orderH] = sort(DHamm, 2); % each row, from low to high
    
    % my mAP
    MAP  = mAP(orderH', cell2mat(LTrain(1:chunki,:)), LQueryt);
    fprintf('       : evaluation ends, MAP is %f\n',MAP);
    
    if param.plugin == 1
        fprintf('With Plugin for incremental learning......\n');
    end
    if strcmp(param.ds_name,"NUSWIDE") && param.nus4w == 1
        fprintf("USE NUSWIDE 40000 datasets......\n");
    end
    
    
    MAP_result(1,chunki)=MAP;
    
    % another mAP calculation method
%     param.unsupervised = 0;
%     Aff = affinity([], [], L_tr(1:n_id,:), L_te, param);
%     param.metric = 'mAP';
%     res = evaluate(B(1:n_id,:) >= 0, XKTest*P >= 0, param, Aff);
%     MAP_result(1,round) = res;

    
    
end
