% updated using subset of parameters from v02
function [aCovW,initW] = run_sgfs_hmc_lr_v02(...
                          dataset,aSd,aFdim,aN,...
                          aAlgo,aLaW,aAl,...
                          aNSmpl,aBgst,aEdst,...
                          aBurn,aLrType,...
                          aLfStep,aLfNum,aPcond,...
                          aItvPrt,aItvPlot,aItvSmpl,aItvSave,...
                          aCvPlot,aMode)
                   
%close all
dbstop if error

% load WW_HMC

%% definitions
SGD = 0;
dSGFS = 1;
fSGFS = 2;
SGLD = 3;

SG_ONLY = 1;
HMC_ONLY = 2;
BOTH = 3;

%% str2double
seed = str2double(aSd);
fdim = str2double(aFdim);
n = str2double(aN);      
algo = str2double(aAlgo);   
laW = str2double(aLaW);
al = str2double(aAl);
nSmpl = str2double(aNSmpl);
bgst = str2double(aBgst);
stscl = str2double(aEdst);
nBurn = str2double(aBurn);
mxit = nBurn + nSmpl;
lrType = str2double(aLrType);
lfStep = str2double(aLfStep);
lfNum = str2double(aLfNum);
pcond = str2double(aPcond);
itvPrt = str2double(aItvPrt);
itvPlot = str2double(aItvPlot);
itvSmpl = str2double(aItvSmpl);
itvSave = str2double(aItvSave);
cvPlot = str2double(aCvPlot);
mode = str2double(aMode);

%%
nCvPair = 25;
tvRatio = 0.2;

% errfun = 'RMSE';
% errfun = 'MISS';
% avgType = 'norm';
% avgType = 'exp';
% memRate = 0.0001;   % for exponential decaying   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data and normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% random seed
stream = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(stream);

perm = 1;
[Xt,Yt,Xv,Yv] = loaddata(dataset,tvRatio,perm,fdim);

%% random seed, This is intentionally stated again here.
stream = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(stream);

Yt = double(Yt);
Yv = double(Yv);

[YBt,YBv] = num2bin(Yt,Yv);
nO = size(YBt,2);
if nO > 2    
    BIN_CLS = 0;
else
    nO = 1;
    YBt = YBt(:,1);
    YBv = YBv(:,1);
    BIN_CLS = 1;
end

% predType = 'max';
predType = 'expect';
[Nt,nD] = size(Xt);

disp('----------- Run with Validation Set -----------');
[Xt,Xv] = stdscale(Xt,Xv);

%% set filename
str = sprintf('LR_%s_s%s_sgfs%s_rg%s_al%s_nsmp%s_br%s_lr%s_st%s-%s_itv%s_%s',...
                dataset,aSd,...
                aAlgo,aLaW,aAl,aNSmpl,aBurn,aLrType,...                
                aBgst(3:end),aEdst(4:end),...
                aItvPlot,aItvSmpl);
str = strrep(str,'.','_');		
filename = getFilename(str)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting for algo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ga = (Nt+n)/n;
nParW = (nD+1)*nO;
if algo ~= 0    % if ~SGD
    laWI = diag(ones(nParW,1)*laW);
    if algo == dSGFS
        aCovW = zeros(nParW,1);
    elseif algo == fSGFS
        aCovW = zeros(nParW);
    elseif algo == SGLD
        aCovW = 0;
    end
end

%% Optimization params
edst = bgst*stscl;
% decLr = (bgst-edst)/mxit;
eta = bgst;

%% Init Param
W = rand(nD+1,nO) - 0.5;
initW = W;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storage & Mem alloc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nErrType = 4;
W_SG = zeros(mxit-nBurn,nParW);
% ET_SG = zeros(mxit-nBurn,1);
if itvPlot
    SV_PLOT = zeros(ceil(mxit/itvPlot),nErrType); 
end
% SV_CPUT = zeros(ceil(mxit/itvPlot),1);
mn = zeros(nCvPair,2);
cv = zeros(nCvPair,2,2);
cvpair = zeros(nCvPair,2);
avgW = 0;
avglYv = 0;
avglYt = 0;
avglYs = 0;
errYvAvg = -1;
errYtAvg = -1;

%% confirm messages

% elaps_t = 0;
% avgEtime = 0;

%% load ground truth
load 120208_201725_HMC_mnist_79_M1_nLf5_lfs0_1_brn3000_nsmpl100000_la1_itv0_1_SAMPLE
mn_h = mean(W_HMC,1);
cv_h = cov(W_HMC);    
clear avgElaps_t ar
% clear W_HMC s avgElaps_t ar
if algo == SGLD % || algo == dSGFS
    W = mn_h';
end

if mode == BOTH || mode == SG_ONLY
    disp('########## RUN SG** ##########'); 
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figures
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if itvPlot
%         f1 = figure(10);
%         set(f1,'name',filename,'windowStyle','docked');
%         title(strrep(filename,'_','-'));

        if cvPlot
            f2 = figure(20);
            set(f2,'name','2d-cov','windowStyle','docked');
            title(strrep(filename,'_','-'));    
%             f3 = figure(30);
%             set(f3,'name','1d','windowStyle','docked');
%             title(strrep(filename,'_','-'));    
        end
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SGXX main iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('## start iteration ##');    
%% prepare iteration
it = 0;    
s = 0;
t = 0;
v = 0;

while it < mxit
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    if it >= nBurn && ~rem(it+itvSmpl,itvSmpl)

        s = s + 1;        
%         ka = 1/s; 

%         lYt = likelyLR(Xt,W);
%         lYv = likelyLR(Xv,W);
%         
%         avglYt = (1-ka)*avglYt + ka*lYt;
%         avglYv = (1-ka)*avglYv + ka*lYv;
        
        %%
        W_SG(s,:) = W(:,1)';
        % ET_SG(s) = elaps_t;
    end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if itvPrt && ~rem(it+itvPrt,itvPrt)
        it
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT & SAVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if itvPlot && ~rem(it+itvPlot,itvPlot)

        t = t + 1;                           
            
%             saveas(f1,filename,'png');

            %% Save        
%             if ~rem(t,itvSave)
%                 v = v + 1;                        
%                 save(strcat(filename,'.mat'),'SV_PLOT','SV_CPUT','aCovW','avgW','-v7.3');
%                 saveas(f1,filename,'fig');
%                 if SUBMIT_HHP
%                     save(strcat(filename,'_SUBMIT.mat'),'it','s','t','pYsAvg','-v7.3');
%                 end
%                 disp('---------------- SAVE COMPLETE! ----------------');
%             end
% 
%             %% eval to plot
%             if (itvSmpl > itvPlot || s < t)   % for reuse
%                 lYv = likelyLR(Xv,W);
%                 lYt = likelyLR(Xt,W);
%             end
% 
%             errYv = getErr(lYv,YBv,errfun); 
%             errYt = getErr(lYt,YBt,errfun);
%             if s > 0
%                 errYvAvg = getErr(avglYv,YBv,errfun); 
%                 errYtAvg = getErr(avglYt,YBt,errfun);         
%             end
% 
%             %% plot and print
%             set(0,'CurrentFigure',f1);        
%             loglog(it+1,errYv,'*r');               
%             title(strrep(filename,'_','-'));
%             hold on; 
%             loglog(it+1,errYt,'*b');                    
%             if s > 0 && ~rem(it+itvSmpl,itvSmpl)                                    
%                 hold on; loglog(it+1,errYvAvg,'*m');
%                 hold on; loglog(it+1,errYtAvg, '*c');
%             end            
%      
%             lr = num2str(eta)            
%             errStr = strcat('it:', num2str(it),...
%                        ', t:', num2str(t),...               
%                        ', s:', num2str(s),...
%                        ', ET_SG:', num2str(errYt),...
%                        ', EV:', num2str(errYv),...
%                        ', ETA:', num2str(errYtAvg),...                  
%                        ', EVA:', num2str(errYvAvg))
%             lr_et_Str = strcat('stsz:', lr, ', et:', num2str(elaps_t));
% 
%             xlabel({errStr;lr_et_Str});
%             drawnow;
%             SV_PLOT(t,:) = [errYt errYv errYtAvg errYvAvg];                   
        
        %% plot 2-D gaussian cov
        
        if cvPlot && it > nBurn                           
            
            set(0,'CurrentFigure',f2); 
            clf; hold on;
            for i=1:nCvPair
                subplot(ceil(sqrt(nCvPair)),ceil(sqrt(nCvPair)),i);                
                hold on;
                j = cvpair(i,1);
                k = cvpair(i,2);
                mn(i,:) = mean([W_SG(1:s,j) W_SG(1:s,k)],1);
                cv(i,:,:) = cov([W_SG(1:s,j) W_SG(1:s,k)]);     
                plot(W_SG(1:s,j),W_SG(1:s,k),'c.','markersize',0.1);
                plot(mn(i,1),mn(i,2),'bo','linewidth',2,'markersize',10);
                plot(mn_h(j),mn_h(k),'r+','linewidth',2,'markersize',10);                
                plotGauss(mn_h(j),mn_h(k),cv_h(j,j),cv_h(k,k),cv_h(j,k),1,'-r',1);
                plotGauss(mn(i,1),mn(i,2),cv(i,1,1),cv(i,2,2),cv(i,1,2),1,'-b',1);
                % axis centering
                axis([mn_h(j)-0.2 mn_h(j)+0.2 mn_h(k)-0.2 mn_h(k)+0.2]);

            end
            
%             if exist('f3','var');
%                 set(0,'CurrentFigure',f3); 
%                 mn2 = mean(W_SG(1:s,:));
%                 mn3 = mean(W_HMC(1:s,:));
%                 nn = 10;
%                 for ii=1:nn
%                     subplot(1,nn,ii);
%                     plot(nBurn+1:nBurn+s,W_SG(1:s,ii),'b.','markersize',0.1);hold on;
%                     plot(nBurn+1:nBurn+s,W_HMC(1:s,ii),'r.','markersize',0.1);hold on;
% %                     plot(nBurn+1:nBurn+s,repmat(mn2(ii),1,s),'m-','LineWidth',2);
%                 end
%             end
            drawnow;
        end

    end
    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    it = it + 1;
    if it == nBurn+1
        % tic
        bg_t = cputime; % to measure time for 1 iteration
    end
    eta = getLearningRate(it,mxit,bgst,edst,lrType);

    % pick random mini-batch
    mIdx = sample_unif(Nt,n,1);
       
    % get gradient
    dWi = getGradientLR(Xt(mIdx,:),YBt(mIdx,:),W);
    if BIN_CLS
        dWt = sum(dWi)';   
        dWti = dWi;
        Wt = W(:);
    else
        dW = squeeze(sum(dWi(mIdx,:,:)));   
        dWi_v = dWi(:,:);    
        dW_v = dW(:);
        W_v = W(:);
    end        
    
    %% update (sub)params 
    if algo == SGD    

        Wt = Wt + eta*(dWt - laW.*Wt);
        
    elseif algo == SGLD

        Wt = Wt + (eta/2)*((Nt/n)*dWt - laW.*Wt) ...
            + sqrt(eta)*randn(size(dWt));
        
    elseif algo == dSGFS || algo == fSGFS
        
        if algo == dSGFS      
            [Wt,aCovW] = updateSGFS_v05(...
                                algo,dWti,dWt,Wt,...
                                aCovW,nParW,it,laW,al,ga,Nt,n);   
        elseif algo == fSGFS            
            [Wt,aCovW] = updateSGFS_v05(...
                                algo,dWti,dWt,Wt,...
                                aCovW,nParW,it,laW,al,ga,Nt,n);               
        end        
                        
    else
        error('undefined algorithm');
    end
     
    % reallocation    
    W = reshape(Wt,nD+1,nO);
    avgW = (1-(1/it))*avgW + (1/it)*W;
               
%     if it == nBurn
%         fprintf('time for 1 iter: %s', num2str(cputime-bg_t));
%     end
  
    %% choose 'k' cov pair for 2D scatter drawing
    if it == nBurn       
        if algo == dSGFS || algo == SGLD
            load cvpair25_gradCov
%             load cvpair25_paraCoef
%             if exist('fsgfs_cvpair.mat','file')
%                 load fsgfs_cvpair
%                 cvpair = cvpair';
%             else
%                 cvpair = randi(nD+1,nCvPair,2);
%             end
        elseif algo == fSGFS
            load cvpair25_gradCov
%             load cvpair25_paraCoef
%             smp = W_SG(1:s,:);
%             cvp = triu(abs(corrcoef(smp)));
% %             cvp = triu(abs(aCovW));
%             cvp(1:nD+2:(nD+1)^2) = 0;
%             % cvpair = randi(nD+1,nCvPair,2);
%             for i=1:nCvPair
%                 [dum,ix] = max(cvp(:));
%                 cvp(ix) = 0;
%                 [cvpair(i,1) cvpair(i,2)] = ind2sub([nD+1,nD+1],ix);            
%             end            
        else
            % do nothing
        end
    end
end

% elapsed time
end_t = cputime;
elaps_t = end_t - bg_t;
% end_t = toc;
% elaps_t = toc;
avgElaps_t = elaps_t/s

if exist('f2','var')
    saveas(f2,filename,'fig');
    saveas(f2,filename,'png');
end
% stop = stop
save(strcat(filename,'_SAMPLE','.mat'),'W_SG','s','avgElaps_t','aCovW','avgW','-v7.3');
disp('END SG');


end 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mode == BOTH || mode == HMC_ONLY

    if pcond == 1 % use stored preconditioning values
        disp('########## RUN HMC with Preconditioner ##########');   
        load 120206_114954_PRECOND
        load cvpair25
        M = aCovW;
        W_hmc = avgW;        
    elseif pcond == 2  % use what's generated previous sgfs run
        save(strcat(filename,'_precond','.mat'),'aCovW','avgW');
        load cvpair25
        M = aCovW;
        W_hmc = avgW;
    else    % no-preconditioning
        disp('########## RUN HMC ''without'' Preconditioner ##########');   
        load 120206_114954_PRECOND
        load cvpair25
        M = eye(nD+1);
        W_hmc = avgW; 
    end
    
    hmc_bin_lr_v02(dataset,Xt,YBt,pcond,lfNum,lfStep,...
               nBurn,nSmpl,M,W_hmc,laW,cvpair,mn_h,cv_h,...
               cvPlot,itvPrt,itvPlot,itvSmpl);
end
