function hmc_bin_lr_v02(dataset,X,Y,pcond,nLeapfrog,leapfrogStep,nBurn,nSmpl,...
                    M,W,la,cvpair,...
                    mn_ref,cv_ref,...
                    cvPlot,itvPrt,itvPlot,itvSmpl)

%close all
dbstop if error

%% str2double
% nBurn = str2double(aBurn);
% nSmpl = str2double(aNsmpl);
mxit = nBurn + nSmpl;
% la = str2double(aLa);
% itvPlot = str2double(aItvPlot);
% itvSmpl = str2double(aItvSmpl);
% itvSave = str2double(aItvSave);

%% HMC params
eps = 1e-5;
% nO = size(Y,2);
[N,nD] = size(X);

%% inverse conditioner
M = N*M + la*diag(ones(nD+1,1));
iM = inv(M);
sM = sqrtm(M);

%% store samples
W_HMC = zeros(nSmpl,nD+1);
% ET_HMC = zeros(nSmpl,1);        % store elapsed time
AR = zeros(nBurn+nSmpl,1);

%% set filename
str = sprintf('HMC_%s_M%s_nLf%s_lfs%s_brn%s_nsmpl%s_la%s_itv%s_%s',...
                dataset,num2str(pcond),num2str(nLeapfrog),num2str(leapfrogStep),...
                num2str(nBurn),num2str(nSmpl),num2str(la),...
                num2str(itvPlot),num2str(itvSmpl));
str = strrep(str,'.','_');
filename = getFilename(str)

%% figures
if itvPlot
%     f1 = figure('name',filename);
%     set(f1,'windowStyle','docked');
%     title('1D-Plot');    

    if cvPlot
        f2 = figure(20);
        set(f2,'name','2d-cov','windowStyle','docked');
%         title(strrep(filename,'_','-'));   
    end
%     f2 = figure('name',filename);
%     set(f2,'windowStyle','docked');
%     title('AcceptRate');    
%     
%     f3 = figure('name',filename);
%     set(f3,'windowStyle','docked');
%     title('2D-CovPlot');    
end


%% run HMC iteration
X = [X ones(length(X),1)];  % add bias term for lr
T = Y;

s = 0;
ar = 0;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p = sM*randn(nD+1,1);  % init momenta
for it=1:mxit

    if it == nBurn+1
        bg_t = cputime; % to measure time for 1 iteration
    end
    
    ga = 1/it;    
    p = sM*randn(nD+1,1);  % init momenta
    
    S = sigmoid(X*W);    
    Ex = - sum(T.*log(S+eps) + (1-T).*log(1-S+eps)) + 0.5*la*(W'*W);
    Hx = Ex + 0.5*(p'*iM*p);
    
    Wt = W;
    
    for i=1:nLeapfrog

        Y = sigmoid(X*Wt);
        dWt = repmat((T-Y),1,nD+1).*X;
        dEt = -sum(dWt)' + la*Wt;
        
        p = p - 0.5*leapfrogStep*dEt;
        Wt = Wt + leapfrogStep*iM*p;

        Y = sigmoid(X*Wt);
        dWt = repmat((T-Y),1,nD+1).*X;
        dEt = -sum(dWt)' + la*Wt;
        
        p = p - 0.5*leapfrogStep*dEt;
        
    end
    
    S = sigmoid(X*Wt);    
    Ey = - sum(T.*log(S+eps) + (1-T).*log(1-S+eps)) + 0.5*la*(Wt'*Wt);
    Hy = Ey + 0.5*p'*iM*p;
    
    DH = Hx - Hy;
    P = min(1,exp(-DH));
    A = rand < P;
    
    
    %% Accept-Reject 
    if A == 1
        W = Wt;
        if it > nBurn
            s = s + 1;
            W_HMC(s,:) = W';            
        else
            % adapt learning lfstep during burn-in
            leapfrogStep = 1.01*leapfrogStep;
        end
    else
        if it > nBurn
            s = s + 1;
            W_HMC(s,:) = W';            
        else
            % adapt learning lfstep during burn-in
            leapfrogStep = 0.9*leapfrogStep;
        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Update acceptance rate
    ar = (1-ga)*ar + ga*A;
    AR(it) = ar;

    %% print
    if itvPrt && ~rem(it+itvPrt,itvPrt)
        it
        ar
        leapfrogStep
        if s > 0
            elaps_t = cputime - bg_t; 
            avgElaps_t = elaps_t/s
        end
    end
    
    %% plot
    if itvPlot && rem(it,itvPlot) == 0 
        
%         set(0,'CurrentFigure',f1);
%         hold on;
%         mn = mean(W_HMC(:,1:s),2);
%         for i=1:nD+1
%             subplot(1,nD+1,i);
%             plot(nBurn+1:nBurn+s,W_HMC(i,1:s),'b.','markersize',0.1);hold on;
%             plot(nBurn+1:nBurn+s,repmat(mn(i),1,s),'r','LineWidth',2);
%         end
        
%         if exist('f2','var');
%             set(0,'Currentfigure',f2);
%             plot(1:it,AR(1:it),'r-');
%             drawnow;        
%         end
        
        if cvPlot && s > 1            
%             set(0,'CurrentFigure',f3);
            set(0,'CurrentFigure',20);
            clf;
            nCvPair = length(cvpair);
            for i=1:nCvPair
                subplot(ceil(sqrt(nCvPair)),ceil(sqrt(nCvPair)),i);
                j = cvpair(i,1);
                k = cvpair(i,2);
                mn = mean([W_HMC(1:s,j) W_HMC(1:s,k)],1);
                cv = cov([W_HMC(1:s,j) W_HMC(1:s,k)]);                                        
                plot(W_HMC(1:s,j),W_HMC(1:s,k),'.k','markersize',0.1); hold on;
                plotGauss(mn_ref(j),mn_ref(k),cv_ref(j,j),cv_ref(k,k),cv_ref(j,k),'-b');
                plotGauss(mn(1),mn(2),cv(1,1),cv(2,2),cv(1,2),'-r');
                plot(mn_ref(j),mn_ref(k),'bo','linewidth',2,'markersize',10);
                plot(mn(1),mn(2),'r+','linewidth',2,'markersize',10);            
                axis([mn_ref(j)-0.2 mn_ref(j)+0.2 mn_ref(k)-0.2 mn_ref(k)+0.2]);
%                 axis auto                
            end 
            drawnow;
        end    
    end
end
elaps_t = cputime - bg_t; 
avgElaps_t = elaps_t/s
%stop = stop
save(strcat(filename,'_SAMPLE','.mat'),'W_HMC','avgElaps_t','ar','s','-v7.3');

