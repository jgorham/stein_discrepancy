% dataset = 'mnist';
% dataset = 'abalone_bin';
% dataset = 'poker_tst_bin';
dataset = 'mnist_79';

% setting = 'LR_SGD';
% setting = 'LR_SGLD';
setting = 'LR_DSGFS';
% setting = 'LR_FSGFS';

sd = '1';

if strcmp(setting,'LR_SGD')
    
    fdim='50'; n='500'; 
    algo='0'; laW='0'; al='0'; nSmpl='500000';
    bgst='0.001'; edst='1e-3'; 
    brn='0';lr='1';
    itvPrt='1000'; itvPlot='10'; 
    itvSmpl='10'; itvSave='20000';  
    cvPlt='0';
        
elseif strcmp(setting,'LR_SGLD')
    
    fdim='50'; n='500'; 
    algo='3'; laW='1'; al='0'; 
    nSmpl='1000000'; bgst='0.0001'; edst='1e-0'; 
    brn='1';lr='1';
    itvPrt='100'; itvPlot='100'; 
    itvSmpl='1'; itvSave='2000000'; 
    cvPlt='1';
    
elseif strcmp(setting,'LR_DSGFS')
    
    fdim='50'; n='500'; 
    algo='1'; laW='1'; al='0';
    nSmpl='100000'; bgst='0.001'; edst='1e-2'; 
    brn='1'; lr='0';
    itvPrt='10'; itvPlot='10'; 
    itvSmpl='1'; itvSave='2000000';
    cvPlt='0';

elseif strcmp(setting,'LR_FSGFS')
    
    fdim='50'; n='500'; 
    algo='2'; laW='1'; al='0'; 
    nSmpl='100000'; bgst='0.0001'; edst='1e-2'; 
    brn='5000'; lr='0'; 
    itvPrt='100'; itvPlot='10'; 
    itvSmpl='1'; itvSave='1000000';
    cvPlt='0'; 
    
else
    error('undefined settings');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SGX = '1';
HMC = '2';
BOTH = '3';

mode = SGX;
% mode = HMC;
lfStep='0.1'; lfNum='40'; precond='0';

if mode == HMC
    brn='10'; nSmpl='100000';     
    itvPrt='100'; itvPlot='0'; 
    itvSmpl='1';
    cvPlt='0'; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,initW] = run_sgfs_hmc_lr_v02(...
    dataset,sd,fdim,n,algo,laW,al,...
    nSmpl,bgst,edst,brn,lr,...
    lfStep,lfNum,precond,...
    itvPrt,itvPlot,itvSmpl,itvSave,cvPlt,mode);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
