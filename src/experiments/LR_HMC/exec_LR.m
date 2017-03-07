% dataset = 'mnist';
% dataset = 'abalone_bin';
% dataset = 'poker_tst_bin';
dataset = 'mnist_79';

% setting = 'LR_SGD';
% setting = 'LR_SGLD';
% setting = 'LR_DSGFS';
setting = 'LR_FSGFS';

sd = '1';

if strcmp(setting,'LR_SGD')
    
    fdim='50'; n_g='500'; n_c='500'; n_p='0';
    algo='0'; laW='0'; al='0'; nSmpl='500000';
    bgst='0.001'; edst='1e-3'; 
    brn='0';lr='1';
    itvPlot='10'; itvSmpl='10'; itvSave='20000';  
    cvPlt='0';
        
elseif strcmp(setting,'LR_SGLD')
    
    fdim='50'; n_g='500'; n_c='500'; n_p='0';
lbrlhdifnbvjkvhrgdn    algo='3'; laW='1'; al='0'; 
    nSmpl='10000'; bgst='0.005'; edst='1e-1'; 
    brn='2000';lr='1';
    itvPlot='200'; itvSmpl='1'; itvSave='20000';  
    cvPlt='1';
    
elseif strcmp(setting,'LR_DSGFS')
    
    fdim='50'; n_g='500'; n_c='500'; n_p='0';
    algo='1'; laW='1'; al='0'; 
    nSmpl='10000'; bgst='0.001'; edst='1e-2'; 
    brn='2000'; lr='0';
    itvPlot='1000'; itvSmpl='1'; itvSave='2000000';
    cvPlt='0';

elseif strcmp(setting,'LR_FSGFS')
    
    fdim='50'; n_g='500'; n_c='500'; n_p='0';
    algo='2'; laW='1'; al='0'; 
    nSmpl='10000'; bgst='0.0001'; edst='1e-2'; 
    brn='2000'; lr='0'; 
    itvPlot='1000'; itvSmpl='1'; itvSave='1000000';
    cvPlt='0'; 
else
    error('undefined settings');
end

SGX = '1';
HMC = '2';
BOTH = '3';
mode = HMC; % 1:sg**-only, 2:hmc-only, 3-both
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,initW] = run_sgfs_hmc_lr_v01(dataset,sd,fdim,n_g,n_c,n_p,algo,laW,al,...
    nSmpl,bgst,edst,brn,lr,itvPlot,itvSmpl,itvSave,cvPlt,mode);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
