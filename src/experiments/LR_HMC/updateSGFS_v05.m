% added sequential update for dSGFS
function [W,aCovW] = updateSGFS_v05(...
                            algo,dWi,dW,W,...
                            aCovW,nParW,it,laW,al,ga,Nt,n)                           

if algo == 1 % dSGFS

    tCovW = var(dWi,1)';    
    aCovW = (1-(1/it))*aCovW + (1/it)*tCovW;   
    
%     FI = ga*Nt*aCovW + laW.*ones(size(aCovW));                               
    FI = ga*Nt*aCovW; % By BCLT 
    G = (Nt/n)*dW - laW.*W;   

    BB = (al^2)*FI; 
    noise = randn(size(aCovW)).*sqrt(BB);
    update = 2*((G + noise)./(FI + BB));                                                     
   
    W = W + update;        
    
elseif algo == 2 % fSGFS
    
    tCovW = cov(dWi);
    aCovW = (1-(1/it))*aCovW + (1/it).*tCovW;

%     FI = ga*Nt*aCovW + laWI;
    FI = ga*Nt*aCovW; % by BCLT
    G = (Nt/n)*dW - laW.*W;

    if 0
        %% normal implementation
        BB = (al^2)*FI;
        if al == 0 
            noise = 0;
        else
            noise = mvnrnd(zeros(nParW,1),BB)';
        end
        update = 2*((FI + BB)\(G + noise));
    else
        %% using Cholesky Decomposition
        L = chol(FI,'lower');    
        if al == 0 
            noise = 0;
        else
            noise = al*L*randn(nParW,1);
        end
        update = 2*(1/(al^2+1))*(L'\(L\(G+noise)));
    end    
    
    W = W + update;    
        
else
    error('undefined SGFS type');
end
