function wts = weights_stein(x, model, varargin)
% Given a sample $x$ and the score function of distribution $q(x)$ (or an
% object of q for which @score_function is defined), provide a set of
% weights $w$ for variance reduction, such that the weighted sample approximate $q$ closely. 
%
%
% Qiang Liu
if isa(model, 'function_handle')
    score_q = model;
elseif isa(model, 'gmdistribution')
    score_q = @(x)score_function_gmm(model,x);
else
    score_q = @(x)score_function(model,x);    
end


%[ksdu2, pu2, bootsample2, infou2] = KSD_U_statistics(x, score_q, 'width',-1, 'nboot', 0, 'bootmethod', 'weighted');                          
[ksd, info] = KSD(x, score_q, varargin{:});


N = size(x,1);
wts = quadprog((info.M+info.M')/2,[], [],[],ones(1,N),1,zeros(N,1),[]);
%wts.U = quadprog(info.M-diag(diag(info.M)),[], [],[],ones(1,N),1,zeros(N,1),[]);
