
function s = sample_unif(N,n,method)

%function s = sample_unif(N,n,method)
%
% INPUT:
% N = range of integeres to sample from 1:N
% n = number of samples
% method = 1 --> with replacement
% method = 2 --> without replacement
% 
% OUTPUT:
% s = samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    
    method = 1;
    
end

if nargin < 2
    
    n = 1;
    
end

if method == 1
    
    s = ceil(N*rand(1,n));
    
elseif method == 2
    
    R = randperm(N);
    
    s = R(1:n);
    
end