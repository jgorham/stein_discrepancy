function [X,compIdx] = qrandgm(prob, N,qstream)
% function [X,compIdx] = qrandgm(prob, N,qstream)
% X = N x d quasi-random vector from the Gaussian mixture model
%   given by GM object prob.
%
% Optional input: qstream is a quasi-random stream of dimension d+1.
%   (This is to be able to make multiple calls with the same stream.  
%    If none is provided, a default new Sobol stream is used.)
%
% compIdx are the component indices for each point.
%
% ASSUME FULL COVARIANCE...

mu_mix = prob.mu;
Sigma_mix = prob.Sigma; % ASSUME FULL COVARIANCE
pi_prob = prob.PComponents;
d = size(mu_mix,2);
K = size(mu_mix,1);

if nargin < 3
    qstream = qrandstream('sobol',d+1,'skip',100);
    % note that skip at least 1st point (which is (0,0,0) etc. as this doesn't
    % work well in inverse transform!!! (gives -inf value!)
end

% get quasi-random numbers:
qR = qrand(qstream,N); % we use last dimension for the discrete component
if size(qR,2) ~= (d+1)
    error('Please pass a quasi-random stream of dimension d+1\n')
end

%% sample the components:
% get cdf:
discrete_cdf = cumsum(pi_prob); % ROW VECTOR
assert(size(discrete_cdf,1) == 1); % checking convention!
% invert it:
check = bsxfun(@le, qR(:,d+1), discrete_cdf);
% look for first non-zero index in each row:
[dummy, compIdx] = max(check,[], 2); % row by row, return *first* one...

%% sample the standard normal:

Xstand = norminv(qR(:,1:d)); % each row is a d dim standard normal

% make transformation to each variable with correct component matrix:
X = zeros(N,d);

for k = 1:K
    % make the Cholesky decomposition:
    [L,p] = chol(Sigma_mix(:,:,k),'lower');
    if(~(p==0) )
        % FL: N.B. Used to get QMC to work with UAV application, not a
        % general solution!!!!
        Ltmp = zeros(d);
        Ltmp(1:p-1,1:p-1) = L;
        L = Ltmp;
    end
    mu = mu_mix(k,:);
    mbrs = find(compIdx == k);
    X(mbrs,:) = bsxfun(@plus, Xstand(mbrs,:)*L', mu); % note the row by row convention...
    % in vetor, we have y = L*x+mu gives correct distribution -- 
    % but here needs to transpose everything due to row-by-row convention
    % for X...
end
    