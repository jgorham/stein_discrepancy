function Sigma  = random_covariance(evalues)
% function Sigma = random_covariance(evalues)
% return a random covariance matrix with determined e-values
%
% INPUT: evalues - a dx1 vector of *positive* e-valuee -- this represents
%    the variances of the distribution in the suitable basis
%
% OUTPUT: a randomly rotated version of this matrix: i.e. let D be the
%    the diagonal matrix with e-values; we return Q*D*Q' where Q is some
%    uniformly distributed orthogonal matrix.
%   
%   Q is obtained by sampling a dxd matrix X with iid standard normal entries
%   and then doing the QR decomposition of X with R having a positive
%   diagonal
%   (see [Steward 1980 -- The Efficient Generation of Random Orthogonal
%   MAtrices with an Application to Condition Estimators -- the method I
%   used is the inefficient one mentioned in the paper])

d = length(evalues);
if any(evalues <= 0)
    error('Please only use stricly positive diagonal for a valid covariance matrix\n')
end
%%
D = diag(evalues);

%%
X = randn(d,d);
[Q,R] = qr(X);

% making sure diagonal of R is positive (depends on Matlab version whether
% this is normally enforced):
s = sign(diag(R));
Q = Q*diag(s);
R = diag(s)*R;

Sigma = Q*D*Q';

end

