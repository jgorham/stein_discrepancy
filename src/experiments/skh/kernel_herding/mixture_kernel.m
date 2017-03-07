function z = mixture_kernel(X,means,weights,sigma2)
% z = mixture_kernel(X,means,weights,sigma2)
% Return evaluation of a mixture of RBF kernel evaluations...
%
% K = number of components of kernel; d = dimension of input space; N =
% number of points to evaluate
%
% X = Nxd matrix of points to evaluate the mixture kernel
% means = Kxd -- position of each RBF bump
% weights = 1xK vector of weights for each RBF kernel
% sigma2 (optional) = parameter of RBF kernel (1 by default) -- sigma squared
%
% Z_i = \sum_k weights(k) RBF(means(k,:),X(i,:))
if nargin < 4
    sigma2 = 1;
end

z = zeros(size(X,1),1);

for k = 1:length(weights)
    difference_vec = bsxfun(@minus, X, means(k,:)); 
    distance_square = sum(difference_vec.*difference_vec,2);
    z = z+weights(k)*exp(-1/(2*sigma2)*(distance_square));
end