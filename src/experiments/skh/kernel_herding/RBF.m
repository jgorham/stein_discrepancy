function z = RBF(X,mean,sigma2)
% z = RBF(X,mean,sigma2)
% Return RBF kernel evaluation at points in X.
% i.e. z(i,1) = RBF_kernel(X(i,:), mean) for i = 1:N
% where RBF_kernel(x,y) = exp(-1/(2*sigma2) ||x-y||^2)
%
% d = dimension of input space; N = number of points to evaluate
%
% X = Nxd matrix of points to evaluate the RBF kernel
% mean = 1xd -- position of RBF bump
% sigma2 (optional) = parameter of RBF kernel (1 by default) -- sigma squared
%
if nargin < 3
    sigma2 = 1;
end

difference_vec = bsxfun(@minus, X, mean); 
distance_square = sum(difference_vec.*difference_vec,2);
z = exp(-1/(2*sigma2)*(distance_square));