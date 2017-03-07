function dWi = getGradientLR(X,T,W)

[N,D] = size(X);
X = [X ones(N,1)];

[nD,nO] = size(W);

%% feed-foward, softmax function
if nO == 1
    Y = sigmoid(X*W);
else
    Z = exp(X*W);
    Y = bsxfun(@rdivide,Z,sum(Z,2));  % softmax
end

%% 
df = T-Y;
dWi = zeros(nO,N,nD);
for i=1:nO
    dWi(i,:,:) = repmat(df(:,i),1,nD).*X;
end
if isnan(dWi)
	error('NaN');
end
dWi = permute(dWi,[2 3 1]);