function lY = likelyLR(X,W)

[N,D] = size(X);
[nD,nO] = size(W);

X = [X ones(N,1)];

%% feed-foward
if nO == 1
    lY = sigmoid(X*W);
else    % softmax
    Z = exp(X*W);      
    lY = bsxfun(@rdivide,Z,sum(Z,2));
end
