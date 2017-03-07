function y=score_function_gmm(model,X)

P = posterior(model,X);
d = size(X,2);
n = size(X,1);
y = 0;
for k = 1:model.NComponents
    sigma = model.Sigma(:,:,min(size(model.Sigma,3), k));    
    y = y +  repmat(P(:,k), 1, d) .* ((-X + repmat(model.mu(k,:), n,1))/sigma);
end

