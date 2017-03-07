function err = rmse(p,t)

[N,D] = size(p);
dist = sqrt(sum((p-t).^2,2));
err = sqrt(sum(dist)/(N*D));


