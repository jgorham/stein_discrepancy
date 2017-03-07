function [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM)

[N,D] = size(X);

% mix the data
if PERM
    perm = randperm(N);
    X = X(perm,:);
    Y = Y(perm,:);
end

Nt = ceil(N*tvRatio);
Xt = X(1:Nt,:);
Xv = X(Nt+1:end,:);
Yt = Y(1:Nt,:);
Yv = Y(Nt+1:end,:);    

