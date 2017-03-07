function [Xt,Yt,Xv,Yv] = loadMnist79()

load('mnist_all.mat');
XtA = double(train7);
XtB = double(train9);
XvA = double(test7);
XvB = double(test9);

%% scaling
X = [XtA;XtB;XvA;XvB];
[N,D] = size(X);

% train data
Xt = X(1:size(XtA,1)+size(XtB,1),:);
[Nt,D] = size(Xt);
Yt = [ones(size(XtA,1),1); zeros(size(XtB,1),1)];

% test data 
Xv = X(Nt+1:end,:);
[Nv,D] = size(Xv);
Yv = [ones(size(XvA,1),1); zeros(size(XvB,1),1)];


