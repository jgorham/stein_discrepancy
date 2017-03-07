function [X Y] = genSyn2dData(N,type,var)

rng = 1.7;

if strcmp(type,'nonlinsep')  
    x1 = -rng:(rng*2)/N:rng;
    x2 = 3*cos(x1) + var*rand(1,length(x1));    
    X1 = [x1' x2' ones(length(x1),1)];    
    X2 = [var*randn(length(x1),1) var*randn(length(x1),1)+1 (-1)*ones(length(x1),1)];    
    X = [X1;X2];
elseif strcmp(type,'circle')
    Nk = ceil(N/2);
    x1 = -rng:(rng*2)/Nk:rng;
    x2 = 2*cos(x1) + var*rand(1,length(x1));    
    X1 = [repmat(x1',2,1) [x2';-x2'] ones(2*length(x1),1)];    
    X2 = [var*randn(length(x1),2) (-1)*ones(length(x1),1)];    
    X = [X1;X2];
elseif strcmp(type,'linsep')
    Nk = ceil(N/2);
    X1 = [var*randn(Nk,2)-1 (1)*ones(Nk,1)];
    X2 = [var*randn(Nk,2)+1 (-1)*ones(Nk,1)];     
    X = [X1;X2];
elseif strcmp(type,'overlap_x1')
    Nk = ceil(N/2);
    X1 = [var*randn(Nk,1)-2 var*randn(Nk,1)-1 (1)*ones(Nk,1)];
    X2 = [var*randn(Nk,1)+2 var*randn(Nk,1) (-1)*ones(Nk,1)]; 
    X = [X1;X2];
elseif strcmp(type,'2x2')
    Nk = ceil(N/4);
    X1 = [var*randn(Nk,1)-3 var*randn(Nk,1)-3 (1)*ones(Nk,1)];
    X2 = [var*randn(Nk,1)+3 var*randn(Nk,1)-3 (-1)*ones(Nk,1)]; 
    X3 = [var*randn(Nk,1)+3 var*randn(Nk,1)+3 (1)*ones(Nk,1)]; 
    X4 = [var*randn(Nk,1)-3 var*randn(Nk,1)+3 (-1)*ones(Nk,1)]; 
    X = [X1;X3;X2;X4];
elseif strcmp(type,'0502max')
    Nk = ceil(N/4);
    off = 1.3;
    X1 = [sqrt(var)*randn(Nk,1)-(sqrt(var)+off) sqrt(var)*randn(Nk,1)-(sqrt(var)+off) (1)*ones(Nk,1)];
    X2 = [sqrt(var)*randn(Nk,1)+(sqrt(var)+off) sqrt(var)*randn(Nk,1)-(sqrt(var)+off) (-1)*ones(Nk,1)]; 
    X3 = [sqrt(var)*randn(Nk,1)+(sqrt(var)+off) sqrt(var)*randn(Nk,1)+(sqrt(var)+off) (1)*ones(Nk,1)]; 
    X4 = [sqrt(var)*randn(Nk,1)-(sqrt(var)+off) sqrt(var)*randn(Nk,1)+(sqrt(var)+off) (-1)*ones(Nk,1)]; 
    X = [X1;X3;X2;X4];    
elseif strcmp(type,'DataForMoNNE')
    Nk = ceil(N/16);
    off = 1.3;
    dist = 13;
    X1 = [sqrt(var)*randn(Nk,1)-(sqrt(var)+off) sqrt(var)*randn(Nk,1)-(sqrt(var)+off) (1)*ones(Nk,1)];
    X2 = [sqrt(var)*randn(Nk,1)+(sqrt(var)+off) sqrt(var)*randn(Nk,1)-(sqrt(var)+off) (-1)*ones(Nk,1)]; 
    X3 = [sqrt(var)*randn(Nk,1)+(sqrt(var)+off) sqrt(var)*randn(Nk,1)+(sqrt(var)+off) (1)*ones(Nk,1)]; 
    X4 = [sqrt(var)*randn(Nk,1)-(sqrt(var)+off) sqrt(var)*randn(Nk,1)+(sqrt(var)+off) (-1)*ones(Nk,1)]; 
    X_o = [X1;X2;X3;X4];    
    X_a = X_o + repmat([-dist dist 0], size(X_o,1),1);
    X_b = X_o + repmat([dist dist 0], size(X_o,1),1);    
    X_c = X_o + repmat([dist -dist 0], size(X_o,1),1);
    X_d = X_o + repmat([-dist -dist 0], size(X_o,1),1);
    X = [X_a; X_b; X_c; X_d];
    
elseif strcmp(type,'rnd')
    Nk = ceil(N/4);
    X1 = [var*randn(Nk,1) var*randn(Nk,1) (1)*ones(Nk,1)];
    X2 = [var*randn(Nk,1) var*randn(Nk,1) (-1)*ones(Nk,1)]; 
    X3 = [var*randn(Nk,1) var*randn(Nk,1) (1)*ones(Nk,1)]; 
    X4 = [var*randn(Nk,1) var*randn(Nk,1) (-1)*ones(Nk,1)]; 
    X = [X1;X3;X2;X4];
elseif strcmp(type,'minimal')
    Nk = ceil(N/4);
    X1 = [zeros(Nk,1)+1.1 zeros(Nk,1)+1.5 (1)*ones(Nk,1)];
    X2 = [zeros(Nk,1)+1.7 zeros(Nk,1)+1.4 (-1)*ones(Nk,1)]; 
    X3 = [zeros(Nk,1)+1.5 zeros(Nk,1)+1.1 (1)*ones(Nk,1)]; 
    X4 = [zeros(Nk,1)-1 zeros(Nk,1)-1 (-1)*ones(Nk,1)]; 
    X = [X1;X3;X2;X4];    
elseif strcmp(type,'2x1')
    Nk = ceil(N/3);
    X1 = [var*randn(Nk,1)+2 var*randn(Nk,1) 1*ones(Nk,1)];
    X2 = [var*randn(Nk,1)-2 var*randn(Nk,1)+2 -1*ones(Nk,1)];
    X3 = [var*randn(Nk,1)-2 var*randn(Nk,1)-2 -1*ones(Nk,1)];
    X = [X1;X2;X3];
    X(:,1:2) = X(:,1:2) + var*randn(Nk*3,2);    
else
    error('undefined type');
end 

[~,D] = size(X);
Y = X(:,D);   
X = X(:,1:2);
% XA = X(Y==1,:);
% XB = X(Y==-1,:); 
% X = [XA;XB];

end