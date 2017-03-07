function [Xt,Xv,Xs] = stdscale(Xt,Xv,Xs)

if nargin == 1
    X = Xt;
    [N,D] = size(X);
    Xs = (X - repmat(mean(X),N,1))./repmat(std(X),N,1);
    Xs(isnan(Xs)) = 0;    
elseif nargin ==2
    Nt = size(Xt,1);
    Nv = size(Xv,1);
    X = [Xt;Xv];   
    [N,D] = size(X);
    X = (X - repmat(mean(X),N,1))./repmat(std(X),N,1);
    X(isnan(X)) = 0;    
    Xt = X(1:Nt,:);
    Xv = X(Nt+1:Nt+Nv,:);        
elseif nargin ==3
    Nt = size(Xt,1);
    Nv = size(Xv,1);
    Ns = size(Xs,1);
    X = [Xt;Xv;Xs];   
    [N,D] = size(X);
    X = (X - repmat(mean(X),N,1))./repmat(std(X),N,1);
    X(isnan(X)) = 0;    
    Xt = X(1:Nt,:);
    Xv = X(Nt+1:Nt+Nv,:);        
    Xs = X(Nt+Nv+1:Nt+Nv+Ns,:);            
else
    error('error nargin');
end


