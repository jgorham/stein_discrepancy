function [YBt,YBv,YBs] = num2bin(Yt,Yv,Ys)

if nargin == 2
    Yt = uint8(Yt);
    Yv = uint8(Yv);    
    [Nt,D] = size(Yt);
    [Nv,D] = size(Yv);
    
    if D ~= 1
        error('Dim must be 1');
    end

    cls = unique([Yt;Yv]);
    nC = length(cls);
    
    YBt = zeros(Nt,nC);
    YBv = zeros(Nv,nC);
    
    %% assume that Yt starts from 0
    
    for i=1:Nt
        YBt(i,:) = (Yt(i)==cls)';
    end
    for i=1:Nv
        YBv(i,:) = (Yv(i)==cls)';
    end
    
elseif nargin == 3
    Yt = uint8(Yt);
    Yv = uint8(Yv);
    Ys = uint8(Ys);
    [Nt,D] = size(Yt);
    [Nv,D] = size(Yv);
    [Ns,D] = size(Ys);
    
    if D ~= 1
        error('Dim must be 1');
    end

    cls = unique([Yt;Yv;Ys]);
    nC = length(cls);
    
    YBt = zeros(Nt,nC);
    YBv = zeros(Nv,nC);
    YBs = zeros(Ns,nC);
    
    %% assume that Yt starts from 0
    for i=1:Nt
        YBt(i,:) = (Yt(i)==cls)';
    end
    for i=1:Nv
        YBv(i,:) = (Yv(i)==cls)';
    end
    for i=1:Ns
        YBs(i,:) = (Ys(i)==cls)';
    end
    
else
    error('error in nargin');
end