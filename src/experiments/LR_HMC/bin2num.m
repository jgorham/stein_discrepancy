function [Yt,Yv,Ys] = bin2num(YBt,YBv,YBs)

if nargin == 1
    [Yt,row] = find(YBt'==1);
elseif nargin == 2
    [Yt,row] = find(YBt'==1);
    [Yv,row] = find(YBv'==1);
elseif nargin == 3
    [Yt,row] = find(YBt'==1);
    [Yv,row] = find(YBv'==1);
    [Ys,row] = find(YBs'==1);
end
