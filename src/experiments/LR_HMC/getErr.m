function [err] = getErr(pred,label,type)

if strcmp(type,'RMSE')
    err = rmse(pred,label);
elseif strcmp(type,'MISS')
    err = missratio(pred,label);
else
    error('undefined error function type');
end