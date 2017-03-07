function [err] = missratio(pred,label)

N = size(pred,1);
nO = size(pred,2);

if nO == 1
    iZ = pred > 0.5;
    iY = label;
else
    [iC,iZ] = max(pred');
    [iC,iY] = max(label');
end
err = sum(iZ ~= iY)./N; 