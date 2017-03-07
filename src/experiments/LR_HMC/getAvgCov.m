function newCov = getAvgCov(avgCov,curCov,avgType,it,memRate)

if strcmp(avgType,'norm')
    newCov = (1-(1/it))*avgCov + (1/it)*curCov;
elseif strcmp(avgType,'exp')
    newCov = (1-memRate)*avgCov + memRate*curCov;
else
    error('undefined avgType');
end