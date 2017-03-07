function cvpair = getMaxCovPairs(aCovW,nCvPair,nD)

cvpair = zeros(nCvPair,2);
cv = triu(abs(aCovW));
cv(1:nD+1:nD^2) = 0;
for i=1:nCvPair
    [dum,ix] = max(cv(:));
    cv(ix) = 0;
    [cvpair(i,1) cvpair(i,2)] = ind2sub([nD,nD],ix);            
end