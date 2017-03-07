function lr = getLearningRate(it,mxit,stRate,edRate,smooth)

%%
if smooth == 0  % linear decaying
    decr = (stRate-edRate)/mxit;
    lr = stRate - it*decr;
elseif stRate == edRate  % fixed learning rate   
    lr = edRate;
else % polynomial decaying
    R = edRate/stRate;
    t0 = mxit*(R^(1/smooth))/(1-R^(1/smooth));
    lambda = stRate*t0^smooth;
    lr = lambda./(t0+it).^smooth;
end

%%
if isnan(lr)
    error('NaN in learning rate');
end

end