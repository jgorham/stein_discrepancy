function [ret] = sigmoid(a)
    ret = 1./(1+exp(-a));
end