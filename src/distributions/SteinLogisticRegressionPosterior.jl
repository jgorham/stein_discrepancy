# Bayesian logistic regression posterior abstract type
using StatsFuns: logistic

abstract SteinLogisticRegressionPosterior <: SteinPosterior

# each row of betas is assumed to be a beta sample
function loglikelihood(d::SteinLogisticRegressionPosterior,
                       beta::Array{Float64, 1};
                       idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    # condprobs is a N x b matrix
    condprobs = logistic(y .* (X * beta))
    sum(log(condprobs))
end

function gradloglikelihood(d::SteinLogisticRegressionPosterior,
                           beta::Array{Float64,1};
                           idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    # logitscores if passed to logit would be conditional probs
    logitscores = vec(y .* (X * beta))
    yX = y .* X

    gradients = logistic(-logitscores) .* yX
    vec(sum(gradients, 1))
end

function diaghessian(d::SteinLogisticRegressionPosterior,
                     beta::Array{Float64,1};
                     idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    # logitscores if passed to logit would be conditional probs
    logitscores = vec(y .* (X * beta))
    obsweights = logistic(-logitscores) .* logistic(logitscores)

    dhessian = sum(broadcast(*, X .^ 2, obsweights), 1)
    priordhessian = priordiaghessian(d, beta)

    vec(dhessian) + priordhessian
end
