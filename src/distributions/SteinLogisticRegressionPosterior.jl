# Bayesian logistic regression posterior abstract type
using StatsBase: logistic

abstract SteinLogisticRegressionPosterior <: SteinPosterior

# each row of betas is assumed to be a beta sample
function loglikelihood(d::SteinLogisticRegressionPosterior,
                       betas::Array{Float64, 2};  # b x p matrix
                       idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    # condprobs is a N x b matrix
    condprobs = logistic(y .* (X * betas'))
    log(condprobs')
end

function logdensity(d::SteinLogisticRegressionPosterior,
                    betas::Array{Float64, 2};  # b x p matrix
                    idx=1:length(d.y))
    logpriorvalue = logprior(d, betas)
    loglikelihoods = loglikelihood(d, betas; idx=idx)
    totalloglikelihood = sum(loglikelihoods, 2)
    vec(logpriorvalue + totalloglikelihood)
end

function gradloglikelihood(d::SteinLogisticRegressionPosterior,
                           betas::Array{Float64, 2};
                           idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    # logitscores if passed to logit would be conditional probs
    logitscores = y .* (X * betas')
    yX = y .* X
    gradients = [
        sum(logistic(-logitscores[:, i]) .* yX, 1)
        for i=1:size(betas, 1)
    ]
    # unpack the list of lists into a matrix
    vcat(gradients...)
end

function gradlogdensity(d::SteinLogisticRegressionPosterior,
                        betas::Array{Float64, 2};
                        idx=1:length(d.y))
    N, p = size(d.X)
    batchratio = N / length(idx)
    gradlogpriorvalue = gradlogprior(d, betas)
    empgradloglikelihood = gradloglikelihood(d, betas; idx=idx)
    gradlogpriorvalue + batchratio .* empgradloglikelihood
end

function loglikelihood(d::SteinLogisticRegressionPosterior,
                       beta::Array{Float64, 1};
                       kwargs...)
    vec(loglikelihood(d, beta'; kwargs...)')
end

function logprior(d::SteinLogisticRegressionPosterior,
                  beta::Array{Float64, 1})
    logprior(d, beta')[1]
end

function logdensity(d::SteinLogisticRegressionPosterior,
                    beta::Array{Float64, 1};
                    kwargs...)
    logdensity(d, beta'; kwargs...)[1]
end

function gradlogdensity(d::SteinLogisticRegressionPosterior,
                        beta::Array{Float64, 1};
                        kwargs...)
    vec(gradlogdensity(d, beta'; kwargs...))
end

function gradloglikelihood(d::SteinLogisticRegressionPosterior,
                           beta::Array{Float64, 1};
                           kwargs...)
    vec(gradloglikelihood(d, beta'; kwargs...))
end

function gradlogprior(d::SteinLogisticRegressionPosterior,
                      beta::Array{Float64, 1};
                      kwargs...)
    vec(gradlogprior(d, beta'; kwargs...))
end
