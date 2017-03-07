# Bayesian huber loss regression w Gaussian priors

type SteinHuberLossRegressionGaussianPrior <: SteinHuberLossRegressionPosterior
    X::Array{Float64, 2}  # n by p matrix of regressors
    y::Array{Float64, 1}
    delta::Float64        # the scaling parameter for Huber loss
    priormu::Array{Float64, 1}
    priorvar::Array{Float64, 1}  # we assume components are independent
    c1::Float64
    c2::Float64
    c3::Float64
end

SteinHuberLossRegressionGaussianPrior(X::Array{Float64, 2}, y::Array{Float64, 1}) =
SteinHuberLossRegressionGaussianPrior(X, y, 1.0,
                                      repmat([0.0], size(X, 2)),
                                      repmat([1.0], size(X, 2)))

SteinHuberLossRegressionGaussianPrior(X::Array{Float64, 2}, y::Array{Float64, 1}, delta::Float64) =
SteinHuberLossRegressionGaussianPrior(X, y, delta,
                                      repmat([0.0], size(X, 2)),
                                      repmat([1.0], size(X, 2)))

# Constructor with canonical setting of (c1,c2,c3) = (1,1,1)
SteinHuberLossRegressionGaussianPrior(X::Array{Float64, 2},
                                      y::Array{Float64, 1},
                                      delta::Float64,
                                      priormu::Array{Float64, 1},
                                      priorvar::Array{Float64, 1}) =
SteinHuberLossRegressionGaussianPrior(X, y, delta, priormu, priorvar,
                                      1.0, 1.0, 1.0)

function logprior(d::SteinHuberLossRegressionGaussianPrior,
                  beta::Array{Float64, 1})
    # Independent Gaussian prior for each component
    logpriorvalue =
        -0.5 * sum((beta .- d.priormu) .^ 2 ./ d.priorvar) -
        -0.5 * sum(log(2 * pi .* d.priorvar))
    logpriorvalue
end

function gradlogprior(d::SteinHuberLossRegressionGaussianPrior,
                      beta::Array{Float64, 1})
    -(beta .- d.priormu) ./ d.priorvar
end

function numdimensions(d::SteinHuberLossRegressionGaussianPrior)
    length(d.priormu)
end
