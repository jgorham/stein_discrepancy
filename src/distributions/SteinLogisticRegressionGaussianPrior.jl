# Bayesian logistic regression with independent Gaussian priors
using StatsFuns: logistic

type SteinLogisticRegressionGaussianPrior <: SteinLogisticRegressionPosterior
    X::Array{Float64, 2}
    y::Array{Float64, 1} # we assume the y = -1, 1
    priormu::Array{Float64, 1}
    priorvar::Array{Float64, 1}  # we assume components are independent
    c1::Float64
    c2::Float64
    c3::Float64
end

SteinLogisticRegressionGaussianPrior(X::Array{Float64, 2}, y::Array{Float64, 1}) =
SteinLogisticRegressionGaussianPrior(X, y,
                                     repmat([0.0], size(X, 2)),
                                     repmat([1.0], size(X, 2)))

# Constructor with canonical setting of (c1,c2,c3) = (1,1,1)
SteinLogisticRegressionGaussianPrior(X::Array{Float64, 2},
                                     y::Array{Float64, 1},
                                     priormu::Array{Float64, 1},
                                     priorvar::Array{Float64, 1}) =
SteinLogisticRegressionGaussianPrior(X, y, priormu, priorvar,
                                     1.0, 1.0, 1.0)

function logprior(d::SteinLogisticRegressionGaussianPrior,
                  beta::Array{Float64, 1})
    # Independent Gaussian prior for each component
    logpriorvalue =
        -0.5 * sum((beta .- d.priormu') .^ 2 ./ d.priorvar') -
        -0.5 * sum(log(2 * pi .* d.priorvar))
    logpriorvalue
end

function gradlogprior(d::SteinLogisticRegressionGaussianPrior,
                      beta::Array{Float64, 1})
    -(beta .- d.priormu) ./ d.priorvar
end

function numdimensions(d::SteinLogisticRegressionGaussianPrior)
    length(d.priormu)
end

function priordiaghessian(d::SteinLogisticRegressionGaussianPrior,
                          beta::Array{Float64, 1})
    -1.0 ./ d.priorvar
end
