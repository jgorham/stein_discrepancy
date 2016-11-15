# Bayesian student-t multivariate regression with independent Gaussian priors

type SteinMultivariateStudentTRegressionGaussianPrior <: SteinMultivariateStudentTRegressionPosterior
    X::Array{Float64, 2}
    y::Array{Float64, 1}
    nu::Float64  # the number of degrees of freedom for the t-distribution
    sigma::Array{Float64, 2} # the covariance of the multivariate t-distribution
    priormu::Array{Float64, 1}
    priorvar::Array{Float64, 1}  # we assume components are independent
    c1::Float64
    c2::Float64
    c3::Float64
end

SteinMultivariateStudentTRegressionGaussianPrior(X::Array{Float64, 2},
                                                 y::Array{Float64, 1},
                                                 nu::Float64) =
SteinMultivariateStudentTRegressionGaussianPrior(X, y, nu, eye(size(X, 1)))

SteinMultivariateStudentTRegressionGaussianPrior(X::Array{Float64, 2},
                                                 y::Array{Float64, 1},
                                                 nu::Float64,
                                                 sigma::Array{Float64, 2}) =
SteinMultivariateStudentTRegressionGaussianPrior(X, y, nu, sigma,
                                                 repmat([0.0], size(X, 2)),
                                                 repmat([1.0], size(X, 2)))

# Constructor with canonical setting of (c1,c2,c3) = (1,1,1)
SteinMultivariateStudentTRegressionGaussianPrior(X::Array{Float64, 2},
                                                 y::Array{Float64, 1},
                                                 nu::Float64,
                                                 sigma::Array{Float64, 2},
                                                 priormu::Array{Float64, 1},
                                                 priorvar::Array{Float64, 1}) =
SteinMultivariateStudentTRegressionGaussianPrior(X, y, nu, sigma, priormu, priorvar,
                                                 1.0, 1.0, 1.0)

function logprior(d::SteinMultivariateStudentTRegressionGaussianPrior,
                  beta::Array{Float64,1})
    logpriorvalue =
        -0.5 * sum((beta .- d.priormu) .^ 2 ./ d.priorvar) -
        -0.5 * sum(log(2 * pi .* d.priorvar))
    logpriorvalue
end

function gradlogprior(d::SteinMultivariateStudentTRegressionGaussianPrior,
                      beta::Array{Float64,1})
    -(beta .- d.priormu) ./ d.priorvar
end

function numdimensions(d::SteinMultivariateStudentTRegressionGaussianPrior)
    length(d.priormu)
end
