# Bayesian student-t multivariate regression with PseudoHuber priors
# Prior pi(beta; mu, delta)
#   \propto exp(delta^2 (1 - sqrt(1+||beta-mu||_2^2/delta^2))

type SteinMultivariateStudentTRegressionPseudoHuberPrior <: SteinMultivariateStudentTRegressionPosterior
    X::Array{Float64,2}
    y::Array{Float64,1}
    nu::Float64  # the number of degrees of freedom for the t-distribution
    sigma::AbstractArray{Float64,2} # the covariance of the multivariate t-distribution (abstract allows for optimization)
    priormu::Array{Float64,1}
    priordelta::Float64 # transition point from quadratic to linear
    priorlogZ::Float64 # the log of the normalizing constant of the prior probability density
    c1::Float64
    c2::Float64
    c3::Float64
    # internal constructor that saves the normalizing constant
    SteinMultivariateStudentTRegressionPseudoHuberPrior(
        X, y, nu, sigma, priormu, priordelta, c1, c2, c3
    ) = (
        logZ = priorlogZ(priordelta, size(X,2));
        new(X, y, nu, sigma, priormu, priordelta, logZ, c1, c2, c3)
    )
end

SteinMultivariateStudentTRegressionPseudoHuberPrior(X::Array{Float64, 2},
                                                    y::Array{Float64, 1},
                                                    nu::Float64) =
SteinMultivariateStudentTRegressionPseudoHuberPrior(X, y, nu, Diagonal(ones(size(X, 1))))

SteinMultivariateStudentTRegressionPseudoHuberPrior(X::Array{Float64, 2},
                                                    y::Array{Float64, 1},
                                                    nu::Float64,
                                                    sigma::AbstractArray{Float64, 2}) =
SteinMultivariateStudentTRegressionPseudoHuberPrior(X, y, nu, sigma,
                                                    repmat([0.0], size(X, 2)),
                                                    repmat([1.0], size(X, 2)))

# Constructor with canonical setting of (c1,c2,c3) = (1,1,1)
SteinMultivariateStudentTRegressionPseudoHuberPrior(X::Array{Float64, 2},
                                                    y::Array{Float64, 1},
                                                    nu::Float64,
                                                    sigma::AbstractArray{Float64, 2},
                                                    priormu::Array{Float64, 1},
                                                    priordelta::Float64) =
SteinMultivariateStudentTRegressionPseudoHuberPrior(X, y, nu, sigma, priormu, priordelta,
                                                    1.0, 1.0, 1.0)
function priorlogZ(delta::Float64, p::Int)
    # Log volume, log(V_p), of p-dimensional hypersphere
    logvolume = log(pi)*p/2 - lgamma(p/2+1)
    # int f(||beta||) = p * V_p * int f(r) r^{p-1} dr
    (integral, error) = quadgk(
        x -> exp(x^(p-1) * delta^2 * (1 - sqrt(1 + (x / delta)^2))),
        0.0,
        Inf)
    log(p)+logvolume+log(integral)
end

function logprior(d::SteinMultivariateStudentTRegressionPseudoHuberPrior,
                  beta::Array{Float64, 1})
    # Compute PseudoHuber prior
    delta = d.priordelta
    logZ = d.priorlogZ
    residuals = beta .- d.priormu
    underroot = 1 + sum(residuals.^2) / delta^2

    delta^2 * (1 - sqrt(underroot))-logZ
end

function gradlogprior(d::SteinMultivariateStudentTRegressionPseudoHuberPrior,
                      beta::Array{Float64, 1})
    delta = d.priordelta
    residuals = beta .- d.priormu
    underroot = 1 + sum(residuals .^ 2) / delta^2
    -residuals / sqrt(underroot)
end

function numdimensions(d::SteinMultivariateStudentTRegressionPseudoHuberPrior)
    length(d.priormu)
end
