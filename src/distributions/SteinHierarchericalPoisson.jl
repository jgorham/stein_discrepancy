# SteinHierarchericalPoisson
#
# We assume we have the following hierarcherical model:
#
# n_i | T_i, lambda_i ~ Pois(T_i lambda_i)
# lambda_i | beta ~ Gamma(alpha, beta)
# beta ~ Gamma(gamma, delta)
#

using Distributions: quantile, Gamma, Poisson

type SteinHierarchericalPoisson <: SteinDistribution
    # the counts from each poisson
    n::Array{Int}
    # the times from each poisson
    T::Array{Float64,1}
    # the shape parameter for the poisson priors
    alpha::Float64
    # the shape parameter for the beta prior
    gamma::Float64
    # the scale parameter for the beta prior
    delta::Float64
end

# Values used in Seth Tribble's thesis:
defaultalpha = 1.802
defaultgamma = 0.1
defaultdelta = 1.0

SteinHierarchericalPoisson(n, T) =
    SteinHierarchericalPoisson(n, T, defaultalpha, defaultgamma, defaultdelta)

# Returns lower bound of support
function supportlowerbound(d::SteinHierarchericalPoisson, i::Int)
    0.0
end

# Returns upper bound of support
function supportupperbound(d::SteinHierarchericalPoisson, i::Int)
    Inf
end

function numpoissons(d::SteinHierarchericalPoisson)
    @assert length(d.n) == length(d.T)
    length(d.n)
end

function rungibbs(d::SteinHierarchericalPoisson,
                  n::Int;
                  qmcvariates=[])
    p = numpoissons(d)
    lambdas = Array(Float64, n+1, p)
    betas = Array(Float64, n+1)
    # work with qmc variates or iid ones
    if size(qmcvariates,1) != 0
        @assert size(qmcvariates) == (n, p+1)
        randvariates = qmcvariates
    else
        randvariates = rand(n, p+1)
    end
    # initialize the samples
    lambdas[1,:] = [1.0 for _ in 1:p]
    betas[1] = 1.0
    # do the gibbs!
    for i in 1:n
        # sample the lambdas
        for j in 1:p
            g = Gamma(d.alpha + d.n[j], (betas[i] + d.T[j])^-1)
            lambdas[i+1,j] = quantile(g, randvariates[i,j])
        end
        # sample the beta
        g = Gamma(d.gamma + 10*d.alpha, (d.delta + sum(lambdas[i+1,:]))^-1)
        betas[i+1] = quantile(g, randvariates[i,p+1])
    end
    # discard the first point
    lambdas[2:end,:], betas[2:end]
end

function gradlogdensity(d::SteinHierarchericalPoisson,
                        point::Array{Float64,1})
    lambdas, beta = point[1:end-1], point[end]
    gradlambda = beta + d.T + ((d.n + d.alpha) ./ lambdas)
    gradbeta = -sum(lambdas) + d.gamma / beta - d.delta
    [gradlambda; gradbeta]
end
