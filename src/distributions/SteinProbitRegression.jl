# SteinProbitRegression
#
# We assume we have the following model:
#
# y_i = \Psi (beta_1 x_1 + ... + beta_p x_p)
#
# where \Psi is assumed to be the noraml CDF (and a flat prior for beta).
# We sample this by introducing the auxiliary variables
# y_i = I{z_i > 0}. Hence we have
#
# beta | Z, y ~ N((X^t X)^-1 (X_t Z), (X^t X)^-1)
# Z | beta, y ~ TN(X beta, I, y)
#
# where TN(mu, sigma, t) is a truncated Gaussian
# with mean mu, variance sigma and truncated to be
# positive if t == 1 and negative if t == 0.

using Distributions: TruncatedNormal, Normal
import Distributions

type SteinProbitRegression <: SteinPosterior
    # the X data
    X::Array{Float64,2}
    # the 0/1 y values
    y::Array{Float64,1}
    # sigma2 is the variance of Z given beta
    sigma2::Float64
end

# Values used in Seth Tribble's thesis:
defaultsigma2 = 1.0

SteinProbitRegression(X, y) =
    SteinProbitRegression(X, y, defaultsigma2)

# Returns lower bound of support
function supportlowerbound(d::SteinProbitRegression, i::Int)
    -Inf
end

# Returns upper bound of support
function supportupperbound(d::SteinProbitRegression, i::Int)
    Inf
end

function rungibbs(d::SteinProbitRegression,
                  n::Int;
                  qmcvariates=[])
    X = d.X
    m, p = size(X)
    betas = Array(Float64, n+1, p)
    Zs = Array(Float64, n+1, m)
    gsn = Normal()
    # work with qmc variates or iid ones
    if size(qmcvariates,1) != 0
        @assert size(qmcvariates) == (n, p+m)
        randvariates = qmcvariates
    else
        randvariates = rand(n, p+m)
    end
    # initialize the samples (z_i = I{y > 0})
    Zs[1,:] = round(Int, d.y .> 0.0)
    betas[1,:] = 0.0
    # do the gibbs!
    for i in 1:n
        # sample the betas
        xtxinv = inv(X' * X)
        xtxinvchol = chol(xtxinv)
        xtz = X' * vec(Zs[i,:])
        beta_mean = vec(xtxinv * xtz)

        noise = zeros(p)
        for j in 1:p
            noise[j] = Distributions.quantile(gsn, randvariates[i,j])
        end
        betas[i+1,:] = beta_mean + vec(xtxinvchol * noise)
        # now sample the z's
        zmean = vec(X * vec(betas[i+1,:]))
        for j in 1:m
            if d.y[j] > 0.0
                g = TruncatedNormal(zmean[j], d.sigma2, 0.0, Inf)
            else
                g = TruncatedNormal(zmean[j], d.sigma2, -Inf, 0.0)
            end
            Zs[i+1,j] = Distributions.quantile(g, randvariates[i,p+j])
        end
    end
    # discard the first point
    betas[2:end,:]
end

# Returns the log prior log pi(beta)
function logprior(d::SteinProbitRegression,
                  beta::Array{Float64, 1})
    0.0
end

# Returns log pi(d.y[idx] | beta)
function loglikelihood(d::SteinProbitRegression,
                       beta::Array{Float64, 1};
                       idx=1:length(d.y))
    y = d.y[idx]
    X = d.X[idx,:]
    ypos = find(y .> 0.0)
    yneg = find(y .<= 0.0)
    u = vec(X * beta)
    gsn = Normal()

    logprobpos = Distributions.logcdf(gsn, u[ypos])
    logprobneg = log(1.0 - Distributions.cdf(gsn, u[yneg]))
    sum(logprobpos) + sum(logprobneg)
end

# Returns the log prior log pi(beta)
function gradlogprior(d::SteinProbitRegression,
                      beta::Array{Float64, 1})
    zeros(size(beta,1))
end

# Returns grad_{beta} log pi(d.y[idx] | beta)
function gradloglikelihood(d::SteinProbitRegression,
                           beta::Array{Float64,1};
                           idx=1:length(d.y))
    y = d.y[idx]
    X = d.X[idx,:]
    ypos = find(y .> 0.0)
    yneg = find(y .<= 0.0)
    u = vec(X * beta)
    gsn = Normal()

    posweights = Distributions.pdf(gsn, u[ypos]) ./ Distributions.cdf(gsn, u[ypos])
    negweights = -Distributions.pdf(gsn, u[yneg]) ./ (1.0 - Distributions.cdf(gsn, u[yneg]))

    xpos = vec(sum(broadcast(*, X[ypos,:], posweights), 1))
    xneg = vec(sum(broadcast(*, X[yneg,:], negweights), 1))

    xneg + xpos
end

function numdimensions(d::SteinProbitRegression)
    size(d.X, 2)
end
