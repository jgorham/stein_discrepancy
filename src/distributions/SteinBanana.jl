# SteinBanana
# The banana posterior distribution p(theta | y_1,...,y_L) induced by the model
#
#   y_l | theta ~iid N(theta1 + theta2^2, sigma2y)
#   theta_1, theta_2 ~iid N(0, sigma2theta)
#
# This distribution is analyzed in the papers
#
#   http://arxiv.org/pdf/1211.3759v2.pdf
#   http://jmlr.org/proceedings/papers/v33/dubois14.pdf

using Distributions: Normal, logpdf
using SteinDistributions

type SteinBanana <: SteinPosterior
    y::Array{Float64, 1}
    sigma2y::Float64
    sigma2theta::Float64
    c1::Array{Float64, 1}
    c2::Array{Float64, 1}
    c3::Array{Float64, 1}
end

# Constructor with uniform setting of (c1,c2,c3) and
# default setting of sigma2y and sigma2theta
defaultsigma2y = 4.0
SteinBanana(y; sigma2y = defaultsigma2y, sigma2theta = 1.0) =
SteinBanana(y, sigma2y, sigma2theta,
            [1.0 for i=1:2],[1.0 for i=1:2],[1.0 for i=1:2])

# Returns lower bound of support
function supportlowerbound(d::SteinBanana, j::Int64)
    -Inf
end

# Returns upper bound of support
function supportupperbound(d::SteinBanana, j::Int64)
    Inf
end

# Draws n datapoints from the likelihood distribution
#   N(theta1 + theta2^2, sigma2y)
#
# Args:
#   mu = theta1 + theta2^2
function randbanana(mu::Float64, n::Int64)
    datanormal = Normal(mu, sqrt(defaultsigma2y))
    rand(datanormal, n)
end

# Computes the log prior for N(theta1 + theta2^2, sigma2y)
function logprior(d::SteinBanana,
                  theta::Array{Float64, 1})
    priornormal = Normal(0.0, sqrt(d.sigma2theta))
    sum(logpdf(priornormal, theta))
end

# Returns log p(theta | d.y[idx]), the log density of the posterior distribution
# evaluated at theta, conditioned only on those datapoints indexed by idx
function loglikelihood(d::SteinBanana,
                       theta::Array{Float64, 1};
                       idx=1:length(d.y))
    y = d.y[idx]
    theta0 = theta[1] + theta[2]^2

    datanormal = Normal(theta0, sqrt(d.sigma2y))
    sum(logpdf(datanormal, y))
end

# Returns grad_{theta} log p(d.y[idx] | theta)
function gradloglikelihood(d::SteinBanana,
                           theta::Array{Float64, 1};
                           idx=1:length(d.y))
    y = d.y[idx]
    theta0 = theta[1] + theta[2]^2

    gradtheta1 = sum(y - theta0) / d.sigma2y
    gradtheta2 = 2 * theta[2] * gradtheta1

    [gradtheta1, gradtheta2]
end

# Returns grad_{theta} log p(theta)
function gradlogprior(d::SteinBanana,
                      theta::Array{Float64, 1})
    -(1/d.sigma2theta) .* theta
end

# diaghessian(x) = (grad_1 b1(x), grad_2 b2(x), ....) where b = grad log p.
function diaghessian(d::SteinBanana, theta::Array{Float64,1}; idx=1:length(d.y))
    m = length(idx)
    priorterm = -1.0/d.sigma2theta
    ysum = sum(d.y[idx])
    likelihoodterm = [-m, -2*ysum + 2*m*theta[1] + 6*m*theta[2]^2] ./ d.sigma2y
    priorterm .+ likelihoodterm
end

function numdatapoints(d::SteinBanana)
    length(d.y)
end
