# SteinGMMPosterior
# A Gaussian mixture model posterior distribution p(theta) = pi(theta | y_1,...,y_L) induced by the model
#
#   y_l | theta ~iid 0.5 N(theta1, sigma2y) + 0.5 N(theta1 + theta2, sigma2y)
#   (theta1, theta2) ~ N(0, diag(sigma2theta1, sigma2theta2))
#
# This distribution is analyzed in the paper
#
#   http://www.columbia.edu/~jwp2128/Teaching/E9801/papers/WellingTeh2011.pdf

using Distributions: Beta, Binomial, MvNormal, Normal
import Distributions

# Default hyperparameter value
const defsigma2y = 2.0

type SteinGMMPosterior <: SteinPosterior
    # Vector of observed datapoints
    y::Array{Float64, 1}
    # Model hyperparameters
    sigma2y::Float64
    sigma2theta1::Float64
    sigma2theta2::Float64
    # Stein factors
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with uniform setting of (c1,c2,c3) and
# default settings of sigma2y and sigma2theta
SteinGMMPosterior(y; sigma2y=defsigma2y, sigma2theta1=10.0, sigma2theta2=1.0,
                  c1=1.0, c2=1.0, c3=1.0) =
SteinGMMPosterior(y, sigma2y, sigma2theta1, sigma2theta2, c1, c2, c3)

# Draws L datapoints from the likelihood distribution
#   0.5 N(theta[1], sigma2y) + 0.5 N(theta[1] + theta[2], sigma2y)
function randgmm(theta::Array{Float64, 1}, L::Int64;
                 sigma2y::Float64=defsigma2y)
    bin = Binomial(L)
    k = Distributions.rand(bin)

    phi1 = Normal(theta[1], sqrt(sigma2y))
    phi2 = Normal(theta[1] + theta[2], sqrt(sigma2y))

    vcat(Distributions.rand(phi1, k), Distributions.rand(phi2, L-k))
end

# Returns the log prior log pi(theta)
function logprior(d::SteinGMMPosterior,
                  theta::Array{Float64, 1})
    priornormal = MvNormal(diagm([d.sigma2theta1, d.sigma2theta2]))
    Distributions.logpdf(priornormal, theta)
end

# Returns log pi(d.y[idx] | theta)
function loglikelihood(d::SteinGMMPosterior,
                       theta::Array{Float64, 1};
                       idx=1:length(d.y))
    y = d.y[idx]

    phi1 = Normal(theta[1]           , sqrt(d.sigma2y))
    phi2 = Normal(theta[1] + theta[2], sqrt(d.sigma2y))
    sum(log(0.5 .* (Distributions.pdf(phi1, y) + Distributions.pdf(phi2, y))))
end

# Returns the gradient of the log prior grad log pi(theta)
function gradlogprior(d::SteinGMMPosterior,
                      theta::Array{Float64, 1})
    -(1.0 ./ [d.sigma2theta1, d.sigma2theta2]) .* theta
end

# Returns grad_{theta} log pi(d.y[idx] | theta)
function gradloglikelihood(d::SteinGMMPosterior,
                           theta::Array{Float64, 1};
                           idx=1:length(d.y))
    y = d.y[idx]

    phi1 = Normal(theta[1]           , sqrt(d.sigma2y))
    phi2 = Normal(theta[1] + theta[2], sqrt(d.sigma2y))
    # we have to be careful computing ratios b/c of numerical stability
    maxlogpdf = max(Distributions.logpdf(phi1, y),
                    Distributions.logpdf(phi2, y))
    logratiodenom = log(
        sum(exp([Distributions.logpdf(phi1, y) Distributions.logpdf(phi2, y)]
                .- maxlogpdf), 2)
    )
    ratio1 = exp(Distributions.logpdf(phi1, y) - maxlogpdf - vec(logratiodenom))
    ratio2 = 1 - ratio1
    # we can ignore the 1/2, since we're taking the gradient
    gradtheta1 = (y - theta[1]           ) .* ratio1 +
                 (y - theta[1] - theta[2]) .* ratio2
    gradtheta2 = (y - theta[1] - theta[2]) .* ratio2

    totgradtheta1 = sum(gradtheta1) / d.sigma2y
    totgradtheta2 = sum(gradtheta2) / d.sigma2y

    [totgradtheta1, totgradtheta2]
end

function numdimensions(d::SteinGMMPosterior)
    2
end

function diaghessian(d::SteinGMMPosterior,
                     theta::Array{Float64,1};
                     idx=1:length(d.y))
    y = d.y[idx]

    phi1 = Normal(theta[1], sqrt(d.sigma2y))
    phi2 = Normal(theta[2], sqrt(d.sigma2y))

    dens1 = Distributions.logpdf(phi1, y)
    dens2 = Distributions.logpdf(phi2, y)
    maxlogpdf = max(dens1, dens2)
    logratiodenom = log(sum(exp([dens1 dens2] .- maxlogpdf), 2))

    w1 = exp(dens1 - maxlogpdf - vec(logratiodenom))
    w2 = 1 .- w1

    ptweights = w1 .* w2 ./ (d.sigma2y)^2
    hess11 =  sum(ptweights .* (y .- theta[1]) .^2)
    hess22 =  sum(ptweights .* (y .- theta[2]) .^2)
    hess12 = -sum(ptweights .* (y .- theta[1]) .* (y .- theta[2]))

    # rotate the hessian b/c means are theta1, theta1 + theta2
    [hess11, hess12 + hess22] .- 1 ./ [d.sigma2theta1, d.sigma2theta2]
end
