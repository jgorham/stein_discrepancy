# Gaussian Stein Distribution
#
# Represents a multivariate or univariate Gaussian distribution
using Distributions: MvNormalCanon, Normal
import Distributions

type SteinGaussian{T<:Number} <: SteinDistribution
    mu::Array{Float64}
    precision::AbstractArray{T,2}
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with Wasserstein-bounding non-uniform setting of
# (c1,c2,c3) = (1,1,1)
SteinGaussian(mu::Array{Float64}, precision::AbstractArray{Float64}) =
SteinGaussian(mu, precision, 1.0, 1.0, 1.0)

# Constructor for p independent standard Gaussian variables
SteinGaussian(p::Int64, c1::Float64, c2::Float64, c3::Float64) =
SteinGaussian(zeros(p), eye(p), c1, c2, c3);

# Constructor for p independent standard Gaussian variables with canonical
# constants c1, c2, c3
SteinGaussian(p::Int64) = SteinGaussian(zeros(p), eye(p))

# Constructor for Univariate standard Gaussian with canonical
# constants c1, c2, c3
SteinGaussian() = SteinGaussian(1)

# Draw n independent samples from distribution
function rand(d::SteinGaussian, n::Int64)
    Distributions.rand(MvNormalCanon(d.precision*d.mu, d.precision), n).';
end

function supportlowerbound(d::SteinGaussian, j::Int64)
    -Inf
end

function supportupperbound(d::SteinGaussian, j::Int64)
    Inf
end

function gradlogdensity(d::SteinGaussian, x::Array{Float64,1})
    d.precision * (d.mu - x)
end

# Gradient of log density of distribution evaluated at each row of
# n x p matrix X
function gradlogdensity(d::SteinGaussian, X::Array{Float64,2})
    (d.mu.' .- X) * d.precision;
end

# Cumulative distribution function (only valid when X is univariate)
function cdf(d::SteinGaussian, t)
    Distributions.cdf(Normal(d.mu[1],1/sqrt(d.precision[1])),t);
end

function numdimensions(d::SteinGaussian)
    length(d.mu)
end

# diaghessian(x) = (grad_1 b1(x), grad_2 b2(x), ....) where b = grad log p.
function diaghessian(d::SteinGaussian, x::Array{Float64,1})
    -diag(d.precision)
end
