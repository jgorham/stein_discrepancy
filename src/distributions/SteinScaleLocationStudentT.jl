# Scale-Location Student-t Stein Distribution
#
# Represents a mu + sigma * X where X is a univariate t-distribution with nu
# degrees of freedom
using Distributions: TDist
import Distributions

type SteinScaleLocationStudentT <: SteinDistribution
    nu::Float64
    mu::Float64
    sigma::Float64
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with canonical setting of (c1,c2,c3) = (1,1,1)
SteinScaleLocationStudentT(nu::Float64, mu::Float64, sigma::Float64) =
SteinScaleLocationStudentT(nu, mu, sigma, 1.0, 1.0, 1.0);

# Constructor for standard t-distribution with canonical constants c1, c2, c3
SteinScaleLocationStudentT(nu::Float64) =
SteinScaleLocationStudentT(nu, 0.0, 1.0, 1.0, 1.0, 1.0);

# Draw n independent samples from distribution
function rand(d::SteinScaleLocationStudentT, n::Int64)
    d.mu + d.sigma * Distributions.rand(TDist(d.nu), n)'';
end

function supportlowerbound(d::SteinScaleLocationStudentT, j::Int64)
    -Inf
end

function supportupperbound(d::SteinScaleLocationStudentT, j::Int64)
    Inf
end

# Gradient of log density of distribution evaluated at each
# entry of n x 1 matrix X
function gradlogdensity(d::SteinScaleLocationStudentT, X::Array{Float64,2})
    -(d.nu + 1) * (X - d.mu) ./ (d.nu * d.sigma^2 + (X - d.mu).^2);
end

# Cumulative distribution function
function cdf(d::SteinScaleLocationStudentT, t)
    Distributions.cdf(TDist(d.nu),(t-d.mu)/d.sigma);
end

function numdimensions(d::SteinScaleLocationStudentT)
    1
end
