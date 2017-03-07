# Gaussian Mixture Model Stein Distribution
#
# Represents a mixture of multivariate Gaussian distributions.
# Notice that sigmas represents the covariance matrix, and thus if you're
# using this for a univariate distribution, you should be passing in
# the variance (sigma2) terms.

using Distributions: Normal, MvNormal, Categorical
import Distributions
import Base.rand

type SteinGMM <: SteinDistribution
    mus::Array{Float64,2}    # each row is the mean for a Gaussian component
    sigmas::Array{Float64,3}  # each slice of tensor is the covariance
    precisions::Array{Float64,3}  # each slice of tensor is the precision matrix
    w::Array{Float64,1}
    c1::Float64
    c2::Float64
    c3::Float64
    # Internal constructor converts covariances to precision matrices
    SteinGMM(mus, sigmas, w, c1, c2, c3) = (
        precisions = Array(Float64, size(sigmas)...);
        for kk in 1:size(sigmas,3)
            precisions[:,:,kk] = inv(sigmas[:,:,kk]);
        end;
        new(mus, sigmas, precisions, w, c1, c2, c3)
    )
end

# Constructor with uniform setting of (c1,c2,c3) = (1,1,1)
SteinGMM(mus::Array{Float64,2}, sigmas::Array{Float64,3}, w::Array{Float64,1}) =
SteinGMM(mus, sigmas, w, 1.0, 1.0, 1.0)

SteinGMM(mus::Array{Float64,1}, sigmas::Array{Float64,3}, w::Array{Float64,1}) =
SteinGMM(mus'', sigmas, w)

# Constructor for equal weighted mixtures
SteinGMM(mus::Array{Float64,2}, sigmas::Array{Float64,3}) =
SteinGMM(mus, sigmas, repmat([1/size(mus,1)], size(mus,1)))

SteinGMM(mus::Array{Float64,1}, sigmas::Array{Float64,3}) =
SteinGMM(mus'', sigmas)

SteinGMM(mus::Array{Float64,2}, sigmas::Array{Float64,2}) = (
    (k,d) = size(mus);
    sigmas3 = Array(Float64, d, d, k);
    for ii in 1:k
        sigmas3[:,:,ii] = sigmas;
    end;
    SteinGMM(mus, sigmas3)
)

SteinGMM(mus::Array{Float64,1}, sigmas::Array{Float64,2}) =
SteinGMM(mus'', sigmas)

SteinGMM(mus::Array{Float64,2}, sigmas::Array{Float64,1}) = (
    (k,d) = size(mus);
    sigmas3 = Array(Float64, d, d, k);
    for ii in 1:k
        sigmas3[:,:,ii] = sigmas[ii] * eye(d);
    end;
    SteinGMM(mus, sigmas3)
)

SteinGMM(mus::Array{Float64,1}, sigmas::Array{Float64,1}) =
SteinGMM(mus'', sigmas)

# Constructor for Univariate standard Gaussian with canonical
# constants c1, c2, c3
SteinGMM(mus::Array{Float64,2}) =
SteinGMM(mus, repmat([1.0], size(mus,1)))

SteinGMM(mus::Array{Float64,1}) = SteinGMM(mus'')

# utility function for grabbing the ith mixture normal distribution
function getmixturedist(dist::SteinGMM, j::Int64)
    MvNormal(vec(dist.mus[j,:]), dist.sigmas[:,:,j])
end

# Draw n independent samples from distribution
function rand(dist::SteinGMM, n::Int64)
    p = numdimensions(dist)
    numdists = length(dist.w)

    mixturedist = Categorical(dist.w)
    mixindices = rand(mixturedist, n)

    samples = Array(Float64, n, p)
    for (ii, idx) in enumerate(mixindices)
        componentdist = getmixturedist(dist, idx)
        samples[ii,:] = rand(componentdist, 1)
    end
    n > 1 ? samples : vec(samples)
end

function supportlowerbound(dist::SteinGMM, j::Int64)
    -Inf
end

function supportupperbound(dist::SteinGMM, j::Int64)
    Inf
end

###
function logdensity(dist::SteinGMM, x::Array{Float64,1})
    d = numdimensions(dist)
    K = length(dist.w)
    # Compute the log of each component density times its weight
    logwtdcomponents = zeros(K,1)
    mvnormcons = (d/2.0) * log(2*pi)
    for kk in 1:K
        residual = x - vec(dist.mus[kk,:])
        logwtdcomponents[kk] = log(dist.w[kk]) - mvnormcons - 0.5*logdet(dist.sigmas[:,:,kk]) -
            0.5 * (residual' * dist.precisions[:,:,kk] * residual)[1]
    end
    # For numerical precision, subtract maximum log weighted component when
    # computing log sum exp
    maxterm = maximum(logwtdcomponents)
    maxterm + log(sum(exp(logwtdcomponents-maxterm)))
end

function gradlogdensity(dist::SteinGMM, x::Array{Float64,1})
    d = numdimensions(dist)
    K = length(dist.w)
    # Compute the gradient of each log component distribution
    gradlogcomponents = zeros(d,K)
    # Compute the log of each component density times its weight
    logwtdcomponents = zeros(K,1)
    mvnormcons = (d/2.0) * log(2*pi)
    for kk in 1:K
        residual = vec(dist.mus[kk,:]) - x
        sigmakinv = dist.precisions[:,:,kk]
        gradlogcomponents[:,kk] = sigmakinv * residual
        logwtdcomponents[kk] = log(dist.w[kk]) - mvnormcons - 0.5*logdet(dist.sigmas[:,:,kk]) -
            0.5 * (residual' * sigmakinv * residual)[1]
    end
    # For numerical precision, subtract maximum log weighted component
    logwtdcomponents = logwtdcomponents - maximum(logwtdcomponents)
    # Compute weighted component density ratios
    ratios = exp(logwtdcomponents - log(sum(exp(logwtdcomponents))))
    # Compute grad log density
    vec(gradlogcomponents * ratios)
end

function cdf(dist::SteinGMM, x::Float64)
    @assert size(dist.mus, 2) == 1

    numcomp = length(dist.w)
    mixturenormals = Array(Distributions.Normal, numcomp)

    for ii in 1:numcomp
        mixturenormals[ii] = Normal(dist.mus[ii,1], sqrt(dist.sigmas[1,1,ii]))
    end

    cdfs = [Distributions.cdf(mixturenormal, x) for mixturenormal in mixturenormals]
    res = sum(dist.w .* cdfs)
    res
end

function numdimensions(dist::SteinGMM)
    size(dist.mus, 2)
end
