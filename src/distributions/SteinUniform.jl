# Uniform Stein Distribution
#
# Represents a uniform distribution with independent components

type SteinUniform <: SteinDistribution
    range::Array{Float64, 2}
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor for p independent uniform([0,1]) variables with non-uniform constants
SteinUniform(p::Int64, c1::Float64, c2::Float64, c3::Float64) =
SteinUniform(repmat([0.0; 1.0], 1, p), c1, c2, c3)

# Constructor with best known Stein factors setting (c1,c2,c3) = (1,1,1)
SteinUniform(range::Array{Float64, 2}) = SteinUniform(range, 1.0, 1.0, 1.0)

# Constructor for p independent uniform([0,1]) variables
SteinUniform(p::Int64) = SteinUniform(repmat([0.0; 1.0], 1, p))

# Univariate uniform([0,1]) constructor
SteinUniform() = SteinUniform(1)

# Draw n independent samples from distribution
function rand(d::SteinUniform, n::Int64)
    p = size(d.range, 2)
    # Sample uniform [0,1] variables
    x = zeros(n, p)
    rand!(x)
    # Shift and scale as needed
    shift = d.range[1,:]
    scale = (d.range[2,:] - d.range[1,:])
    scaled_x = broadcast(*, x, scale')
    broadcast(+, scaled_x, shift')
end

function supportlowerbound(d::SteinUniform, j::Int64)
    d.range[1, j];
end

function supportupperbound(d::SteinUniform, j::Int64)
    d.range[2, j];
end

# Gradient of log density of distribution evaluated at each row of
# n x p matrix X
function gradlogdensity(d::SteinUniform, X::Array{Float64,1})
    zeros(size(X));
end

function gradlogdensity(d::SteinUniform, X::Array{Float64,2})
    zeros(size(X));
end

# Cumulative distribution function (only valid when X is univariate)
function cdf(d::SteinUniform,t)
    @assert numdimensions(d) == 1
    lower = supportlowerbound(d,1);
    upper = supportupperbound(d,1);
    min(max((t-lower)/(upper-lower),0),1);
end

function numdimensions(d::SteinUniform)
    size(d.range, 2)
end
