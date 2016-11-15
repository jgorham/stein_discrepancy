# SteinSquaredFourierSumDensity
# This model is used in Section 5.1 of http://icml.cc/2012/papers/683.pdf
#
# The density has support [0,1] and is given by
#
# p(x) \propto (\sum_{i=1}^d a_i cos(2i\pi x) + b_i sin(2i\pi x))^2
#
# If d=0 then we reduce this to the uniform density

type SteinSquaredFourierSumDensity <: SteinDistribution
    a::Array{Float64, 1}
    b::Array{Float64, 1}
    c1::Float64
    c2::Float64
    c3::Float64
end

# Just the empty d=0 case
SteinSquaredFourierSumDensity() = SteinSquaredFourierSumDensity([], [])

# Scalar a,b
SteinSquaredFourierSumDensity(a::Float64, b::Float64) =
SteinSquaredFourierSumDensity([a], [b])

# Default (c1,c2,c3) = (1,1,1)
SteinSquaredFourierSumDensity(a::Array{Float64, 1}, b::Array{Float64, 1}) =
SteinSquaredFourierSumDensity(a, b, 1.0, 1.0, 1.0)

function supportlowerbound(d::SteinSquaredFourierSumDensity, j::Int64)
    0.0
end

function supportupperbound(d::SteinSquaredFourierSumDensity, j::Int64)
    1.0
end

function gradlogdensity(d::SteinSquaredFourierSumDensity, X::Array{Float64,2})
    p = length(d.a)
    if p == 0
        return zeros(size(X))
    end
    cosix = [cos(2*pi*i*x) for i=1:p, x=X]
    sinix = [sin(2*pi*i*x) for i=1:p, x=X]
    ia_i = d.a .* (1:p)
    ib_i = d.b .* (1:p)

    numer = cosix .* ib_i - sinix .* ia_i
    denom = cosix .* d.a  + sinix .* d.b

    gradlog = 4*pi*(sum(numer, 1) ./ sum(denom, 1))
    reshape(gradlog, size(X)...)
end

function numdimensions(d::SteinSquaredFourierSumDensity)
    length(d.a)
end
