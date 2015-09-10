# SteinDistribution
# An abstract type representing a distribution P on R^d with support
# (alpha_1,beta_1) x ... x (alpha_d,beta_d) with -inf <= alpha_j < beta_j <= inf
abstract SteinDistribution
import Base.rand

# Returns first Stein factor
function getC1(d::SteinDistribution)
    return d.c1
end
# Returns second Stein factor
function getC2(d::SteinDistribution)
    return d.c2
end
# Returns third Stein factor
function getC3(d::SteinDistribution)
    return d.c3
end
# Returns lower bound of support for j-th dimension marginal of P
# Should be overridden for constrained distributions
function supportlowerbound(d::SteinDistribution, j::Int64)
    -Inf
end
# Returns upper bound of support for j-th dimension marginal of P
# Should be overridden for constrained distributions
function supportupperbound(d::SteinDistribution, j::Int64)
    Inf
end
