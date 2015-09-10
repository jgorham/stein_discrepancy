# Extends the result from the lengevin classical discrepancy to the boundary
# (if applicable) for a univariate distribution. The only nontrivial part here is to
# enure that g never exits the interval [-c1, c1], which is outlined in the appendix
# of the paper http://arxiv.org/abs/1506.03039
#
# Args:
#   res - result from a univariate langevin discrepancy
#   P   - univariate SteinDistribution that is the target distribution
#
# Returns:
#   Another LangevinDiscrepancyResult with the endpoints included.

using SteinDistributions: numdimensions, supportlowerbound, supportupperbound, getC2, getC3

# Simple helper function for interpolating the gradient
#
# Params:
# lb:    the integral of the lower bound m_i of grad g (as outlined in the paper)
# ub:    the integral of the upper bound M_i of grad g (as outlined in the paper)
# gradb: the value of m_i at the unknown value of x
# gradu: the value of M_i at the unknown value of x
# gdiff: the difference g at the endpoints
# Returns:
# the value of grad g that would result from weighting m_i and M_i so the integral matches gdiff
function interpolate_grad(lb::Float64, gradb::Float64, ub::Float64, gradu::Float64, gdiff::Float64)
    zeta = (ub - gdiff) / (ub - lb)
    zeta*gradb + (1-zeta)*gradu
end

function extend_langevin_classical_result(res::LangevinDiscrepancyResult, P::SteinDistribution)
    d = numdimensions(P)
    if (d != 1)
        error("extend_langevin_graph_result only implemented for univariate targets.")
    end
    tic()

    alpha = supportlowerbound(P,1)
    beta = supportupperbound(P,1)
    c2, c3 = getC2(P), getC3(P)

    newpoints = copy(res.points)
    newweights = copy(res.weights)
    newedges = copy(res.edges)
    newg = copy(res.g)
    newgradg = copy(res.gradg)
    newoperatorg = copy(res.operatorg)

    if isfinite(alpha)
        i1 = indmin(newpoints)
        x1, g1, gradg1 = newpoints[i1], newg[i1], newgradg[i1]
        g_alpha, gradg_alpha = 0, None

        # wlog we can force gradg1 >= 0
        signflip = gradg1 >= 0 ? 1 : -1
        gradg1 *= signflip
        g1 *= signflip
        # these give bounds on grad g_alpha
        left_upper_grad = min(gradg1 + c3*(x1-alpha), c2)
        left_lower_grad = max(gradg1 - c3*(x1-alpha), -c2)
        # these bound the difference g1 - g_alpha
        left_upper_bound = gradg1*(x1-alpha) + (c3/2)*(x1-alpha)^2 - 1/(2c3) * max(gradg1+c3*(x1-alpha)-c2, 0)^2
        left_lower_bound = gradg1*(x1-alpha) - (c3/2)*(x1-alpha)^2 + 1/(2c3) * max(-gradg1+c3*(x1-alpha)-c2, 0)^2
        # break into cases depending if the bottom bound m_i crosses x-axis
        if gradg1 >= c3*(x1-alpha)
            # we can take a linear interpolation w/o risking exiting [-c1, c1]
            gradg_alpha = interpolate_grad(
                left_lower_bound, left_lower_grad, left_upper_bound, left_upper_grad, g1)
        else
            left_mid_bound = gradg1^2/(2*c3)
            if g1 >= left_mid_bound
                gradg_alpha = interpolate_grad(left_mid_bound, 0.0, left_upper_bound, left_upper_grad, g1)
            else
                gradg_alpha = interpolate_grad(left_lower_bound, left_lower_grad, left_mid_bound, 0.0, g1)
            end
        end
        gradg_alpha *= signflip

        newpoints = vcat(alpha, newpoints)
        newweights = vcat(0, newweights)
        newedges = vcat([0 i1], newedges)
        newedges += 1
        newg = vcat(g_alpha, newg)
        newgradg = vcat(gradg_alpha, newgradg)
        newoperatorg = vcat(gradg_alpha, newoperatorg)
    end
    if isfinite(beta)
        in = indmax(newpoints)
        xn, gn, gradgn = newpoints[in], newg[in], newgradg[in]
        n = length(newpoints)
        g_beta, gradg_beta = 0, None

        # wlog we can force gradgn >= 0
        signflip = gradgn >= 0 ? 1 : -1
        gradgn *= signflip
        gn *= signflip
        # these give bounds on grad g_alpha
        right_upper_grad = min(gradgn + c3*(beta-xn), c2)
        right_lower_grad = max(gradgn - c3*(beta-xn), -c2)
        # these bound the difference g1 - g_alpha
        right_upper_bound = gradgn*(beta-xn) + (c3/2)*(beta-xn)^2 - 1/(2c3) * max(gradgn+c3*(beta-xn)-c2, 0)^2
        right_lower_bound = gradgn*(beta-xn) - (c3/2)*(beta-xn)^2 + 1/(2c3) * max(-gradgn+c3*(beta-xn)-c2, 0)^2
        # break into cases depending if the bottom bound m_i crosses x-axis
        if gradgn >= c3*(beta-xn)
            # we can take a linear interpolation w/o risking exiting [-c1, c1]
            gradg_beta = interpolate_grad(
                right_lower_bound, right_lower_grad, right_upper_bound, right_upper_grad, -gn)
        else
            right_mid_bound = gradgn^2/(2*c3)
            if -gn >= left_mid_bound
                gradg_beta = interpolate_grad(right_mid_bound, 0.0, right_upper_bound, right_upper_grad, -gn)
            else
                gradg_beta = interpolate_grad(right_lower_bound, right_lower_grad, right_mid_bound, 0.0, -gn)
            end
        end
        gradg_beta *= signflip

        newpoints = vcat(newpoints, beta)
        newweights = vcat(newweights, 0)
        newedges = vcat(newedges, [in (n+1)])
        newg = vcat(newg, g_beta)
        newgradg = vcat(newgradg, gradg_beta)
        newoperatorg = vcat(newoperatorg, gradg_beta)
    end
    extendtime = toc()

    LangevinDiscrepancyResult(
        newpoints,
        newweights,
        newedges,
        res.objectivevalue,
        newg,
        newgradg,
        newoperatorg,
        res.edgetime,
        res.solvetime + extendtime
    )
end

