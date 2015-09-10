# Extends the result from the lengevin graph discrepancy to the boundary
# (if applicable) for a univariate distribution.
#
# We need a few inequalities to hold in order to make this work. We'll describe
# the left endpoint (the right is symmetric):
#
# g_alpha = 0,
# |grad g_alpha| <= c2
# |grad g_alpha - grad g_1| <= c3 (x_1 - alpha)
# |g_1 - grad g_alpha (x_1 - alpha)| <= c3/2 (x_1 - alpha)^2
#
# Hence so long as the result is a valid result, we can pick g_alpha = g_1.
#
# Args:
#   res - result from a univariate langevin discrepancy
#   P   - univariate SteinDistribution that is the target distribution
#
# Returns:
#   Another LangevinDiscrepancyResult with the endpoints included.

using SteinDistributions: numdimensions, supportlowerbound, supportupperbound, getC2, getC3

function extend_langevin_graph_result(res::LangevinDiscrepancyResult, P::SteinDistribution)
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

        maxerror = max(
            gradg1 - c2,
            -gradg1 - c2,
            g1 - gradg1*(x1-alpha) - (c3/2)*(x1-alpha)^2,
            -g1 + gradg1*(x1-alpha) - (c3/2)*(x1-alpha)^2
        )
        if maxerror > 0
            warn(
                 @sprintf("
                      The left endpoint was %e outside of at least one constraint.
                      This could be due to numerical inaccuracy in the convex
                      solver, otherwise it is a bad langevin result.",
                 maxerror)
            )
        end
        g_alpha = 0
        gradg_alpha = gradg1

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

        maxerror = max(
            gradgn - c2,
            -gradgn - c2,
            gn - gradgn*(xn-beta) - (c3/2)*(xn-beta)^2,
            -gn + gradgn*(xn-beta) - (c3/2)*(xn-beta)^2
        )
        if maxerror > 0
            warn(
                 @sprintf("
                      The right endpoint was %e outside of at least one constraint.
                      This could be due to numerical inaccuracy in the convex
                      solver, otherwise it is a bad langevin result.",
                 maxerror)
            )
        end
        g_beta = 0
        gradg_beta = gradgn

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

