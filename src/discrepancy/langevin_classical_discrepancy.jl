using JuMP
using MathProgBase.SolverInterface: AbstractMathProgSolver
using SteinDistributions
using SteinDiscrepancy: LangevinDiscrepancyResult

# Computes classical Stein discrepancy based on the Langevin operator
# for univariate target distributions: 
#
#     max_{g} sum_i q(i) * (g(x(i)) * (d/dx)log(p(x(i))) + g'(x(i)))
#     subject to, for all z, y
#     |g(z)| <= c_{1} I[alpha < z < beta], |g'(z)| <= c_{2},
#     |g(z) - g(y)| <= c_{2} |z-y|,
#     |g'(z) - g'(y)| <= c_3 |z-y|,
#     |g(z) - g(y) - g'(y)(z - y)| <= (c_3/2) |z-y|^2,
#     |g(z) - g(y) - g'(z)(z - y)| <= (c_3/2) |z-y|^2
#    where (alpha,beta) is the support of target.
#
# Despite this being an infinite dimensional feasible set, there
# is a set of finite dimensional constraints that can exactly
# recover the optimal solution g, g' on the points x_i.
#
# Args:
#  sample - SteinDiscrete object representing a sample
#  target - SteinDistribution object representing target distribution
#  solver - optimization program solver supported by JuMP that can solve
#   a quadratically constrained quadratic program (QCQP)
function langevin_classical_discrepancy(sample::SteinDiscrete,
                                        target::SteinDistribution,
                                        solver::AbstractMathProgSolver)
    # get primary objects
    points = sample.support
    weights = vec(sample.weights)
    n = length(weights)
    d = size(points,2)
    # check if d is 1
    (d == 1) || error("The classical discrepancy only works for univariate distributions.")
    ## Find spanner edges (consecutive pairs)
    println("[Computing spanner edges]")
    tic(); edges = getspanneredges(points); edgetime = toc()
    println("\tusing $(size(edges,1)) of $(binomial(n,2)) edges")
    tic()
    # start setting up the model
    m = Model(solver=solver)
    # Get Stein factors
    c1 = getC1(target)
    c2 = getC2(target)
    c3 = getC3(target)
    # Define variables and specify single variable bounds
    @defVar(m, -c1 <= g[1:n] <= c1)
    @defVar(m, -c2 <= gprime[1:n] <= c2)
    # Introduce classical Stein program slack variables
    @defVar(m, -Inf <= tb[1:(n-1)] <= Inf)
    @defVar(m, -Inf <= tu[1:(n-1)] <= Inf)
    # Objective coefficients for each g(x_i)
    gradlogdensities = gradlogdensity(target, points)
    gcoefficients = vec(weights .* gradlogdensities)
    # set objective
    gobj = AffExpr(g[1:n], gcoefficients, 0.0)
    gprimeobj = AffExpr(gprime[1:n], weights, 0.0)
    @setObjective(m, Max, gobj + gprimeobj)
    # add gprime constraints
    xdistances = points[2:n, 1] - points[1:(n-1), 1]
    for i=1:(n-1)
        @addConstraint(m, gprime[i] - gprime[i+1] <= c3*xdistances[i])
        @addConstraint(m, gprime[i] - gprime[i+1] >= -c3*xdistances[i])
    end
    # Introduce constraints tb >= Lb and tu >= Lu:
    # tb_i >= c3/2 * (x_{i+1} - x_i) - (g'(x_i) + g'(x_{i+1}))/2 - c2
    # tu_i >= c3/2 * (x_{i+1} - x_i) + (g'(x_i) + g'(x_{i+1}))/2 - c2
    slackoffsets = (c3/2) * xdistances - c2
    for i=1:(n-1)
        @addConstraint(m, tb[i] + 0.5 * gprime[i] + 0.5 * gprime[i+1] >= slackoffsets[i])
        @addConstraint(m, tu[i] - 0.5 * gprime[i] - 0.5 * gprime[i+1] >= slackoffsets[i])
    end
    # Introduce quadratic buffer constraints:
    # |g(x_i)| <= c_1 - (1/(2c_3)) g'(x_i)^2
    for i=1:n
        qexp = QuadExpr([gprime[i]], [gprime[i]], [1/(2*c3)], AffExpr([g[i]], [1.0], 0.0))
        @addConstraint(m, qexp <= c1)
        qexp = QuadExpr([gprime[i]], [gprime[i]], [1/(2*c3)], AffExpr([g[i]], [-1.0], 0.0))
        @addConstraint(m, qexp <= c1)
    end
    # Add sharp constraints linking g and g':
    # g(x_{i+1}) - g(x_i) + (g'(x_{i+1}) - g'(x_i))^2/(4c3) - (x_{i+1} - x_i)*(g'(x_i) + g'(x_{i+1}))/2 + (1/c3)(L_u)^2_+ <= (c3/4)(x_{i+1} - x_i)^2
    # g(x_i) - g(x_{i+1}) + (g'(x_{i+1}) - g'(x_i))^2/(4c3) + (x_{i+1} - x_i)*(g'(x_i) + g'(x_{i+1}))/2 + (1/c3)(L_b)^2_+ <= (c3/4)(x_{i+1} - x_i)^2
    scaledsquaredxdistances = (xdistances.^2) .* (c3/4)
    for i=1:(n-1)
        rexp = AffExpr(
            [g[i:(i+1)], gprime[i:(i+1)]],
            [1.0, -1.0,  xdistances[i]/2, xdistances[i]/2],
            0.0
        )
        qbexp = QuadExpr(
            [gprime[[i, i, i+1]], tb[i]],
            [gprime[[i, i+1, i+1]], tb[i]],
            [1.0, -2.0, 1.0, 4.0] ./ (4*c3),
            rexp
        )
        quexp = QuadExpr(
            [gprime[[i, i, i+1]], tu[i]],
            [gprime[[i, i+1, i+1]], tu[i]],
            [1.0, -2.0, 1.0, 4.0] ./ (4*c3),
            -1 * rexp
        )

        @addConstraint(m, qbexp <= scaledsquaredxdistances[i])
        @addConstraint(m, quexp <= scaledsquaredxdistances[i])
    end

    # get boundary of support
    alpha = supportlowerbound(target, 1)
    beta = supportupperbound(target, 1)
    # add support slack constraints if necessary
    if isfinite(alpha)
        xdistance = points[1,1] - alpha
        # |g(x_1)| <= c2 (x_1 - alpha)
        @addConstraint(m, g[1] <= c2*xdistance)
        @addConstraint(m, g[1] >= -c2*xdistance)
        # g(x_1) <= g'(x_1)*(x_1 - alpha) + (c3/2)*(x_1-alpha)^2 - 1/(2c3) max{g'(x_1) + c3(x_1 - alpha) - c2, 0}^2,
        # g(x_1) >= g'(x_1)*(x_1 - alpha) - (c3/2)*(x_1-alpha)^2 + 1/(2c3) max{-g'(x_1) + c3(x_1 - alpha) - c2, 0}^2
        @defVar(m, -Inf <= alphab <= Inf)
        @defVar(m, -Inf <= alphau <= Inf)
        @addConstraint(m, alphab - gprime[1] >= c3 * xdistance - c2)
        @addConstraint(m, alphau + gprime[1] >= c3 * xdistance - c2)
        rexp = AffExpr(
            [g[1], gprime[1]],
            [1.0, -xdistance],
            0.0
        )
        qbexp = QuadExpr([alphab], [alphab], [1/(2*c3)], rexp)
        quexp = QuadExpr([alphau], [alphau], [1/(2*c3)], -1 * rexp)
        @addConstraint(m, qbexp <= (c3/2) * xdistance^2)
        @addConstraint(m, quexp <= (c3/2) * xdistance^2)
    end
    if isfinite(beta)
        xdistance = beta - points[end,1]
        # |g(x_n)| <= c2 (beta - x_n)
        @addConstraint(m, g[n] <= c2*xdistance)
        @addConstraint(m, g[n] >= -c2*xdistance)
        # -g(x_n) <= g'(x_n)*(beta-x_n) + (c3/2)*(beta-x_n)^2 - 1/(2c3) max{g'(x_n) + c3(beta-x_n) - c2, 0}^2
        # -g(x_n) >= g'(x_n)*(beta-x_n) - (c3/2)*(beta-x_n)^2 + 1/(2c3) max{-g'(x_n) + c3(beta-x_n) - c2, 0}^2
        @defVar(m, -Inf <= betab <= Inf)
        @defVar(m, -Inf <= betau <= Inf)
        @addConstraint(m, betab + gprime[n] >= c3 * xdistance - c2)
        @addConstraint(m, betau - gprime[n] >= c3 * xdistance - c2)
        rexp = AffExpr(
            [g[n], gprime[n]],
            [1.0, xdistance],
            0.0
        )
        qbexp = QuadExpr([betab], [betab], [1/(2*c3)], rexp)
        quexp = QuadExpr([betau], [betau], [1/(2*c3)], -1 * rexp)
        @addConstraint(m, qbexp <= (c3/2) * xdistance^2)
        @addConstraint(m, quexp <= (c3/2) * xdistance^2)
    end
    # Solve the optimization program
    @time status = JuMP.solve(m)
    solvetime = [toc()]

    # Package the results
    objval = [getObjectiveValue(m)]
    gopt = getValue(g)[1:n]
    gprimeopt = getValue(gprime)[1:n]
    operatorgopt = (gopt .* gcoefficients) + (gprimeopt .* weights)

    LangevinDiscrepancyResult(
        points,
        weights,
        edges,
        objval,
        gopt,
        gprimeopt,
        operatorgopt,
        edgetime,
        solvetime
    )
end
