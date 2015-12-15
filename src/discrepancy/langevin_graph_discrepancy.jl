using JuMP
using MathProgBase.SolverInterface: AbstractMathProgSolver
using SteinDistributions
using SteinDiscrepancy: LangevinDiscrepancyResult

# Computes graph Stein discrepancy based on the Langevin operator and a greedy
# 2-spanner of the complete graph on sample points x(i,:) with edges weighted by
# ||x(i,:) - x(i2,:)||
#
# For each j in {1,...,d}, returns a solution g_j to
#   max_{g_j}
#   sum_i q(i) (g_j(x(i,:)) * (d/dx_j)log(p(x(i,:))) + (d/dx_j)g_j(x(i,:)))
#   subject to, for each spanner edge (i, i2),
#    |g_j(x(i,:))| <= c_{1}, ||grad g_j(x(i,:))||^* <= c_{2},
#    |g_j(x(i,:)) - g_j(x(i2,:))| <= c_{2} ||x(i,:) - x(i2,:)||,
#    ||grad g_j(x(i,:)) - grad g_j(x(i2,:))||^*
#      <= c_3 ||x(i,:) - x(i2,:)||
#    |g_j(x(i,:)) - g_j(x(i2,:)) - <grad g_j(x(i2,:)), x(i,:) - x(i2,:)>|
#      <= (c_3/2) (||x(i,:) - x(i2,:)||)^2
#    |g_j(x(i,:)) - g_j(x(i2,:)) - <grad g_j(x(i,:)), x(i,:) - x(i2,:)>|
#      <= (c_3/2) (||x(i,:) - x(i2,:)||)^2
# where the norm ||.|| is the L1 norm and ||.||^* is its dual.
#
# Args:
#  sample - SteinDiscrete object representing a sample
#  target - SteinDistribution object representing target distribution
#  solver - optimization program solver supported by JuMP
function langevin_graph_discrepancy(sample::SteinDiscrete, 
                                    target::SteinDistribution, 
                                    solver::AbstractMathProgSolver)
    ## Extract inputs
    points = sample.support
    weights = sample.weights
    n = length(weights)
    d = size(points,2)
    c1 = getC1(target)
    c2 = getC2(target)
    c3 = getC3(target)
    # Objective coefficients for each g(x_i)
    gradlogdensities = gradlogdensity(target, points)
    gcoefficients = weights .* gradlogdensities

    ## Find spanner edge set
    println("[Computing spanner edges]")
    tic(); edges = getspanneredges(points); edgetime = toc()
    println("\tusing $(size(edges,1)) of $(binomial(n,2)) edges");
    if n > 1
        # Compute differences between points connected by an edge
        diffs = points[edges[:,1],:] - points[edges[:,2],:]
        # Compute L1 distances between points connected by an edge
        distances = sum(abs(diffs),2)
        scaled_squared_distances = (c3/2)*distances.^2
    end

    ## Prepare return values
    objval = Array(Float64, 1, d)
    gopt = Array(Float64, n, d)
    gradgopt = Array(Float64, n, d, d)
    solvetime = Array(Float64, 1, d)

    # Distance cutoff for enforcing Lipschitz function constraints
    lipfunccutoff = 2*c1/c2
    # Distance cutoff for enforcing Lipschitz gradient constraints
    lipgradcutoff = 2*c2/c3
    # Distance cutoff for enforcing Taylor compatibility constraints
    taylorcutoff = 4*c2/c3

    ## Solve a different problem for each sample coordinate
    println("[Solving optimization program]")
    for j = 1:d
        tic()
        ## Define optimization problem
        m = Model(solver=solver)
        # Define optimization variables
        @defVar(m, -c1 <= g[i=1:n] <= c1)
        @defVar(m, -c2 <= gradg[i=1:n,k=1:d] <= c2)
        # Define the optimization objective
        gobj = AffExpr(g[1:n], vec(gcoefficients[:,j]), 0.0)
        gradgobj = AffExpr(gradg[1:n,j], vec(weights), 0.0)
        @setObjective(m, Max, gobj + gradgobj)
        # Find finite limits of support in dimension j
        limits = [
            supportlowerbound(target, j),
            supportupperbound(target, j)
        ]
        limits = filter(isfinite, limits)
        # Add boundary constraints if needed
        for i = 1:n, bj = limits
            slackij = points[i,j] - bj
            # Add constraints to ensure gj can vanish on boundary
            # whenever abs(slackij) < lipfunccutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < lipfunccutoff
                @addConstraint(m, g[i] >= -c2 * abs(slackij))
                @addConstraint(m, g[i] <= c2 * abs(slackij))
            end
            # Add constraints to ensure \grad gj can vanish in non-j dimension
            # whenever abs(slackij) < lipgradcutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < lipgradcutoff
                constrained_dims = filter(x -> x != j, 1:d)
                for k = constrained_dims
                    @addConstraint(m, gradg[i,k] >= -c3 * abs(slackij))
                    @addConstraint(m, gradg[i,k] <= c3 * abs(slackij))
                end
            end
            # Add \grad gj constraints on jth dimension
            # whenever abs(slackij) < taylorcutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < taylorcutoff
                @addConstraint(m, g[i] - slackij * gradg[i,j] <= (c3/2) * slackij^2)
                @addConstraint(m, g[i] - slackij * gradg[i,j] >= (-c3/2) * slackij^2)
            end
        end
        # Add pairwise constraints
        if n > 1
            for i = 1:length(distances)
                v1 = edges[i,1]; v2 = edges[i,2]
                # Add Lipschitz function constraints
                # whenever distance < lipgradcutoff
                # (otherwise constraints will never be active)
                if distances[i] < lipfunccutoff
                    @addConstraint(m, g[v1] - g[v2] <= c2 * distances[i])
                    @addConstraint(m, g[v1] - g[v2] >= -c2 * distances[i])
                end
                # Add Lipschitz gradient constraints
                # whenever distances[i] < lipgradcutoff
                # (otherwise constraints will never be active)
                if distances[i] < lipgradcutoff
                    for k = 1:d
                        @addConstraint(m, gradg[v1,k] - gradg[v2,k] <= c3*distances[i])
                        @addConstraint(m, gradg[v1,k] - gradg[v2,k] >= -c3*distances[i])
                    end
                end
                # Add Taylor compatibility constraints relating g and gradg
                # whenever distance < taylorcutoff
                # (otherwise constraints will never be active)
                if distances[i] < taylorcutoff
                    expr1 = AffExpr(vec(gradg[v1,:]), vec(diffs[i,:]), 0.0)
                    @addConstraint(m, g[v1] - g[v2] - expr1 <= scaled_squared_distances[i])
                    @addConstraint(m, g[v1] - g[v2] - expr1 >= -scaled_squared_distances[i])
                    expr2 = AffExpr(vec(gradg[v2,:]), vec(diffs[i,:]), 0.0)
                    @addConstraint(m, g[v1] - g[v2] - expr2 <= scaled_squared_distances[i])
                    @addConstraint(m, g[v1] - g[v2] - expr2 >= -scaled_squared_distances[i])
                end
            end
        end
        # Solve the problem
        @time status = JuMP.solve(m)
        solvetime[j] = toc()
        # Package the results
        objval[j] = getObjectiveValue(m)
        gopt[:,j] = getValue(g)[1:n]
        gradgopt[:,:,j] = getValue(gradg)[1:n, 1:d]
    end
    # Compute (T_Pg)(x) = <grad, g(x)> + <g(x),grad log p(x)>
    operatorgopt = sum(gopt .* gradlogdensities,2)
    for j = 1:d
        operatorgopt += gradgopt[:,j,j]
    end

    LangevinDiscrepancyResult(
        points,
        weights,
        edges,
        objval,
        gopt,
        gradgopt,
        operatorgopt,
        edgetime,
        solvetime
    );
end
