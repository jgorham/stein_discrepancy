# compare_discrepancies
#
# This script compares the values of the univariate classical and graph Stein
# discrepancies obtained when samples are drawn i.i.d. from their targets.
# We use non-uniform discrepancies with the best Stein factors drawn from the
# literature to enable comparison with a lower-bounding Wasserstein distance.

using SteinDistributions: SteinGaussian, SteinUniform
using SteinDiscrepancy: stein_discrepancy, wasserstein1d

include("experiment_utils.jl")

# random number generator seed
# we allow this to be set on the commandline
@parseintcli seed "s" "seed" 7
# set the target distribution
@parsestringcli distname "t" "target" "uniform"
# Specify dimension of sampled vectors
d = 1
# Specify a maximum number of samples
n = 30000
if distname == "uniform"
    # Independent Uniform([0.0,1.0]) with best known Stein constants
    (c1,c2,c3) = (0.5,0.5,1.0)
    target = SteinUniform(d, c1, c2, c3)
elseif distname == "gaussian"
    # Independent standard Gaussians with best known Stein constants
    (c1,c2,c3) = (1.0,4.0,2.0)
    target = SteinGaussian(d, c1, c2, c3)
end

# setup solvers for stein optimization problem
#graphsolver = classicalsolver = None
graphsolver = classicalsolver = "gurobi"
#graphsolver = "clp" # Only solves LPs
# Draw samples from target distribution
X = @setseed rand(target, n)
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:min(1000,n), 2000:1000:min(10000,n), 20000:10000:n)

# Solve optimization problem at each sample size
for i = ns
    @printf("[Beginning n=%d]\n", i)
    # Create program with data subset
    Xi = X[1:i,:]

    classicalobjective = graphobjective = wassdist = None
    if classicalsolver != None
        @printf("[Beginning classical solver n=%d]\n", i)
        classicalresult = stein_discrepancy(points=Xi, target=target,
                                            solver=classicalsolver,
                                            classical=true)
        classicalobjective = classicalresult.objectivevalue
    end
    if graphsolver != None
        @printf("[Beginning graph solver n=%d]\n", i)
        graphresult = stein_discrepancy(points=Xi, target=target,
                                        solver=graphsolver)
        graphobjective = graphresult.objectivevalue
    end

    @printf("[Computing wasserstein distance n=%d]\n", i)
    (wassdist, error) = wasserstein1d(points=Xi, target=target)

    # save data
    instance_data = Dict({
        "distname" => distname,
        "n" => i,
        "d" => d,
        "seed" => seed,
        "graphobjective" => graphobjective,
        "classicalobjective" => classicalobjective,
        "wassdist" => wassdist
    })
    save_json(
        instance_data;
        dir="compare_discrepancies",
        distname=distname,
        n=i,
        d=d,
        seed=seed
    )
end

@printf("COMPLETE!")
