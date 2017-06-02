# compare_classical_and_graph_discrepancies
#
# This script compares the values of the univariate classical and graph Stein
# discrepancies obtained when samples are drawn i.i.d. from their targets.
# We use non-uniform discrepancies with the best Stein factors drawn from the
# literature to enable comparison with a lower-bounding Wasserstein distance.

include("experiment_utils.jl")

using SteinDistributions: SteinGaussian, SteinUniform, gradlogdensity,
    supportlowerbound, supportupperbound, cdf
using SteinDiscrepancy: stein_discrepancy, wasserstein1d

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
    target = SteinUniform(d)
elseif distname == "gaussian"
    # Independent standard Gaussians with best known Stein constants
    (c1,c2,c3) = (1.0,4.0,2.0)
    target = SteinGaussian(d)
end
lowerbounds = [supportlowerbound(target, 1)]
upperbounds = [supportupperbound(target, 1)]
# define gradlogp
function gradlogp(x::Array{Float64,1})
    gradlogdensity(target, x)
end
# define targetcdf
function targetcdf(x::Float64)
    cdf(target, x)
end
# setup solvers for stein optimization problem
#graphsolver = classicalsolver = nothing
graphsolver = classicalsolver = "gurobi"
#graphsolver = "clp" # Only solves LPs
# Draw samples from target distribution
X = @setseed rand(target, n)
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:min(1000,n), 2000:1000:min(10000,n), 20000:10000:n)

# Solve optimization problem at each sample size
for i in ns
    @printf("[Beginning n=%d]\n", i)
    # Create program with data subset
    Xi = X[1:i,:]

    classicalobjective = graphobjective = nothing
    classicalsolvetime = graphsolvetime = nothing
    if classicalsolver != nothing
        @printf("[Beginning classical solver n=%d]\n", i)
        classicalresult = stein_discrepancy(points=Xi,
                                            gradlogdensity=gradlogp,
                                            solver=classicalsolver,
                                            method="classical",
                                            c1=c1,
                                            c2=c2,
                                            c3=c3,
                                            supportlowerbounds=lowerbounds,
                                            supportupperbounds=upperbounds)
        classicalobjective = classicalresult.objectivevalue
        classicalsolvetime = classicalresult.solvetime
    end
    if graphsolver != nothing
        @printf("[Beginning graph solver n=%d]\n", i)
        graphresult = stein_discrepancy(points=Xi,
                                        gradlogdensity=gradlogp,
                                        solver=graphsolver,
                                        c1=c1,
                                        c2=c2,
                                        c3=c3,
                                        supportlowerbounds=lowerbounds,
                                        supportupperbounds=upperbounds)
        graphobjective = graphresult.objectivevalue
        graphsolvetime = graphresult.solvetime
    end

    @printf("[Computing wasserstein distance n=%d]\n", i)
    (wassdist, error) = wasserstein1d(Xi, targetcdf=targetcdf)

    # save data
    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "n" => i,
        "d" => d,
        "seed" => seed,
        "graphobjective" => graphobjective,
        "graphsolvetime" => graphsolvetime,
        "classicalobjective" => classicalobjective,
        "classicalsolvetime" => classicalsolvetime,
        "wassdist" => wassdist
    )
    save_json(
        instance_data;
        dir="compare_classical_and_graph_discrepancies",
        distname=distname,
        n=i,
        d=d,
        seed=seed
    )
end

@printf("COMPLETE!")
