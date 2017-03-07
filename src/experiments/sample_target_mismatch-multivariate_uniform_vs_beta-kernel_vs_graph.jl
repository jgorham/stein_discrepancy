# sample_target_mismatch-multivariate_uniform_vs_beta-kernel_vs_graph
#
# This script compares the values of the multivariate graph Stein
# discrepancy and kernel discrepancy obtained when samples are drawn i.i.d.
# from their targets.

include("experiment_utils.jl")

# we see first if we want to parallelize
@parseintcli parallel "p" "parallel" "false"
if parallel == "true"
    addprocs(CPU_CORES - 1)
end

using Distributions: Beta
using SteinDistributions: SteinUniform
using SteinKernels: SteinChampionLenardMillsKernel
using SteinDiscrepancy: stein_discrepancy

# random number generator seed
# we allow this to be set on the commandline
@parseintcli seed "s" "seed" 7
# set the target distribution
@parsestringcli distname "t" "target" "uniform"
# set the discrepancy
@parsestringcli discrepancytype "d" "discrepancy" "championlenardmills"
# Specify dimension of sampled vectors
d = 2
# Specify a maximum number of samples
maxn = 200000
# uniform target
uniform_target = SteinUniform(d)
# beta target
beta_target = Beta(3, 3)
# the graphsolver
# setup kernel (if necessary)
graphsolver = nothing
kernel = nothing
if discrepancytype == "championlenardmills"
    kernel = SteinChampionLenardMillsKernel([0.0,1.0], 2)
elseif discrepancytype == "graph-gurobi"
    graphsolver = "gurobi"
elseif discrepancytype == "graph-clp"
    graphsolver = "clp"
end
# Draw samples from target distribution
if distname == "beta"
    X = @setseed reshape(rand(beta_target, d*maxn), maxn, d)
elseif distname == "uniform"
    X = @setseed rand(uniform_target, maxn)
end
# Sample sizes at which optimization problem will be solved
ns = vcat(10:1:min(10,maxn),
          100:100:min(1000,maxn),
          2000:1000:min(10000,maxn),
          20000:10000:min(100000,maxn),
          200000:100000:maxn)

# Solve optimization problem at each sample size
for i in ns
    @printf("[Beginning n=%d]:\n", i)
    # Create program with data subset
    Xi = X[1:i,:]
    discrepancy = nothing
    solvetime = nothing
    if kernel != nothing
        @printf("[Beginning kernel computations n=%d]\n", i)
        result = stein_discrepancy(points=Xi,
                                   target=uniform_target,
                                   method="kernel",
                                   kernel=kernel)
        discrepancy = sqrt(result.discrepancy2)
        solvetime = result.solvetime
    elseif discrepancytype in ["graph-clp", "graph-gurobi"]
        @printf("[Beginning graph solver n=%d]\n", i)
        result = stein_discrepancy(points=Xi,
                                   target=uniform_target,
                                   method="graph",
                                   solver=graphsolver)
        discrepancy = sum(result.objectivevalue)
        solvetime = sum(result.solvetime)
    end
    # save data
    instance_data = Dict{Any, Any}(
        "target" => distname,
        "n" => i,
        "d" => d,
        "seed" => seed,
        "discrepancytype" => discrepancytype,
        "discrepancy" => discrepancy,
        "nprocs" => nprocs(),
        "solvetime" => solvetime,
    )
    save_json(
        instance_data;
        dir="sample_target_mismatch-multivariate_uniform_vs_beta-kernel_vs_graph",
        target=distname,
        discrepancytype=discrepancytype,
        n=i,
        d=d,
        ncores=nprocs(),
        seed=seed
    )
end

@printf("COMPLETE!")
