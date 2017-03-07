# sample_target_mismatch-gaussian_vs_t-kernel_vs_graph
#
# This script compares the values of the univariate graph Stein
# discrepancy and kernel discrepancy obtained when samples are drawn i.i.d.
# from their targets.

using SteinDistributions: SteinGaussian, SteinScaleLocationStudentT
using SteinKernels: SteinGaussianKernel
using SteinDiscrepancy: stein_discrepancy

include("experiment_utils.jl")

# random number generator seed
# we allow this to be set on the commandline
@parseintcli seed "s" "seed" 7
# set the target distribution
@parsestringcli distname "t" "target" "gaussian"
# set the discrepancy
@parsestringcli discrepancytype "d" "discrepancy" "gaussiankernel"
# Specify dimension of sampled vectors
d = 1
# Specify a maximum number of samples
maxn = round(Int, 5e5)
# gaussian target
gaussian_target = SteinGaussian(d)
# the graphsolver
graphsolver = "gurobi"
#graphsolver = "clp"
# setup kernel (if necessary)
kernel = nothing
if discrepancytype == "gaussiankernel"
    kernel = SteinGaussianKernel()
end
# setup target
if distname == "studentt"
    df = 10.0
    target = SteinScaleLocationStudentT(df, 0.0, sqrt((df-2.0)/df))
elseif distname == "gaussian"
    target = SteinGaussian(d)
end

# Draw samples from target distribution
X = @setseed rand(target,maxn)
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:min(1000,maxn),
          2000:1000:min(10000,maxn),
          20000:10000:min(100000,maxn),
          200000:100000:maxn)

# Solve optimization problem at each sample size
for i in ns
    @printf("[Beginning n=%d]:\n", i)
    # Create program with data subset
    Xi = X[1:i,:]
    discrepancy = solvetime = nothing
    if kernel != nothing
        @printf("[Beginning kernel computations n=%d]\n", i)
        result = stein_discrepancy(points=Xi,
                                   target=gaussian_target,
                                   method="kernel",
                                   kernel=kernel)
        discrepancy = sqrt(result.discrepancy2)
        solvetime = result.solvetime
    elseif discrepancytype == "graph"
        @printf("[Beginning graph solver n=%d]\n", i)
        result = stein_discrepancy(points=Xi,
                                   target=gaussian_target,
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
        "solvetime" => solvetime,
    )
    save_json(
        instance_data;
        dir="sample_target_mismatch-gaussian_vs_t-kernel_vs_graph",
        target=distname,
        discrepancytype=discrepancytype,
        n=i,
        d=d,
        seed=seed
    )
end

@printf("COMPLETE!")
