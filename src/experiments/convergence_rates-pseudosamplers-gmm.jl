# Convergence experiment (convergence_rates-pseudosamplers-gmm)
#
# Compares the Stein discrepancy convergence behavior of herding
# with that of independent random sampling. Make sure to run
# the matlab script at
#
# src/experiments/skh/MoG_experiments/script_test.m
#
# with all the desired parameters used here.

using MAT

using SteinDistributions: SteinGMM, SteinDiscrete
using SteinDiscrepancy: stein_discrepancy, wasserstein1d, approxwasserstein

include("experiment_utils.jl")

# this controls the seed used for the approximating the wasserstein metric
@parseintcli seed "s" "seed" 7
# the dimension of the space
@parseintcli d "d" "dimension" 1
# the sampler
@parsestringcli sampler "q" "sampler" "IID"
# the gap between the two modes
@parseintcli gap "g" "gap" 5
# approx wasserstein sample sizes
@parseintcli wasserstein_n "w" "wassersteinn" 5000

# Solve optimization problem for each sampler at each sample size
# samplers = ["FW", "FCFW", "IID", "QMC"]
# Select an optimization problem solver
#solver = "clp"
solver = "gurobi"
# the number of components
K = 2
# distribution sampled from
distname = "$K-gmm"
# get the results directory of the sample files
resultsdir = joinpath("src", "experiments", "skh", "MoG_experiments", "results")

# open dat matlab file
samplefile = @sprintf("matlab_skh_gmm_method=%s_d=%d_comps=%d_gap=%d.mat",
                       sampler, d, K, gap)
samplefilepath = joinpath(resultsdir, samplefile)
samples = matread(samplefilepath)
# get some params from the data
X = samples["X"]
n = size(X,1)
pointweights = samples["weights"]
gmmweights = vec(samples["pi_prob"])
gmmsigmas = samples["Sigma_mix"]
gmmmus = samples["mu_mix"]
target = SteinGMM(gmmmus, gmmsigmas, gmmweights)
if sampler == "IID"
    # we use our own iid sampler so we can run multiple seeds
    X = rand(target, n)
end
# Sample sizes at which optimization problem will be solved
ns = vcat(10:10:min(1000,n))

@printf("Beginning optimization for %s, dimension=%d\n", sampler, d)
for i in ns
    sampleweights = vec(pointweights[i+1,1:i])
    # Compute Stein discrepancy for first i points in this trial
    res = stein_discrepancy(points=X[1:i,:],
                            weights=sampleweights,
                            target=target,
                            solver=solver)
    println("\tn = $(i), objective = $(res.objectivevalue)")
    wasserstein = Inf
    wasserstein_lb = Inf
    wasserstein_ub = Inf
    if d == 1
       (wasserstein, error) = wasserstein1d(X[1:i,:],
                                            weights=sampleweights,
                                            target=target)
        wasserstein_lb = wasserstein - error
        wasserstein_ub = wasserstein + error
    else
        (wasserstein_lb, wasserstein_ub) =
            @setseed seed approxwasserstein(points=X[1:i,:],
                                            weights=sampleweights,
                                            target=target,
                                            solver=solver,
                                            samplesize=wasserstein_n)
        wasserstein = (wasserstein_lb + wasserstein_ub) / 2
    end
    # Package and save results
    instance_data = Dict{Any, Any}(
        "seed" => seed,
        "mus" => gmmmus,
        "sigmas" => gmmsigmas,
        "distweights" => gmmweights,
        "sampler" => sampler,
        "gap" => gap,
        "distname" => distname,
        "n" => i,
        "d" => d,
        "X" => res.points,
        "q" => res.weights,
        "g" => res.g,
        "gprime" => res.gradg,
        "objectivevalue" => res.objectivevalue,
        "operatorg" => res.operatorg,
        "wasserstein" => wasserstein,
        "wasserstein_lb" => wasserstein_lb,
        "wasserstein_ub" => wasserstein_ub,
        "edgetime" => res.edgetime,
        "solvetime" => res.solvetime,
        "wasserstein_n" => wasserstein_n,
    )

    save_json(
        instance_data;
        dir="convergence_rates-pseudosamplers-gmm",
        distname=distname,
        sampler=sampler,
        n=i,
        d=d,
        gap=gap,
        wassersteinn=wasserstein_n,
        seed=seed,
    )
end

@printf("COMPLETE!\n")
