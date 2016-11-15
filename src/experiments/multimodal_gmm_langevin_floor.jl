# multimodal_gmm_langevin_floor
#
# Considers the stein discrepancy for a 1-d GMM with gap between modes
# g > 0. It uses a bad distribution [which only samples from one mode]
# and the actual iid sequence.

using Distributions: Normal

using SteinDistributions: SteinGMM
using SteinDiscrepancy: stein_discrepancy, wasserstein1d

include("experiment_utils.jl")

# this controls the seed used for the approximating the wasserstein metric
@parseintcli seed "s" "seed" 7
# the gap between the two modes
@parseintcli gap "g" "gap" 5
# the gap between the two modes
@parsestringcli sampler "q" "sampler" "iid"

# select largest n
maxn = 30000
# Select an optimization problem solver
#solver = "clp"
solver = "gurobi"
# dimension
d = 1
# the number of components
K = 2
# distribution sampled from
distname = "$K-gmm"
# target distribution
gmmmus = gap .* [-0.5, 0.5]
gmmsigmas = [1.0, 1.0]
gmmweights = [0.5, 0.5]
target = SteinGMM(gmmmus, gmmsigmas)
# Sample sizes at which optimization problem will be solved
ns = vcat(10:10:min(100,maxn),
          200:100:min(1000,maxn),
          2000:1000:min(10000,maxn),
          20000:10000:min(100000,maxn))

if sampler == "iid"
    X = rand(target, maxn)
elseif sampler == "unimodal"
    wrongtarget = Normal(gmmmus[1], gmmsigmas[1,1])
    X = rand(wrongtarget, maxn)
end

@printf("Beginning optimization\n")
for i in ns
    res = stein_discrepancy(points=X[1:i,:],
                            target=target,
                            solver=solver)
    println("\tn = $(i), objective = $(res.objectivevalue)")
    wasserstein = Inf
    wasserstein_lb = Inf
    wasserstein_ub = Inf
    (wasserstein, error) = wasserstein1d(X[1:i,:], target=target)
    wasserstein_lb = wasserstein - error
    wasserstein_ub = wasserstein + error

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
        "g" => res.g,
        "gprime" => res.gradg,
        "objectivevalue" => res.objectivevalue,
        "operatorg" => res.operatorg,
        "wasserstein" => wasserstein,
        "wasserstein_lb" => wasserstein_lb,
        "wasserstein_ub" => wasserstein_ub,
        "edgetime" => res.edgetime,
        "solvetime" => res.solvetime,
    )

    save_json(
        instance_data;
        dir="multimodal_gmm_langevin_floor",
        distname=distname,
        sampler=sampler,
        n=i,
        d=d,
        gap=gap,
        seed=seed,
    )
end

@printf("COMPLETE!\n")
