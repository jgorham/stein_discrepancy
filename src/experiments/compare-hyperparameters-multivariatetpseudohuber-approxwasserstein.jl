# compare-hyperparameters-multivariatetpseudohuber-approxwasserstein
#
# This script tries to compute an approximation of the wasserstein metric
# for the samples drawn from the Bayesian regession for the AIS dataset.

using SteinDiscrepancy: wassersteindiscrete, wasserstein1d

include("experiment_utils.jl")

# this controls the seed used for the subsampling
@parseintcli seed "s" "seed" 7
# set the sampler
@parsestringcli sampler "q" "sampler" "runsgrld"
# set the delta param for the Huber prior
@parsestringcli nu "n" "nu" "10.0"
# set the nu param degrees of freedom for the d-distribution
@parsestringcli delta "d" "delta" "0.1"
# Select a solver for Stein discrepancy optimization problem
#solver = "clp"
solver = "gurobi"
# the n & d used
n = 2000; d = 4
# the thinby parameter
thinby = 100
# distname is multivariatetpseudohuber
distname = "multivariatetpseudohuber"
# the SGRLD epsilons to run the experiemnt
epsilons = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# batchsize
batchsize = 30
# wasserstein n - the n to thin from
wasserstein_n = 5000
# the directory the data is stored
dir = "compare-hyperparameters-multivariatetpseudohuber"
dirout = @sprintf("%s-approxwasserstein", dir)

# prep the MARLA data
marla_jsonpath = @sprintf(
    "results/%s-MARLA/data/julia_distname=%s_dataset=ais_n=2000000_thinby=20_epsilon=0.01_nu=%s_delta=%s_d=4_seed=7.json",
     dir,
     distname,
     nu,
     delta)

marla_json = load_json(marla_jsonpath)
marla_betas0 = marla_json["betas"]
marla_betas = Array(Float64, wasserstein_n, d)

for epsilon in epsilons
    println("Computing approx wasserstein for epsilon=$(epsilon)")
    epsilon = string(epsilon)
    # load up the jsons
    sample_jsonpath = @sprintf(
        "results/%s/data/julia_distname=%s_n=%s_thinby=%s_sampler=%s_epsilon=%s_nu=%s_delta=%s_batchsize=%s_d=%s_seed=%s.json",
         dir,
         distname,
         n,
         thinby,
         sampler,
         epsilon,
         nu,
         delta,
         batchsize,
         d,
         seed)
    sample_json = load_json(sample_jsonpath)
    # prep the data for computing
    sample_betas0 = sample_json["betas"]
    sample_betas = Array(Float64, n, d)
    for j in 1:d
        sample_betas[:,j] = convert(Array{Float64,1}, sample_betas0[j])
        thinrate = round(Int, length(marla_betas0[1]) / wasserstein_n)
        marla_betas[:,j] = convert(Array{Float64,1},
                                   marla_betas0[j][thinrate:thinrate:(wasserstein_n*thinrate)])
    end
    # conmpute the marginal 1d and 2d wasserstein metrics
    marginal1d = Dict{Any,Any}[]
    marginal2d = Dict{Any,Any}[]
    for ii in 1:d
        for jj in 1:d
            if ii >= jj
                continue
            end
            tic()
            (wasserstein, numnodes, numedges, status) =
                wassersteindiscrete(xpoints=sample_betas[:,[ii,jj]],
                                    ypoints=marla_betas[:,[ii,jj]],
                                    solver=solver)
            solvetime = toc()
            result = Dict{Any, Any}(
                "wasserstein" => wasserstein,
                "numnodes" => numnodes,
                "numedges" => numedges,
                "wasserstein_n" => wasserstein_n,
                "betai" => ii,
                "betaj" => jj,
                "solvetime" => solvetime,
            )
            push!(marginal2d, result)
        end
        all_marla_betas = convert(Array{Float64,1}, marla_betas0[ii])
        (wasserstein, numnodes, numedges, status) =
            wasserstein1d(xpoints=sample_betas[:,ii],
                          ypoints=all_marla_betas)
        result = Dict{Any, Any}(
            "wasserstein" => wasserstein,
            "betai" => ii,
        )
        push!(marginal1d, result)
    end

    sample_json["marginal1d_wasserstein"] = marginal1d
    sample_json["marginal2d_wasserstein"] = marginal2d

    save_json(
        sample_json;
        dir=dirout,
        distname=distname,
        n=n,
        thinby=thinby,
        sampler=sampler,
        epsilon=epsilon,
        nu=nu,
        delta=delta,
        batchsize=batchsize,
        d=d,
        seed=seed,
        wassn=wasserstein_n
    )
end
