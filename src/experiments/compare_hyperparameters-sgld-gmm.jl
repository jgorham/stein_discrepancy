# compare_hyperparameters-sgld-gmm
#
# This script compares the quality of samples generated from different
# step size hyperparameter settings of a Stochastic Gradient Langevin Dynamics
# sampler for a Gaussian mixture model posterior target distribution.

using Iterators: product

using SteinDistributions: SteinGMMPosterior, randgmm, numdimensions, numdatapoints
using SteinDiscrepancy: stein_discrepancy
using SteinSamplers: runsgld

include("experiment_utils.jl")

## Experiment settings
# Target size of sample to draw from sampler
n = 1000
# Run one complete experiment per random number generator seed
seeds = 1:50
# Select a solver for Stein discrepancy optimization problem
#solver = "clp"
solver = "gurobi"
## Sampler settings for SGLD
distname = "sgld-gmm"

## Target distribution settings for SteinGMMPosterior
# Generate random dataset from GMM
truex = [0.0, 1.0] # Parameter vector used to generate dataset
L = 100 # Number of datapoints to generate
data_generation_seed = 1
y = @setseed data_generation_seed randgmm(truex, L)
# save the y values for later
save_json(
    Dict({"y" => y});
    dir="compare_hyperparameters-$(distname)-y",
    numsamples=L,
    x=string(truex)
)
# Create posterior distribution based on dataset
target = SteinGMMPosterior(y)
# Fix mini-batch size
batchsize = 5
# Number of sweeps to make through the data
numsweeps = int(ceil(n / floor(numdatapoints(target)/batchsize)))
# Hyperparameter to vary: step size epsilon
epsilons = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
# Define SGLD sampler with step size epsilon and starting sample value x0
sampler(x0,epsilon) = runsgld(target, x0;
                              epsilonfunc=t -> epsilon,
                              numsweeps=numsweeps,
                              batchsize=batchsize)

## Draw and evaluate samples for different hyperparameter and seed combinations
d = numdimensions(target)
for (epsilon, seed) in product(epsilons, seeds)
    # Generate starting sample value x0
    x0 = @setseed seed randn(d)

    println("Generating sample for seed=$(seed), epsilon=$(epsilon)")
    # Run sampler
    X = []; numgrad = NaN
    @trycatchcontinue(
        begin
            (X, numgrad) = @setseed seed sampler(x0,epsilon)
            if any(isnan(X))
                println("[$(distname):seed=$(seed),epsilon=$(epsilon)] NaNs found. Skipping.")
                continue
            end
        end,
        println("[$(distname):seed=$(seed),epsilon=$(epsilon)]:")
    )

    # Compute Stein discrepancy for first n points in sample
    # (with equal point weights)
    println("Computing Stein discrepancy for n=$(n), d=$(d), dist=$(distname)")
    res = None
    @trycatchcontinue(
        begin
            res = stein_discrepancy(points=X[1:n,:], target=target, solver=solver)
        end,
        println("[n=$(n)|sampler=$(distname)]:")
    )
    println("\tn = $(n), objective = $(res.objectivevalue)")

    # Package results
    edges = res.edges
    edge_pairs = [vec(edges[i, :]) for i in 1:size(edges, 1)]
    instance_data = Dict({
        "distname" => distname,
        "n" => n,
        "solver" => solver,
        "d" => d,
        "seed" => seed,
        "epsilon" => epsilon,
        "X" => X[1:n,:],
        "q" => res.weights,
        "xnorm" => res.xnorm,
        "edges" => edge_pairs,
        "g" => res.g,
        "gradg" => res.gradg,
        "objectivevalue" => res.objectivevalue,
        "numgrad" => numgrad * (n / size(X, 1)),
        "edgetime" => res.edgetime,
        "solvetime" => res.solvetime
    })

    # Save results
    save_json(
        instance_data;
        dir="compare_hyperparameters-$(distname)",
        distname=distname,
        n=n,
        solver=solver,
        epsilon=epsilon,
        d=d,
        seed=seed
    )
end
