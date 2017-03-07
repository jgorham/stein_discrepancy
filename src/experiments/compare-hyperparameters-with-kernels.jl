# compare-hyperparameters-with-kernels
#
# This script compares the quality of samples generated from different
# step size hyperparameter settings of a Stochastic Gradient Langevin Dynamics
# sampler for a Gaussian mixture model posterior target distribution.

using Iterators: product

using SteinDistributions:
    SteinGMMPosterior,
    randgmm,
    numdimensions,
    numdatapoints
using SteinKernels:
    SteinGaussianKernel,
    SteinMaternRadialKernel,
    SteinGaussianPowerKernel,
    SteinInverseMultiquadricKernel
using SteinDiscrepancy:
    stein_discrepancy
using SteinSamplers: runsgld

include("experiment_utils.jl")

# set the discrepancy
@parsestringcli discrepancytype "k" "discrepancytype" "maternradial"

## Experiment settings
# Target size of sample to draw from sampler
n = 1000
# Run one complete experiment per random number generator seed
seeds = 1:1
# Sampler settings for SGLD
distname = "sgld-gmm"
# set the kernel
kernel = nothing
if discrepancytype == "maternradial"
    kernel = SteinMaternRadialKernel()
elseif discrepancytype == "gaussian"
    kernel = SteinGaussianKernel()
elseif discrepancytype == "gaussianpower"
    kernel = SteinGaussianPowerKernel(2.0)
elseif discrepancytype == "inversemultiquadric"
    kernel = SteinInverseMultiquadricKernel(0.5)
end

## Target distribution settings for SteinGMMPosterior
# Generate random dataset from GMM
truex = [0.0, 1.0] # Parameter vector used to generate dataset
L = 100 # Number of datapoints to generate
data_generation_seed = 1
y = @setseed data_generation_seed randgmm(truex, L)
# save the y values for later
save_json(
    Dict{Any, Any}("y" => y);
    dir="compare-hyperparameters-with-kernels-y",
    numsamples=L,
    x=string(truex)
)
# Create posterior distribution based on dataset
target = SteinGMMPosterior(y)
# Fix mini-batch size
batchsize = 5
# Number of sweeps to make through the data
numsweeps = round(Int, ceil(n / floor(numdatapoints(target)/batchsize)))
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
    res = nothing
    @trycatchcontinue(
        begin
            res = stein_discrepancy(points=X[1:n,:],
                                    target=target,
                                    method="kernel",
                                    kernel=kernel)
        end,
        println("[n=$(n)|sampler=$(distname)]:")
    )
    if res == nothing
        continue
    end
    # compute the coercive function
    kerneldiscrepancy = sqrt(res.discrepancy2)

    # Package results
    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "discrepancytype" => discrepancytype,
        "n" => n,
        "d" => d,
        "seed" => seed,
        "epsilon" => epsilon,
        "X" => X[1:n,:],
        "q" => res.weights,
        "steindiscrepancy" => kerneldiscrepancy,
        "numgrad" => numgrad * (n / size(X, 1)),
        "solvetime" => res.solvetime,
    )

    # Save results
    save_json(
        instance_data;
        dir="compare-hyperparameters-with-kernels",
        distname=distname,
        discrepancytype=discrepancytype,
        n=n,
        epsilon=epsilon,
        d=d,
        seed=seed
    )
end
