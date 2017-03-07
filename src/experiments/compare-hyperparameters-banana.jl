# compare-hyperparameters-banana
#
# This script compares the quality of samples drawn
# for the banana dataset.

# parallelize now!
addprocs(CPU_CORES - 1)

using SteinDiscrepancy: stein_discrepancy
using SteinDistributions:
    SteinBanana,
    randbanana,
    coercivefunction
using SteinKernels:
    SteinGaussianKernel,
    SteinGaussianPowerKernel,
    SteinMaternRadialKernel,
    SteinInverseMultiquadricKernel
using SteinSamplers: runsgld, runapproxslice, runapproxmh

include("experiment_utils.jl")

# random number generator seed
# this controls the seed used for the subsampling
@parseintcli seed "s" "seed" 7
# set the sampler
@parsestringcli sampler "q" "sampler" "approxslice"
# set the sampler
@parsestringcli subsamplescorefraction "f" "fraction" "0.0"
# set the discrepancy
@parsestringcli kernelname "k" "kernelname" "maternradial"
# distname is bananas!
distname = "banana-posterior"
# sampler seed
sampler_seed = 7
# number of banana samples
N = 100
# the max number of likelihood evals to run for
# (if seed=7, epsilon=0.5, sampler=approxslice, this is n=10000)
maxevals = 1823670
# the dimension of the data
d = 2
# the true value of theta1 + theta2^2
truetheta0 = 1.0
# draw banana samples
y = @setseed sampler_seed randbanana(truetheta0, N)
# create banana posterior distribution
target = SteinBanana(y)
# starting theta0
theta0 = randn(d)
# get the subsamplesize
subsamplescoresize = 0
if subsamplescorefraction != "0.0"
    subsamplescoresize = round(Int, float(subsamplescorefraction) * N)
end
# set the kernel
kernel = nothing
if kernelname == "gaussian"
    kernel = SteinGaussianKernel()
elseif kernelname == "gaussianpower"
    kernel = SteinGaussianPowerKernel()
elseif kernelname == "maternradial"
    kernel = SteinMaternRadialKernel()
elseif kernelname == "inversemultiquadric"
    kernel = SteinInverseMultiquadricKernel()
end
# batchsize for SGLD algorithm
batchsize = 30
# the SGLD epsilons to run the experiemnt
sgld_epsilons = [1e-2, 5e-3, 1e-3, 5e-4]
# the approx slice epsilons to run the experiemnt
approxslice_epsilons = [0.0, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1]
# the approx slice epsilons to run the experiemnt
approxmh_epsilons = [0.0, 1e-3, 1e-2, 1e-1, 2e-1]
# Define sampler
if sampler == "sgld"
    # the number of sweeps to make through the data
    n = 10000
    numsweeps = round(Int, ceil(n / floor(N/batchsize)))
    epsilons = sgld_epsilons
    runsampler(x0,epsilon) = runsgld(target, x0;
                                     epsilonfunc=t -> epsilon,
                                     numsweeps=numsweeps,
                                     batchsize=batchsize)
elseif sampler == "approxslice"
    epsilons = approxslice_epsilons
    runsampler(x0,epsilon) = runapproxslice(target, x0;
                                            epsilon=epsilon,
                                            numlikelihood=maxevals,
                                            batchsize=batchsize)
elseif sampler == "approxmh"
    epsilons = approxmh_epsilons
    runsampler(x0,epsilon) = runapproxmh(target, x0;
                                         epsilon=epsilon,
                                         numlikelihood=maxevals,
                                         batchsize=batchsize)
end

for epsilon in epsilons
    println("Generating samples for sampler=$(sampler),epsilon=$(epsilon)")

    thetas = []; numeval = NaN
    @trycatchcontinue(
        begin
            (thetas, numeval) = @setseed sampler_seed runsampler(theta0,epsilon)
            if any(isnan(thetas))
                println("[$(distname):seed=$(seed),epsilon=$(epsilon)] NaNs found. Skipping.")
                continue
            end
        end,
        println("[$(distname):seed=$(seed),epsilon=$(epsilon)]:")
    )

    # Find stein discrepancy
    @printf("[Beginning kernel computations]\n")
    result = @setseed seed stein_discrepancy(points=thetas,
                                             target=target,
                                             method="kernel",
                                             kernel=kernel,
                                             subsamplescoresize=subsamplescoresize)

    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "n" => size(thetas,1),
        "d" => d,
        "seed" => seed,
        "epsilon" => epsilon,
        "thetas" => thetas,
        "discrepancy2" => result.discrepancy2,
        "numeval" => numeval,
        "sampler" => sampler,
        "batchsize" => batchsize,
        "kernel" => kernelname,
        "nprocs" => nprocs(),
        "solvertime" => result.solvetime,
        "subsamplescorefraction" => subsamplescorefraction,
        "sampler_seed" => sampler_seed,
    )

    save_json(
        instance_data;
        dir="compare-hyperparameters-banana",
        distname=distname,
        n=size(thetas,1),
        sampler=sampler,
        epsilon=epsilon,
        batchsize=batchsize,
        subsamplescorefraction=subsamplescorefraction,
        kernel=kernelname,
        nprocs=nprocs(),
        d=d,
        seed=seed,
        sampler_seed=sampler_seed,
    )
end

# print out the y's used for visualization
save_json(
    Dict{Any, Any}("y" => y);
    dir="compare-hyperparameters-banana-y",
    distname="banana-y",
    N=N,
    sampler_seed=sampler_seed,
)
