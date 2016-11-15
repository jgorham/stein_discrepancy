# surrogate_ground_truth_radon-mala-huberloss
#
# This script runs a long chain based on the
# Metropolis-adjusted Langevin algorithm (MALA). This is used as a gold
# standard as this procedure yields an asymptotically unbiased chain
# and thus is used to compare the efficiency of the biased chains produced
# in the experiment bias_variance-arwmh-huberloss.

using DataFrames
using Distributions: Normal

using SteinDistributions: SteinHuberLossRegressionGaussianPrior
using SteinSamplers: runmala

include("experiment_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7

# Set up the data
data_set = "radon"
rdat = readtable("src/experiments/radon/srrs2.dat")
cdat = readtable("src/experiments/radon/cty.dat")
rename!(cdat, :ctfips, :cntyfips)
# join the two datasets
dat = join(rdat, cdat, on = [:stfips, :cntyfips])
# subset only to Minnesota! (Gelman does this)
dat = dat[dat[:st] .== "MN",:]
# log radon
dat[:logradon] = log(dat[:activity])
dat[dat[:activity] .== 0, :logradon] = log(0.1)
# log uranium
dat[:loguranium] = log(dat[:Uppm])
dat[dat[:Uppm] .== 0, :loguranium] = log(0.1)
# make up X and y
X = Array(Float64, nrow(dat), 3)
X[:,1] = 1.0
X[:,2] = dat[:floor]
X[:,3] = dat[:loguranium]
y = convert(Array, dat[:logradon])

# N is number of samples, p is dimension of parameter
N, d = size(X)
# the max number of samples to run in the solver
n = 100000000
# name of prior + posterior
distname = "huberloss-gaussianprior"
# setup distribution
target = SteinHuberLossRegressionGaussianPrior(X, y)
# intiial guess for beta
beta0 = zeros(d)
# the epsilons to run the experiemnt
epsilons = [1e-4]

for epsilon = epsilons
    # Draw MALA samples
    (betas_mala, numgrad_mala) =
        @setseed runmala(target, beta0;
                         epsilonfunc=t -> epsilon,
                         numiter=n)
    # estimate the final beta moments
    betahat = vec(mean(betas_mala, 1))
    betavcov = cov(betas_mala)
    betasd = sqrt(diag(betavcov))
    # compute the running estimated yhat E{Y | X = x}
    # when using the first k values from the chain
    ks = round(Int, linspace(0, n, 1001))[2:end]
    meanyhats = Array(Float64, N, length(ks))
    for (i,k) in enumerate(ks)
        meanbeta = mean(betas_mala[1:k,:], 1)
        yhats = X * meanbeta'
        meanyhats[:,i] = vec(yhats)
    end
    # scale the mu's by the l_infty norm of each X_i
    scaledyhats = meanyhats ./ vec(maximum(X, 2))
    # fetch constants for computing confidence intervals of beta
    phi = Normal()
    zscore = quantile(phi, 1 - 0.05/2)
    # Package and save results
    summary_data = Dict{Any, Any}(
        "n" => n,
        "seed" => seed,
        "distname" => distname,
        "epsilon" => epsilon,
        "beta.hat" => betahat,
        "beta.vcov" => betavcov,
        "beta.sd" => betasd,
        "beta.hat.low" => betahat - zscore .* betasd,
        "beta.hat.high" => betahat + zscore .* betasd,
        "scaled.yhat" => scaledyhats,
        "scaled.yhat.ii" => ks,
        "numgrad" => numgrad_mala,
        "sampler" => "radon-mala"
    )
    save_json(
        summary_data;
        dir="bias_variance-arwmh-huberloss-MALA",
        distname=distname,
        dataset=data_set,
        n=n,
        epsilon=epsilon,
        d=d,
        seed=seed
    )
end
