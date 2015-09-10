# surrogate_ground_truth-mala-logistic
#
# This script runs a long chain based on the
# Metropolis-adjusted Langevin algorithm (MALA). This is used as a gold
# standard as this procedure yields an asymptotically unbiased chain
# and thus is used to compare the efficiency of the biased chains produced
# in the experiment bias_variance-arwmh-logistic.
#
# The target distribution is a Bayesian logistic regression posterior
# under independent Gaussian priors based on a prostate cancer dataset with
# binary outcomes indicating whether cancer has spread to surrounding lymph
# nodes.

using Distributions: Normal
using RDatasets
using StatsBase: logistic
using SteinDistributions: SteinLogisticRegressionGaussianPrior
using SteinSamplers: runmala

include("experiment_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7
# setup the data
data_set = "nodal"
X = y = None
nodal = dataset("boot", "nodal")
y = 2 * convert(Array{Float64},nodal[:R]) - 1
# X includes an intercept term
X = array(nodal[[:M, :Aged, :Stage, :Grade, :Xray, :Acid]])
X = convert(Array{Float64, 2}, X)

# N is number of samples, p is dimension of parameter
N, d = size(X)
# the max number of samples to run in the solver
n = 10000000
# setup distribution
target = SteinLogisticRegressionGaussianPrior(X, y)
# name of prior + posterior
distname = "logisticgaussianprior"
# intiial guess for beta
beta0 = zeros(d)
# the epsilons to run the experiemnt
epsilons = [1e-2]

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
    # compute the running estimated mu (i.e. P(Y_i = 1 | X_i))
    # when using the first k values from the chain
    ks = int(linspace(0, n, 1001))[2:end]
    meanpredprobs = Array(Float64, N, length(ks))
    for (i,k) in enumerate(ks)
        meanbeta = mean(betas_mala[1:k,:], 1)
        predprobs = logistic(X * meanbeta')
        meanpredprobs[:,i] = vec(predprobs)
    end
    # scale the mu's by the l_infty norm of each X_i
    scaledpredprobs = meanpredprobs ./ vec(maximum(X, 2))
    # fetch constants for computing confidence intervals of beta
    phi = Normal()
    zscore = quantile(phi, 1 - 0.05/2)
    # Package and save results
    summary_data = Dict({
        "n" => n,
        "seed" => seed,
        "distname" => distname,
        "epsilon" => epsilon,
        "beta.hat" => betahat,
        "beta.vcov" => betavcov,
        "beta.sd" => betasd,
        "beta.hat.low" => betahat - zscore .* betasd,
        "beta.hat.high" => betahat + zscore .* betasd,
        "scaled.pred.probs" => scaledpredprobs,
        "scaled.pred.probs.i" => ks,
        "numgrad" => numgrad_mala,
        "sampler" => "MALA-long-nodal"
    })
    save_json(
        summary_data;
        dir="bias_variance-arwmh-logistic-MALA",
        distname=distname,
        dataset=data_set,
        n=n,
        epsilon=epsilon,
        d=d,
        seed=seed
    )
end
