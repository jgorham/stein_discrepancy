# surrogate_ground_truth_mnist_7_or_9_mala
#
# This script runs a long chain based on the
# Metropolis-adjusted Langevin algorithm (MALA). This is used as a gold
# standard as this procedure yields an asymptotically unbiased chain
# and thus is used to compare the efficiency of the biased chains produced
# in the experiment mnist_7_or_9_sgfs_sgld.

using StatsFuns: logistic
using SteinDistributions: SteinLogisticRegressionGaussianPrior
using SteinSamplers: runmala

include("experiment_utils.jl")
include("mnist/mnist_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7
# setup the data
data_set = "mnist"
# number of components in reduced matrix
L = 50
# labels of the y data to keep around
labels = [7,9]
# get eigenfaces
V = mnist_eigenfaces(L, labels)
# setup training
X, y, Xtest, ytest = mnist_reduce_dimension_pca(L, labels)
# get the training and test sizes
ntrain = size(X, 1); ntest = size(Xtest, 1)
# give X an intercept term
X = hcat(ones(ntrain) ./ sqrt(ntrain), X)
# give Xtest an intercept term (the sqrt(ntrain) is not a mistake!)
Xtest = hcat(ones(ntest) ./ sqrt(ntrain), Xtest)
# make y be -1 for 7 and 1 for 9
y = 2.0 * (y .== 9) - 1.0
# make ytest be 0 for 7 and 1 for 9
ytest = round(Int, ytest .== 9)
# Parameter dimension
d = L + 1
# name of prior + posterior
distname = "logisticgaussianprior"
# sigma2 term for prior
sigma2 = 1.0
# setup distribution
target = SteinLogisticRegressionGaussianPrior(X, y, zeros(d), sigma2 .* ones(d))
# the max number of samples to run in the solver
n = 500000
# number of burnin samples
b = 10000
# the epsilons to run the experiemnt
epsilons = [5e-2]
# initial guess for beta
beta0 = zeros(d)

for epsilon = epsilons
    # Draw MALA samples
    (betas_mala, numgrad_mala) =
        @setseed runmala(target, beta0;
                         epsilonfunc=t -> epsilon,
                         numiter=n+b)

    betas_mala = betas_mala[(b+1):(b+n),:]
    # beta estimates
    betahat = vec(mean(betas_mala, 1))
    sigmahat = cov(betas_mala)
    # Keep track of predictive probabilities
    predprobs = logistic(Xtest * betas_mala')
    yhat = vec(mean(predprobs, 2))
    # Package and save results
    summary_data = Dict{Any, Any}(
        "epsilon" => epsilon,
        "distname" => distname,
        "n" => n,
        "d" => d,
        "seed" => seed,
        "eigenfaces" => V,
        "betahat" => betahat,
        "sigmahat" => sigmahat,
        "mean.pred.probs" => yhat,
        "true.pred" => ytest,
        "epsilon" => epsilon,
        "numgrad" => numgrad_mala,
        "sampler" => "MALA-long-nodal",
    )
    save_json(
        summary_data;
        dir="mnist_7_or_9_mala",
        distname=distname,
        dataset=data_set,
        n=n,
        epsilon=epsilon,
        priorvariance=sigma2,
        d=d,
        seed=seed
    )
end
