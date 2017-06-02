# bias_variance-arwmh-logistic
#
# This script compares the Stein discrepancy of sample sequences drawn
# from asymptotically biased approximate random walk Metropolis-Hastings
# (ARWMH) chains (http://arxiv.org/pdf/1304.5299v4.pdf) and asymptotically
# exact RWMH chains.  By plotting the Stein discrepancy results against
# a measure of computation required per sample (e.g., the number of
# datapoint likelihood evaluations needed), we can quantify a bias-variance
# trade-off underlying the approximate MCMC scheme.
#
# The target distribution is a Bayesian logistic regression posterior
# under independent Gaussian priors based on a prostate cancer dataset with
# binary outcomes indicating whether cancer has spread to surrounding lymph
# nodes.

include("experiment_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 37

using RDatasets
using StatsFuns: logistic
using SteinDiscrepancy: gsd
using SteinDistributions: SteinLogisticRegressionGaussianPrior, gradlogdensity
using SteinSamplers: runapproxmh

# Select a solver for Stein discrepancy optimization problem
#solver = "clp"
solver = "gurobi"

# Setup the data
data_set = "nodal"
nodal = dataset("boot", "nodal")
y = 2 * convert(Array{Float64},nodal[:R]) - 1
# X includes an intercept term
X = convert(Array{Float64, 2}, nodal[[:M, :Aged, :Stage, :Grade, :Xray, :Acid]])

# Parameter dimension
d = size(X,2)
# the max number of samples to run in the solver
n = 1000
# Associate sample weights
q = ones(n, 1) ./ n
# name of prior + posterior
distname = "logisticgaussianprior"
# setup distribution
target = SteinLogisticRegressionGaussianPrior(X, y)
# define gradlogp
function gradlogp(x::Array{Float64,1})
    gradlogdensity(target, x)
end
# intiial guess for beta
beta0 = zeros(d)
# burnin likelihoods
burninlikelihoods = 1e3
# max number of likelihoods
maxlikelihoods = 1e5
# batchsize for different algorithms [nodal set is ~50 rows]
batchsize = 2
# the epsilons to run the experiemnt
epsilons = [0.0, 0.1, 0.2]
# Sample sizes at which Stein discrepancy will be computed
ns = vcat(10:10:min(400,n), 420:20:min(600,n), 650:50:min(1000,n))

for epsilon = epsilons
    @printf("Generating samples for epsilon=%f\n", epsilon)

    (betas, numlikelihoods) =
        @setseed seed runapproxmh(target, beta0;
                                  epsilon=epsilon,
                                  numlikelihood=maxlikelihoods,
                                  batchsize=batchsize,
                                  proposalvariance=0.01)

    # Thin to n samples
    firstn = find([(l >= burninlikelihoods) for l in numlikelihoods])[1]
    lastn = length(numlikelihoods)
    idx = round(Int, linspace(firstn, lastn, min(lastn - firstn + 1, n)))

    # Keep track of predictive probabilities (to use as test functions for
    # external validation of bias-variance trade-off)
    predprobs = logistic(X * betas[idx,:]')
    scaledpredprobs = predprobs ./ vec(maximum(X, 2))

    for i in ns
        println("Beginning optimization for n=$(i), d=$(d), epsilon=$(epsilon)")
        # Compute Stein discrepancy for first i points
        betai = betas[idx[1:i],:]
        res = gsd(points=betai, gradlogdensity=gradlogp, solver=solver)
        println("\tn = $(i), objective = $(res.objectivevalue)")

        # Package and save results
        edges = res.edges
        edge_pairs = [vec(edges[i, :]) for i in 1:size(edges, 1)]

        instance_data = Dict{Any, Any}(
            "n" => i,
            "distname" => distname,
            "d" => d,
            "seed" => seed,
            "epsilon" => epsilon,
            "batchsize" => batchsize,
            "X" => betai,
            "xnorm" => res.xnorm,
            "edges" => edge_pairs,
            "g" => res.g,
            "gradg" => res.gradg,
            "objectivevalue" => vec(res.objectivevalue),
            "scaled.pred.probs" => vec(mean(scaledpredprobs[:,1:i], 2)),
            "numlikelihood" => numlikelihoods[idx[i]],
            "edgetime" => res.edgetime,
            "solvetime" => res.solvetime,
        )

        save_json(
            instance_data;
            dir="bias_variance-arwmh-logistic",
            distname=distname,
            dataset=data_set,
            n=i,
            burninlikelihoods=int(burninlikelihoods),
            epsilon=epsilon,
            batchsize=batchsize,
            d=d,
            seed=seed
        )
    end
end
