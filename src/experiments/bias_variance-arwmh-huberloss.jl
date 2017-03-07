# bias_variance-arwmh-pseduohuber
#
# This script compares the Stein discrepancy of sample sequences drawn
# from asymptotically biased approximate random walk Metropolis-Hastings
# (ARWMH) chains (http://arxiv.org/pdf/1304.5299v4.pdf) and asymptotically
# exact RWMH chains.  By plotting the Stein discrepancy results against
# a measure of computation required per sample (e.g., the number of
# datapoint likelihood evaluations needed), we can quantify a bias-variance
# trade-off underlying the approximate MCMC scheme.
#
# The target distribution is a Bayesian huber loss regression posterior
# under independent Gaussian priors based on Gelman's Radon dataset.
# The regression tries to model the relationship between a few exploratory
# variables and the levels of Radon present in houses.

using DataFrames

using SteinDiscrepancy: stein_discrepancy
using SteinDistributions: SteinHuberLossRegressionGaussianPrior
using SteinSamplers: runapproxmh

include("experiment_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7
# set the epsilon param
@parsestringcli epsiloninput "e" "epsilon" "1e-5"
epsilon = float(epsiloninput)
# Select a solver for Stein discrepancy optimization problem
#solver = "clp"
solver = "gurobi"

# Setup the data
data_set = "radon"
rdat = readtable("src/experiments/radon/srrs2.dat")
cdat = readtable("src/experiments/radon/cty.dat")
rename!(cdat, :ctfips, :cntyfips)

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
# Parameter dimension
d = size(X,2)
# the max number of samples to run in the solver
n = 1000
# Associate sample weights
q = ones(n, 1) ./ n
# name of prior + posterior
distname = "huberloss-gaussianprior"
# setup distribution
target = SteinHuberLossRegressionGaussianPrior(X, y)
# intiial guess is OLS estimate
beta0 = inv(X' * X) * (X' * y)
# burnin likelihoods
burninlikelihoods = 1e5
# max number of likelihoods
maxlikelihoods = 1e7
# batchsize for different algorithms [ataset is ~1000 rows]
batchsize = 5
# Sample sizes at which Stein discrepancy will be computed
ns = vcat(10:10:min(400,n), 420:20:min(600,n), 650:50:min(1000,n))

@printf("Generating samples for epsilon=%f\n", epsilon)

(betas, numlikelihoods) =
    @setseed seed runapproxmh(target, beta0;
                              epsilon=epsilon,
                              numlikelihood=maxlikelihoods,
                              batchsize=batchsize,
                              proposalvariance=1e-4)

# Thin to n samples
firstn = find([(l >= burninlikelihoods) for l in numlikelihoods])[1]
lastn = length(numlikelihoods)
idx = round(Int, linspace(firstn, lastn, min(lastn - firstn + 1, n)))

# Keep track of yhats
yhat = X * betas[idx,:]'
scaledyhat = yhat ./ vec(maximum(X, 2))

for i in ns
    println("Beginning optimization for n=$(i), d=$(d), epsilon=$(epsilon)")
    # Compute Stein discrepancy for first i points
    betai = betas[idx[1:i],:]
    res = stein_discrepancy(points=betai,
                            target=target,
                            solver=solver)
    println("\tn = $(i), objective = $(res.objectivevalue)")

    instance_data = Dict{Any, Any}(
        "n" => i,
        "distname" => distname,
        "d" => d,
        "seed" => seed,
        "epsilon" => epsilon,
        "batchsize" => batchsize,
        "X" => res.points,
        "q" => res.weights,
        "xnorm" => res.xnorm,
        "g" => res.g,
        "gradg" => res.gradg,
        "objectivevalue" => vec(res.objectivevalue),
        "scaled.yhat" => vec(mean(scaledyhat[:,1:i], 2)),
        "numlikelihood" => numlikelihoods[idx[i]],
        "edgetime" => res.edgetime,
        "solvetime" => res.solvetime,
    )

    save_json(
        instance_data;
        dir="bias_variance-arwmh-huberloss",
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
