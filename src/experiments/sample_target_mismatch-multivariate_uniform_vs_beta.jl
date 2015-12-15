# incorrect_target_divergence_multivariate
#
# Script for testing the divergence of Stein diagnostic bounds
# when the samples aren't drawn from the target distribution.

using Distributions: Beta
using SteinDistributions: SteinUniform
using SteinDiscrepancy: stein_discrepancy

# set a seed
seed = 7
srand(seed)
# Specify dimension of sampled vectors
d = 2
# Specify number of samples
n = 1000
# select sample distribution
beta_target = Beta(3, 3)
# select target distribution
uniform_target = SteinUniform(d)
# Select a solver for Stein discrepancy optimization
solver = "clp"
#solver = "gurobi"
# Draw samples from target distribution
uniformX = rand(uniform_target, n)
# Draw samples from target distribution
betaX = reshape(rand(beta_target, d*n), n, d)

uniform_result = stein_discrepancy(points=uniformX,
                                   target=uniform_target,
                                   solver=solver)

beta_result = stein_discrepancy(points=betaX,
                                target=uniform_target,
                                solver=solver)

# the Stein discrepancy for the uniform distribution (with uniform as target)
sum(vec(uniform_result.objectivevalue))
# the Stein discrepancy for the bets distribution (with uniform as target)
sum(vec(beta_result.objectivevalue))
# the test function g (and its gradient) that achieves the maximum discrepancy
uniform_result.g, uniform_result.gradg
