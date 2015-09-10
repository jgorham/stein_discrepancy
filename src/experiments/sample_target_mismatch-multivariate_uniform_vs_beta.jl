# incorrect_target_divergence_multivariate
#
# Script for testing the divergence of Stein diagnostic bounds
# when the samples aren't drawn from the target distribution.
#

using Distributions: Beta
using SteinDistributions: SteinUniform
using SteinDiscrepancy: stein_discrepancy

include("experiment_utils.jl")

# random number generator seed
# we allow this to be set on the commandline
@parseintcli seed "s" "seed" 7
# Specify dimension of sampled vectors
d = 2
# Specify a maximum number of samples
n = 200000
# select sample distribution
beta_target = Beta(3, 3)
# select target distribution
uniform_target = SteinUniform(d)
# Select a solver for Stein discrepancy optimization
#solver = "clp"
solver = "gurobi"
# Draw samples from target distribution
uniformX = @setseed rand(uniform_target, n)
# Draw samples from target distribution
betaX = @setseed reshape(rand(beta_target, d*n), n, d)
# Associate sample weights
q = ones(n, 1) ./ n
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:min(1000,n), 2000:1000:min(10000,n), 20000:10000:n)

# Solve optimization problem at each sample size
for i = ns
    @printf("[Beginning n=%d,d=%d]\n", i, d)
    # Create program with data subset
    @printf("[Beginning solver for uniform n=%d]\n", i)
    uniform_result = stein_discrepancy(points=uniformX[1:i,:],
                                       target=uniform_target,
                                       solver=solver)

    @printf("[Beginning solver for beta n=%d]\n", i)
    beta_result = stein_discrepancy(points=betaX[1:i,:],
                                    target=uniform_target,
                                    solver=solver)
    # save data
    instance_data = Dict({
        "target" => "uniform",
        "uniform_X" => uniform_result.points,
        "uniform_g" => uniform_result.g,
        "uniform_gradg" => uniform_result.gradg,
        "uniform_objectivevalue" =>  vec(uniform_result.objectivevalue),
        "beta_X" => beta_result.points,
        "beta_g" => beta_result.g,
        "beta_gradg" => beta_result.gradg,
        "beta_objectivevalue" => vec(beta_result.objectivevalue),
        "n" => i,
        "d" => d,
        "seed" => seed,
    })
    save_json(
        instance_data;
        dir="sample_target_mismatch-multivariate_uniform_vs_beta",
        target="uniform",
        n=i,
        d=d,
        seed=seed
    )
end

@printf("COMPLETE!")
