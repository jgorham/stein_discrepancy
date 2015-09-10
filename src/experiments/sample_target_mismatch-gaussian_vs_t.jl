# sample_target_mismatch-gaussian_vs_t
#
# Compares the Stein discrepancy of samples drawn i.i.d. from a standard
# Gaussian target to the discrepancy of samples drawn i.i.d. from a scaled
# Student's t distribution with matching mean and variance

using SteinDistributions: SteinGaussian, SteinScaleLocationStudentT
using SteinDiscrepancy: stein_discrepancy

include("experiment_utils.jl")

# random number generator seed
# we allow this to be set on the commandline
@parseintcli seed "s" "seed" 7
# Specify dimension of sampled vectors
d = 1
# Specify a maximum number of samples
n = 200000
# select sample distribution
gaussian_target = SteinGaussian(d)
# select target distribution
df = 10.0
t_target = SteinScaleLocationStudentT(df, 0.0, sqrt((df-2.0)/df))
# Select a solver for Stein discrepancy optimization
#solver = "clp"
solver = "gurobi"
# Draw samples from target distribution
gaussianX = @setseed rand(gaussian_target, n)
# Draw samples from target distribution
tX = @setseed rand(t_target, n)
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:min(1000,n), 2000:1000:min(10000,n), 20000:10000:n)

# Solve optimization problem at each sample size
for i = ns
    @printf("[Beginning n=%d]\n", i)
    # Compute Stein discrepancy for first i points in each sample
    @printf("[Beginning solver for gaussian n=%d]\n", i)
    gaussian_result = stein_discrepancy(points=gaussianX[1:i,:],
                                        target=gaussian_target,
                                        solver=solver)

    @printf("[Beginning solver for t-dist n=%d]\n", i)
    t_result = stein_discrepancy(points=tX[:,1:i]',
                                 target=gaussian_target,
                                 solver=solver)

    # save data
    instance_data = Dict({
        "target" => "gaussian",
        "df" => df,
        "gaussian_X" => gaussian_result.points,
        "gaussian_g" => gaussian_result.g,
        "gaussian_gprime" => gaussian_result.gradg,
        "gaussian_objectivefunc" =>  gaussian_result.operatorg,
        "studentt_X" => t_result.points,
        "studentt_g" => t_result.g,
        "studentt_gprime" => t_result.gradg,
        "studentt_objectivefunc" => t_result.operatorg,
        "n" => i,
        "d" => d,
        "seed" => seed,
        "studentt_objectivevalue" => t_result.objectivevalue,
        "gaussian_objectivevalue" => gaussian_result.objectivevalue,
    })
    save_json(
        instance_data;
        dir="sample_target_mismatch-gaussian_vs_t",
        target="gaussian",
        n=i,
        d=d,
        seed=seed
    )
end

@printf("COMPLETE!")
