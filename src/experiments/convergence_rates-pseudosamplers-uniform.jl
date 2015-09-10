# Pseudosampling experiment
#
# Compares the Stein discrepancy convergence behavior of herding and
# quasi Monte Carlo pseudosampling with that of independent random sampling.
#
# Before running this script, run src/experiments/pseudosample/pseudosample.m from the
# repository base directory to generate pseudosamples.

using MAT
using SteinDistributions: SteinUniform
using SteinDiscrepancy: stein_discrepancy

include("experiment_utils.jl")

# Load generated pseudosamples from experiment results directory
experdir = joinpath("src", "experiments", "pseudosample", "results", "pseudosample")
samplefile = joinpath(experdir,"samples.mat")
samples = matread(samplefile)

# Select an optimization problem solver
solver = "clp"
#solver = "gurobi"
# Specify number of samples and dimension of sampled vectors
n = length(samples["samples_herding"])
# dimension of x
d = 1
# distribution sampled from
distname = "uniform"
# Independent Uniform([0.0,1.0]) with best known Stein constants
(c1,c2,c3) = (0.5,0.5,1.0)
target = SteinUniform(d, c1, c2, c3)
# Sample sizes at which optimization problem will be solved
ns = 1:200
# Solve optimization problem for each sampler at each sample size
samplers = ["herding","sobol","independent"]

# Collect all objective values for interactive plotting
for sampler in samplers, i = ns
    @printf("Beginning optimization for %s, n=%d, d=%d\n", sampler, i, d)
    # Create program with data subset
    # each column represents a separate trial
    simulateddata = samples["samples_$sampler"]
    numtrials = size(simulateddata, 2)

    for trial in 1:numtrials
        # Compute Stein discrepancy for first i points in this trial
        res = stein_discrepancy(points=simulateddata[1:i,trial]'', target=target,
                                solver=solver)
        println("\tn = $(i), objective = $(res.objectivevalue)")

        # Package and save results
        instance_data = Dict({
            "sampler" => sampler,
            "distname" => distname,
            "n" => i,
            "trial" => trial,
            "d" => d,
            "seed" => int(samples["seed"]),
            "X" => res.points,
            "q" => res.weights,
            "g" => res.g,
            "gprime" => res.gradg,
            "objectivevalue" => res.objectivevalue,
            "edgetime" => res.edgetime,
            "solvetime" => res.solvetime
        })

        save_json(
            instance_data;
            dir="convergence_rates-pseudosamplers-uniform",
            distname=distname,
            sampler=sampler,
            n=i,
            trial=trial,
            d=d,
            seed=int(samples["seed"])
        )
    end
end

@printf("COMPLETE!")
