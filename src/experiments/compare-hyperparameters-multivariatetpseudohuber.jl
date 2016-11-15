# compare-hyperparameters-multivariatetpseudohuber
#
# This script compares the quality of samples drawn
# from a Bayesian regession for the AIS dataset.

using DataFrames

using SteinDiscrepancy: stein_discrepancy
using SteinDistributions: SteinMultivariateStudentTRegressionPseudoHuberPrior
using SteinSamplers: runsgrld

include("experiment_utils.jl")
include("ais/ais_utils.jl")

# random number generator seed
# this controls the seed used for the subsampling
@parseintcli seed "s" "seed" 7
# set the sampler
@parsestringcli sampler "q" "sampler" "runsgrld"
# set the delta param for the Huber prior
@parsestringcli clinu "n" "nu" "10.0"
nu = float(clinu)
# set the nu param degrees of freedom for the d-distribution
@parsestringcli clidelta "d" "delta" "0.1"
delta = float(clidelta)
# Select a solver for Stein discrepancy optimization problem
#eval(Expr(:import,:Clp))
#solver = Main.Clp.ClpSolver(SolveType=5,LogLevel=4)
solver = "gurobi"
# distname is multivariatetpseudohuber
distname = "multivariatetpseudohuber"
###            Pull in data here             ###
### ais dataset is available in sn R package ###
data_set = "ais"
yfeature = "LBM"
dat = readtable("src/experiments/ais/ais.tsv")
X = convert(Array{Float64,2}, dat[[:RCC, :WCC, :Fe]])
# standardize columns of X
X = X .- mean(X,1)
X = [ones(size(X,1)) X]
(N, d) = size(X)
column_norms = zeros(1, d)
for ii in 1:d
    column_norms[1,ii] = norm(X[:,ii])
end
X = X ./ column_norms
y = convert(Array{Float64,1}, dat[:LBM])
y = (y - mean(y)) / std(y)
# create posterior distribution
sigma = Diagonal(ones(size(X,1)))
target = SteinMultivariateStudentTRegressionPseudoHuberPrior(X, y, nu, sigma, zeros(d), delta)
# starting beta0
beta0 = getapproxmode(target)
# batchsize for SGRLD algorithm
batchsize = 30
# the SGRLD epsilons to run the experiemnt
sgrld_epsilons = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# define the covariance terms for the Riemannian-Langevin diffusion
function volatility_covariance(x::Array{Float64,1})
    p = length(x)
    sqrt(1 + sum(x.^2) / delta^2) * eye(p)
end
function grad_volatility_covariance(x::Array{Float64,1})
    delta^(-2) * x / sqrt(1 + sum(x.^2) / delta^2)
end

# Define sampler
n = 2000
thinby = 100
numsweeps = round(Int, ceil(n * thinby / floor(N/batchsize)))
epsilons = sgrld_epsilons
runsampler(x0,epsilon) = runsgrld(target,
                                  x0,
                                  volatility_covariance,
                                  grad_volatility_covariance;
                                  epsilonfunc=t -> epsilon,
                                  numsweeps=numsweeps,
                                  batchsize=batchsize)

for epsilon in epsilons
    println("Generating samples for sampler=$(sampler),epsilon=$(epsilon)")

    betas = []; numeval = NaN
    @trycatchcontinue(
        begin
            (betas, numeval) = @setseed seed runsampler(beta0,epsilon);
            if any(isnan(betas))
                error("[$(distname):seed=$(seed),epsilon=$(epsilon)] NaNs found. Skipping.");
            end;
            betas = betas[thinby:thinby:(thinby*n),:]
        end,
        println("[$(distname):seed=$(seed),epsilon=$(epsilon)]:")
    )

    # Find stein discrepancy
    @printf("[Computing the stein discrepancy]\n")
    result = @setseed seed stein_discrepancy(points=betas,
                                             solver=solver,
                                             target=target,
                                             operator="riemannian-langevin",
                                             volatility_covariance=volatility_covariance,
                                             grad_volatility_covariance=grad_volatility_covariance)

    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "n" => size(betas,1),
        "d" => d,
        "seed" => seed,
        "thinby" => thinby,
        "epsilon" => epsilon,
        "betas" => betas,
        "objectivevalue" => vec(result.objectivevalue),
        "numeval" => numeval,
        "sampler" => sampler,
        "nu" => nu,
        "delta" => delta,
        "batchsize" => batchsize,
        "nprocs" => nprocs(),
        "edgetime" => result.edgetime,
        "solvertime" => result.solvetime
    )

    save_json(
        instance_data;
        dir="compare-hyperparameters-multivariatetpseudohuber",
        distname=distname,
        n=size(betas,1),
        thinby=thinby,
        sampler=sampler,
        epsilon=epsilon,
        nu=nu,
        delta=delta,
        batchsize=batchsize,
        d=d,
        seed=seed,
    )
end
