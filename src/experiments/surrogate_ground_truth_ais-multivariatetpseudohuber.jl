# surrogate_ground_truth_ais-multivariatetpseudohuber
#
# This script runs a long chain based on the
# Metropolis-adjusted Riemannian Langevin algorithm (MARLA). This is used as
# a gold standard as this procedure yields an asymptotically unbiased chain
# and thus is used to compare the efficiency of the biased chains produced
# in the experiment compare-hyperparameters-multivariatetpseudohuber.

using DataFrames

using SteinDistributions: SteinMultivariateStudentTRegressionPseudoHuberPrior
using SteinSamplers: runmarla

include("experiment_utils.jl")
include("ais/ais_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7
# set the delta param for the Huber prior
@parsestringcli clinu "n" "nu" "10.0"
nu = float(clinu)
# set the nu param degrees of freedom for the d-distribution
@parsestringcli clidelta "d" "delta" "1.0"
delta = float(clidelta)

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
# number of samples to make
n = 5000000
# the thinning rate
thinby = 20
# create the target distribution
sigma = Diagonal(ones(size(X,1)))
target = SteinMultivariateStudentTRegressionPseudoHuberPrior(X, y, nu, sigma, zeros(d), delta)
# starting beta0
beta0 = getapproxmode(target)
# the epsilons to run the experiemnt
epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
# define the covariance terms for the Riemannian-Langevin diffusion
function volatility_covariance(x::Array{Float64,1})
    p = length(x)
    sqrt(1 + sum(x.^2) / delta^2) * eye(p)
end
function grad_volatility_covariance(x::Array{Float64,1})
    delta^(-2) * x ./ sqrt(1 + sum(x.^2) / delta^2)
end

for epsilon = epsilons
    # Draw MARLA samples
    (betas_marla, numgrad_marla, acceptance_ratio) =
        @setseed runmarla(target,
                          beta0,
                          volatility_covariance,
                          grad_volatility_covariance;
                          epsilonfunc=t -> epsilon,
                          verbose=true,
                          numiter=n*thinby)
    # Thin the samples
    betas_marla_thinned = betas_marla[thinby:thinby:(thinby*n),:]

    # Package and save results
    summary_data = Dict{Any, Any}(
        "n" => n,
        "seed" => seed,
        "distname" => distname,
        "epsilon" => epsilon,
        "nu" => nu,
        "delta" => delta,
        "betas" => betas_marla_thinned,
        "numgrad" => numgrad_marla,
        "acceptance_ratio" => acceptance_ratio,
        "sampler" => "ais-marla"
    )
    save_json(
        summary_data;
        dir="compare-hyperparameters-multivariatetpseudohuber-MARLA",
        distname=distname,
        dataset=data_set,
        n=n,
        thinby=thinby,
        epsilon=epsilon,
        nu=nu,
        delta=delta,
        d=d,
        seed=seed
    )
end
