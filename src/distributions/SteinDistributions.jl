# Representations of distributions suitable for Stein discrepancy calculations
module SteinDistributions

export
# Abstract types
SteinDistribution,
SteinPosterior,
# Specific distributions
SteinDiscrete,
SteinGMM,
SteinGMMPosterior,
SteinGaussian,
SteinLogisticRegressionPosterior,
SteinLogisticRegressionGaussianPrior,
SteinMultivariateStudentTRegressionPosterior,
SteinMultivariateStudentTRegressionGaussianPrior,
SteinMultivariateStudentTRegressionPseudoHuberPrior,
SteinHuberLossRegressionPosterior,
SteinHuberLossRegressionGaussianPrior,
SteinProbitRegression,
SteinScaleLocationStudentT,
SteinUniform,
SteinBanana,
SteinHierarchericalPoisson,
SteinSquaredFourierSumDensity,
# Common functions operating on distributions
# CDF of a distribution
cdf,
# Get Stein factors
getC1,
getC2,
getC3,
# Lower bound on range of coordinate
supportlowerbound,
# Upper bound on range of coordinate
supportupperbound,
# Number of dimensions of target variable
numdimensions,
# get number of samples when distribution is posterior of data
numdatapoints,
# the log prior for posterior distributions
logprior,
# the likelihood [without a prior for posterior distributions]
loglikelihood,
# Log density [will include prior for posterior distributions]
logdensity,
# Gradient of the prior
gradlogprior,
# Gradient of the log density
gradlogdensity,
# Gradient of the log likelihood
gradloglikelihood,
# Random samples drawn from distribution
rand,
# random GMM samples
randgmm,
# random SteinBanana samples
randbanana,
# random procedure for SteinHierarchericalPoisson
rungibbs

# Include abstract types first
include("SteinDistribution.jl")
include("SteinPosterior.jl")
# Include specific distributions
include("SteinDiscrete.jl")
include("SteinGMM.jl")
include("SteinGMMPosterior.jl")
include("SteinGaussian.jl")
include("SteinLogisticRegressionPosterior.jl")
include("SteinLogisticRegressionGaussianPrior.jl")
include("SteinMultivariateStudentTRegressionPosterior.jl")
include("SteinMultivariateStudentTRegressionGaussianPrior.jl")
include("SteinMultivariateStudentTRegressionPseudoHuberPrior.jl")
include("SteinHuberLossRegressionPosterior.jl")
include("SteinHuberLossRegressionGaussianPrior.jl")
include("SteinProbitRegression.jl")
include("SteinScaleLocationStudentT.jl")
include("SteinUniform.jl")
include("SteinBanana.jl")
include("SteinHierarchericalPoisson.jl")
include("SteinSquaredFourierSumDensity.jl")
end
