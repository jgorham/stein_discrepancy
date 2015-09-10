# script for producing bias_variance-arwmh-logistic visualizations
library(plyr)

source('results_data_utils.R')

compute_avg_runtimes <- function(dat) {
    times.df <- ldply(dat, function(res) {
        solvetimes <- res$solvetime
        names(solvetimes) <- paste0('d', 1:length(solvetimes))
        spanningtime <- as.numeric(res$edgetime)
        c(solvetimes, spanningtime=spanningtime)
    })
    colMeans(times.df)
}

####################################
# bias_variance-arwh-logistic-MALA #
####################################
dir <- "bias_variance-arwmh-logistic"
distname <- "logisticgaussianprior"
seed <- NULL
d <- NULL
dataset <- "nodal"
epsilon <- NULL
burninlikelihoods <- 1e3
n <- 1000

bias.dat <- concatDataList(
    dir=dir,
    distname=distname,
    dataset=dataset,
    n=n,
    burninlikelihoods=burninlikelihoods,
    epsilon=epsilon,
    d=d,
    seed=seed
)
compute_avg_runtimes(bias.dat)

########
# sgld #
########
dir <- "compare_hyperparameters-sgld-gmm"
d <- 2
n <- 1000
seed <- NULL
solver <- "gurobi"
distname <- "sgld-gmm"

sgld.dat <- concatDataList(
    dir=dir,
    distname=distname,
    n=n,
    d=d,
    seed=seed
)
compute_avg_runtimes(sgld.dat)

####################################
# convergence_rates-pseudosamplers #
####################################
dir <- "convergence_rates-pseudosamplers-uniform"
sampler <- NULL
distname <- "uniform"
n <- 200
d <- 1
trial <- NULL
seed <- 7

conv.dat <- concatDataList(
    dir="convergence_rates-pseudosamplers-uniform",
    distname=distname,
    sampler=sampler,
    n=n,
    trial=trial,
    d=d,
    seed=seed
)
compute_avg_runtimes(conv.dat)
