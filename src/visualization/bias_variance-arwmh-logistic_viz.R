# script for producing bias_variance-arwmh-logistic visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)
library(boot)

source('results_data_utils.R')

STEIN.BOUND <- "Spanner Stein discrepancy"
MEAN.ERROR <- "Mean error"
COV.ERROR <- "Covariance error"
UNCEN.COV.ERROR <- "Second moment error"
PROB.ERROR <- "Normalized prob. error"

ERROR_ORDER <- c(STEIN.BOUND, PROB.ERROR, MEAN.ERROR, UNCEN.COV.ERROR)

####################################
# Stein Diagnostic for each method #
####################################
dir <- "bias_variance-arwmh-logistic"
distname <- "logisticgaussianprior"
seed <- 37
d <- NULL
dataset <- "nodal"
batchsize <- 2
epsilon <- NULL
burninlikelihoods <- 1e3

program.dat <- concatDataList(
    dir=dir,
    distname=distname,
    dataset=dataset,
    burninlikelihoods=burninlikelihoods,
    epsilon=epsilon,
    batchsize=batchsize,
    d=d,
    seed=seed
)

#########################
# Comparison of Moments #
#########################
norm1 <- function(x) {
  sum(abs(x))
}
relative.error <- function(true, est) {
  norm1(true - est) / norm1(true)
}

Linfty.error <- function(true, est) {
  max(abs(true - est))
}

long.run.res <- concatDataList(
    dir="bias_variance-arwmh-logistic-MALA",
    distname=distname,
    dataset=dataset,
    d=NULL,
    seed=NULL
)[[1]]

example.df <- ldply(program.dat, function(res) {
    true.beta <- long.run.res$beta.hat
    true.vcov <- do.call(cbind, long.run.res$beta.vcov)
    true.uncentered.cov <- true.vcov + true.beta %*% t(true.beta)
    true.probs <- tail(long.run.res$scaled.pred.probs, 1)[[1]]

    betas <- do.call(cbind, res$X)
    beta.probs <- res$scaled.pred.probs
    stein.objectives <- res$objectivevalue

    n <- res$n
    beta.hat <- colMeans(betas)
    beta.vcov <- cov(betas)
    beta.uncentered.cov <- beta.vcov + beta.hat %*% t(beta.hat)

    mean.error <- Linfty.error(true.beta, beta.hat)
    vcov.error <- Linfty.error(true.vcov, beta.vcov)
    uncentered.cov.error <- Linfty.error(true.uncentered.cov, beta.uncentered.cov)
    pred.probs.error <- Linfty.error(true.probs, beta.probs)
    stein.error <- sum(stein.objectives)

    data.frame(
        n=n,
        numlikelihood=res$numlikelihood,
        batchsize=res$batchsize,
        seed=res$seed,
        measure=c(stein.error, pred.probs.error, mean.error, uncentered.cov.error),
        diagnostic=factor(ERROR_ORDER, levels=ERROR_ORDER),
        epsilon=res$epsilon
    )
})

example.df <- example.df[
  with(example.df, order(epsilon, numlikelihood)),
]
# Extract only the desired epsilon values
example.df <- subset(example.df, epsilon == 0.0 | epsilon == 0.1)
example.df$epsilon <- paste0("eps", example.df$epsilon)

filename <- sprintf(
    "../../results/%s/figs/julia_bias_variance-arwmh-logistic_dataset=%s_batchsize=%d_seed=%d.pdf",
    dir, dataset, example.df$batchsize[1], example.df$seed[1]
)
makeDirIfMissing(filename)

# Construct plot
final.plot <-
  ggplot(data=example.df, aes(x=numlikelihood, y=measure)) +
  geom_point(aes(color=epsilon, shape=epsilon), size=1) +
  geom_path(aes(color=epsilon, linetype=epsilon)) +
  facet_wrap(~ diagnostic, scales="free_y", nrow=1) +
  scale_x_log10(
      breaks=c(3*10^3, 10^4, 3*10^4, 10^5),
      limits=range(example.df$numlikelihood)
  ) +
  scale_color_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1))
      )
  ) +
  scale_shape_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1))
      )
  ) +
  scale_linetype_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1))
      )
  ) +
  labs(x="Number of likelihood evaluations", y="Discrepancy") +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=10),
        legend.title = element_text(size=9),
        axis.text = element_text(size=8),
        strip.text = element_text(size=8))
# Save to file
pdf(file=filename, width=8.5, height=1.5)
plot(final.plot)
dev.off()

