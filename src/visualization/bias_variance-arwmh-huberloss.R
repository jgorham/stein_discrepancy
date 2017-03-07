# script for producing bias_variance-arwmh-huberloss visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)
library(boot)

source('results_data_utils.R')

STEIN.BOUND <- "Langevin Stein discrepancy"
MEAN.ERROR <- "Mean error"
COV.ERROR <- "Covariance error"
UNCEN.COV.ERROR <- "Second moment error"
YHAT.ERROR <- "Normalized predictive error"

ERROR_ORDER <- c(STEIN.BOUND, YHAT.ERROR, MEAN.ERROR, UNCEN.COV.ERROR)

####################################
# Stein Diagnostic for each method #
####################################
dir <- "bias_variance-arwmh-huberloss"
distname <- "huberloss-gaussianprior"
seed <- 7
d <- NULL
dataset <- "radon"
batchsize <- 5
epsilon <- NULL
burninlikelihoods <- 100000
n <- NULL

program.dat <- concatDataList(
    dir=dir,
    distname=distname,
    dataset=dataset,
    n=n,
    burninlikelihoods=sprintf("%d",burninlikelihoods),
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
    dir="bias_variance-arwmh-huberloss-MALA",
    distname=distname,
    dataset=dataset,
    d=NULL,
    seed=NULL
)[[1]]

example.df <- ldply(program.dat, function(res) {
    true.beta <- long.run.res$beta.hat
    true.vcov <- do.call(cbind, long.run.res$beta.vcov)
    true.uncentered.cov <- true.vcov + true.beta %*% t(true.beta)
    true.yhats <- tail(long.run.res$scaled.yhat, 1)[[1]]

    betas <- do.call(cbind, res$X)
    beta.weights <- res$q[[1]]
    beta.yhats <- res$scaled.yhat
    stein.objectives <- res$objectivevalue

    n <- res$n
    beta.hat <- colSums(sweep(betas, 1, beta.weights, '*'))
    beta.vcov <- cov.wt(betas, wt = beta.weights)$cov
    beta.uncentered.cov <- cov.wt(betas, wt = beta.weights, center = F)$cov

    mean.error <- Linfty.error(true.beta, beta.hat)
    vcov.error <- Linfty.error(true.vcov, beta.vcov)
    uncentered.cov.error <- Linfty.error(true.uncentered.cov, beta.uncentered.cov)
    yhat.error <- Linfty.error(true.yhats, beta.yhats)
    stein.error <- sum(stein.objectives)

    data.frame(
        n=n,
        numlikelihood=res$numlikelihood,
        batchsize=res$batchsize,
        seed=res$seed,
        measure=c(stein.error, yhat.error, mean.error, uncentered.cov.error),
        diagnostic=factor(ERROR_ORDER, levels=ERROR_ORDER),
        epsilon=res$epsilon,
        solvetime=mean(res$solvetime),
        edgetime=res$edgetime
    )
})

example.df <- example.df[
  with(example.df, order(epsilon, numlikelihood)),
]
# Extract only the desired epsilon values
#example.df <- subset(example.df, epsilon == 0.0 | epsilon == 0.1)
example.df$epsilon <- paste0("eps", example.df$epsilon)

range.df <- ddply(
    example.df,
    .(batchsize, epsilon),
    function (df) {
        c(min=min(df$numlikelihood), max=max(df$numlikelihood))
    })

filename <- sprintf(
    "../../results/%s/figs/julia_bias_variance-arwmh-huberloss_dataset=%s_batchsize=%d_seed=%d.pdf",
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
      limits=c(max(range.df$min), min(range.df$max)),
      breaks=c(3*10^4, 10^5, 3*10^5, 10^6, 3*10^6, 10^7)
  ) +
  scale_color_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1)),
          "eps0.2"=bquote(paste(epsilon, " = ", 0.2))
      )
  ) +
  scale_shape_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1)),
          "eps0.2"=bquote(paste(epsilon, " = ", 0.2))
      )
  ) +
  scale_linetype_discrete(
      guide = guide_legend(title = "Hyperparameter"),
      labels = c(
          "eps0"=bquote(paste(epsilon, " = ", 0)),
          "eps0.1"=bquote(paste(epsilon, " = ", 0.1)),
          "eps0.2"=bquote(paste(epsilon, " = ", 0.2))
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
pdf(file=filename, width=8.5, height=2.5)
plot(final.plot)
dev.off()

