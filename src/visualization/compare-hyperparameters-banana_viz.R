# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(plyr)
library(gridExtra)
library(reshape2)
library(coda)
library(Hmisc)

source('results_data_utils.R')
q025 <- function(...) {quantile(..., probs=0.025)}
q975 <- function(...) {quantile(..., probs=0.975)}

############
# Bananas! #
############
dir <- "compare-hyperparameters-banana"
distname <- "banana-posterior"
n <- NULL
seed <- NULL
d <- NULL
epsilon <- NULL
batchsize <- NULL
subsamplescorefraction <- NULL
sampler <- 'approxslice'

program.dat <- concatDataList(
    dir=dir,
    distname=distname,
    n=n,
    sampler=sampler,
    epsilon=epsilon,
    batchsize=batchsize,
    subsamplescorefraction=subsamplescorefraction,
    d=d,
    seed=seed
)

no.subsample.dat <- Filter(function (res) {
    is.list(res$subsamplescorefraction)
}, program.dat)
subsample.dat <- Filter(function (res) {
    !is.list(res$subsamplescorefraction)
}, program.dat)

part1.df <- ldply(no.subsample.dat, function(res) {
    thetas <- do.call(cbind, res$thetas)
    ess <- sum(effectiveSize(thetas))
    data.frame(
        n=res$n,
        measure=c(res$discrepancy, ess),
        diagnostic=c('Kernel Stein, subsample none', 'ESS'),
        epsilon=res$epsilon,
        sampler=res$sampler
    )
})
part2.df <- ldply(subsample.dat, function(res) {
    fraction <- res$subsamplescorefraction
    diagnostic <- sprintf('Kernel Stein, subsample %s', fraction)
    data.frame(
        n=res$n,
        measure=res$discrepancy,
        diagnostic=diagnostic,
        epsilon=res$epsilon,
        sampler=res$sampler
    )
})
meas.df <- rbind(part1.df, part2.df)

meas.df <- meas.df[order(meas.df$epsilon), ]
meas.df$epsilon <- as.character(meas.df$epsilon)
eps.range <- sort(unique(meas.df$epsilon))
eps.labels <- sapply(eps.range[-1], function(eps.value) {
    sci.value <- sprintf("%.0e", as.numeric(eps.value))
    bquote(.(sci.value))
})
eps.labels <- c("0", eps.labels)

ess.vs.kernelstein.plot <-
    ggplot(data=meas.df, aes(x=epsilon, y=log(measure))) +
    stat_summary(aes(color=diagnostic), fun.ymin=q025, fun.ymax=q975,
                 geom="errorbar", width=0.25) +
    stat_summary(aes(color=diagnostic, shape=diagnostic), geom="point",
                 fun.y="mean") +
    stat_summary(aes(color=diagnostic, shape=diagnostic, group=diagnostic),
                 geom="path", fun.y="mean") +
    facet_wrap(~ diagnostic, scales="free_y", ncol=1) +
    labs(x=expression(epsilon), y="Log diagnostic") +
    scale_x_discrete(
        breaks=eps.range,
        labels=eps.labels
    ) +
    scale_y_continuous(
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          legend.position = "none",
          axis.text = element_text(size=6),
          legend.text = element_text(size=9),
          strip.text = element_text(size=8),
          strip.background = element_rect(fill="white",
          color="black", size=1)
    )

# now setup scatter plots
program.round <- Filter(function(res) {
    res$seed == 1 && res$subsamplescorefraction == "0.5"
}, program.dat)
theta.samples.df <- ldply(program.round, function (res) {
    thetas <- do.call(cbind, res$thetas)
    data.frame(
        theta1=thetas[,1],
        theta2=thetas[,2],
        epsilon=res$epsilon
    )
})

banana.y <- concatDataList(
    dir="compare-hyperparameters-banana-y",
    distname="banana-y"
)[[1]][['y']]

SIGMA2Y <- 4.0
SIGMA2THETA <- 1.0
banana.unnormalizedlogpdf <- function(theta) {
    theta0 <- theta[1] + theta[2]^2
    log.prior.lik <- dnorm(theta[1], sd=sqrt(SIGMA2THETA), log=T) +
        dnorm(theta[2], sd=sqrt(SIGMA2THETA), log=T)
    log.likelihood <- dnorm(banana.y, mean=theta0, sd=sqrt(SIGMA2Y), log=T)
    log.prior.lik + sum(log.likelihood)
}
unnormalizedlogpdf.df <- get_contour2d_df(
    banana.unnormalizedlogpdf,
    xlim=c(-3, 2),
    ylim=c(-2.5, 2.5),
    nx=200,
    ny=200
)
contour.breaks <- c(-214.5, -215, -216.5)

facet_labeller <- function(value.df) {
    lapply(value.df, function(values) {
        sapply(values, function (value) {
            if (value == 0) {
                sci.value <- "0"
            } else {
                sci.value <- sprintf("%0.e", value)
            }
            n <- Filter(function (res) {res$epsilon == value}, program.round)[[1]]$n
            bquote(paste(epsilon, " = ", .(sci.value), " (", n, " = ", .(n), ")"))
        })
  })
}

approxslice.plots <- ggplot(data=theta.samples.df, aes(x=theta1, y=theta2)) +
    geom_point(size=0.7, color="black", alpha=0.1) +
    stat_contour(
        aes(x=x1, y=x2, z=logdens, color=..level..),
        size=0.4,
        breaks=contour.breaks,
        data=unnormalizedlogpdf.df) +
    facet_grid(. ~ epsilon,
               labeller=facet_labeller) +
    scale_color_gradient(low="green",
                         high="red",
                         guide=FALSE) +
    labs(x=expression(theta[1]), y=expression(theta[2])) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          axis.text = element_text(size=8),
          legend.text = element_text(size=9),
          strip.text = element_text(size=8),
          strip.background = element_rect(fill="white",
          color="black", size=1)
    )

filename <- sprintf(
    "../../results/%s/figs/julia_%s_sampler=%s_maxn=%d_seed=%s.pdf",
    dir, dir, sampler, max(meas.df$n), seed
)
makeDirIfMissing(filename)

pdf(file=filename, width=10, height=3)
grid.arrange(ess.vs.kernelstein.plot, approxslice.plots, nrow=1,
  widths = unit.c(unit(0.2, "npc"), unit(0.78, "npc")))
dev.off()
