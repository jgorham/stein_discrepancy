# script for producing mnist_7_or_8_sgfs_sgld visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)
library(rhdf5)
library(ellipse)
library(gridExtra)

source('results_data_utils.R')

STEIN.BOUND <- "Stein kernel discrepancy"
MEAN.ERROR <- "Mean error"
COV.ERROR <- "Covariance error"

ERROR_ORDER <- c(STEIN.BOUND, MEAN.ERROR, COV.ERROR)

####################
# Helper functions #
####################
tr <- function(...) sum(diag(...))

optimalTwoMarginals <- function(
    beta1, sigma1,
    beta2, sigma2,
    worst=TRUE
) {
    p <- length(beta1)
    ijs <- expand.grid(i=1:p,j=1:p)
    ijs <- subset(ijs, i < j)
    kl.divs <- apply(ijs, 1, function(row) {
        b1 <- beta1[row]
        b2 <- beta2[row]
        s1 <- sigma1[row,row]
        s2 <- sigma2[row,row]
        t <- t(b2 - b1) %*% solve(s2) %*% (b2 - b1)
        t <- t + log(det(s2) / det(s1))
        t <- t + tr(solve(s2) %*% s1)
        t <- t - 2
        0.5 * as.numeric(t)
    })
    opt.pair.idx <- which.max((2 * worst - 1) * kl.divs)
    opt.pair <- ijs[opt.pair.idx,]
    as.numeric(opt.pair)
}

#########
# Data! #
#########
dir <- "mnist_7_or_9_sgfs"
distname <- "logisticgaussianprior"
n <- NULL
dataset <- "mnist"
kernel <- "inversemultiquadric"

sgfsd.dat <- concatDataList(
    dir=dir,
    distname=distname,
    dataset=dataset,
    kernel=kernel,
    sampler="SGFS-d"
)[[1]]

sgfsf.dat <- concatDataList(
    dir=dir,
    distname=distname,
    dataset=dataset,
    kernel=kernel,
    sampler="SGFS-f"
)[[1]]

####################################
# Stein Diagnostic for each method #
####################################
hmc.betas <- h5read('../experiments/mnist/hmc_sample_points.h5', 'hmc')
sgfsd.betas <- do.call(cbind, sgfsd.dat$betas)
sgfsf.betas <- do.call(cbind, sgfsf.dat$betas)

true.beta <- colMeans(hmc.betas)
true.vcov <- cov(hmc.betas)

sgfsd.betahat <- colMeans(sgfsd.betas)
sgfsd.vcov <- cov(sgfsd.betas)

sgfsf.betahat <- colMeans(sgfsf.betas)
sgfsf.vcov <- cov(sgfsf.betas)

worst.sgfsd.ij <- optimalTwoMarginals(
    true.beta, true.vcov,
    sgfsd.betahat, sgfsd.vcov,
    worst=TRUE
)
best.sgfsd.ij <- optimalTwoMarginals(
    true.beta, true.vcov,
    sgfsd.betahat, sgfsd.vcov,
    worst=FALSE
)
worst.sgfsf.ij <- optimalTwoMarginals(
    true.beta, true.vcov,
    sgfsf.betahat, sgfsf.vcov,
    worst=TRUE
)
best.sgfsf.ij <- optimalTwoMarginals(
    true.beta, true.vcov,
    sgfsf.betahat, sgfsf.vcov,
    worst=FALSE
)

makeContourPlot <- function(
    raw.betas,
    sgfs.means,
    sgfs.vcov,
    true.means,
    true.vcov,
    x.idx,
    is.diagonal,
    is.worst
) {
    minx <- min(true.vcov$x)
    miny <- min(true.vcov$y)
    xdelta <- diff(range(true.vcov$x))
    ydelta <- diff(range(true.vcov$y))

    raw.betas <- cbind(
        raw.betas,
        sampler=ifelse(is.diagonal, 'SGFS-d', 'SGFS-f'),
        marginal=ifelse(is.worst, 'WORST', 'BEST')
    )

    ggplot() +
        geom_bin2d(data=raw.betas, aes(x=X1, y=X2), bins=60) +
        geom_point(data=sgfs.means, aes(x=X1, y=X2), color="blue") +
        geom_point(data=true.means, aes(x=X1, y=X2), color="red") +
        geom_path(data=sgfs.vcov, aes(x=x, y=y), color="blue", linetype=2) +
        geom_path(data=true.vcov, aes(x=x, y=y), color="red") +
        facet_grid(marginal ~ sampler) +
        xlim(c(minx - 0.1 * xdelta, minx + 1.1 * xdelta)) +
        ylim(c(miny - 0.1 * ydelta, miny + 1.1 * ydelta)) +
        scale_fill_gradient2(low="white", high="black") +
        labs(x=bquote(x[.(x.idx[1])]),
             y=bquote(x[.(x.idx[2])])) +
        theme_bw() +
        theme(
            legend.position = "none",
            axis.text = element_text(size=8),
            plot.margin = unit(c(0,0,0,0), "npc"),
            strip.background = element_rect(fill="white",
            color="black", size=1)
        )
}

best.sgfsf.plot <- makeContourPlot(
    data.frame(
        sgfsf.betas[,best.sgfsf.ij]
    ),
    data.frame(
        t(sgfsf.betahat[best.sgfsf.ij])
    ),
    data.frame(
        ellipse(sgfsf.vcov[best.sgfsf.ij,best.sgfsf.ij], centre=sgfsf.betahat[best.sgfsf.ij])
    ),
    data.frame(
        t(true.beta[best.sgfsf.ij])
    ),
    data.frame(
        ellipse(true.vcov[best.sgfsf.ij,best.sgfsf.ij],
                centre=true.beta[best.sgfsf.ij])
    ),
    best.sgfsf.ij,
    FALSE,
    FALSE
)

worst.sgfsf.plot <- makeContourPlot(
    data.frame(
        sgfsf.betas[,worst.sgfsf.ij]
    ),
    data.frame(
        t(sgfsf.betahat[worst.sgfsf.ij])
    ),
    data.frame(
        ellipse(sgfsf.vcov[worst.sgfsf.ij,worst.sgfsf.ij], centre=sgfsf.betahat[worst.sgfsf.ij])
    ),
    data.frame(
        t(true.beta[worst.sgfsf.ij])
    ),
    data.frame(
        ellipse(true.vcov[worst.sgfsf.ij,worst.sgfsf.ij],
                centre=true.beta[worst.sgfsf.ij])
    ),
    worst.sgfsf.ij,
    FALSE,
    TRUE
)

best.sgfsd.plot <- makeContourPlot(
    data.frame(
        sgfsd.betas[,best.sgfsd.ij]
    ),
    data.frame(
        t(sgfsd.betahat[best.sgfsd.ij])
    ),
    data.frame(
        ellipse(sgfsd.vcov[best.sgfsd.ij,best.sgfsd.ij], centre=sgfsd.betahat[best.sgfsd.ij])
    ),
    data.frame(
        t(true.beta[best.sgfsd.ij])
    ),
    data.frame(
        ellipse(true.vcov[best.sgfsd.ij,best.sgfsd.ij],
                centre=true.beta[best.sgfsd.ij])
    ),
    best.sgfsd.ij,
    TRUE,
    FALSE
)

worst.sgfsd.plot <- makeContourPlot(
    data.frame(
        sgfsd.betas[,worst.sgfsd.ij]
    ),
    data.frame(
        t(sgfsd.betahat[worst.sgfsd.ij])
    ),
    data.frame(
        ellipse(sgfsd.vcov[worst.sgfsd.ij,worst.sgfsd.ij], centre=sgfsd.betahat[worst.sgfsd.ij])
    ),
    data.frame(
        t(true.beta[worst.sgfsd.ij])
    ),
    data.frame(
        ellipse(true.vcov[worst.sgfsd.ij,worst.sgfsd.ij],
                centre=true.beta[worst.sgfsd.ij])
    ),
    worst.sgfsd.ij,
    TRUE,
    TRUE
)

marginal.plots <- arrangeGrob(
    best.sgfsd.plot,
    worst.sgfsd.plot,
    best.sgfsf.plot,
    worst.sgfsf.plot,
    ncol=2)


kernel.df <- ldply(list(sgfsd.dat, sgfsf.dat), function(res) {
    sampler <- res$sampler
    data.frame(
        n=res$n,
        discrepancy=res$discrepancy,
        sampler=sampler
    )
})

x.pow <- seq(2, 5, by=0.5)
x.breaks <- 10^x.pow
x.labels <- sapply(x.pow, function (xp) {
    bquote(10^.(xp))
})

kernel.plot <-
  ggplot(data=kernel.df, aes(x=n, y=discrepancy)) +
  geom_point(aes(color=sampler, shape=sampler), size=1) +
  geom_path(aes(color=sampler, linetype=sampler)) +
  scale_x_log10(
      breaks=x.breaks,
      labels=x.labels
  ) +
  labs(x="Number of sample points, n", y="IMQ kernel Stein discrepancy") +
  guides(
    color = guide_legend(title="Sampler"),
    shape = guide_legend(title="Sampler"),
    linetype = guide_legend(title="Sampler")
  ) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=9),
        axis.title = element_text(size=10),
        legend.title = element_text(size=10),
        axis.text = element_text(size=8),
        strip.text = element_text(size=8))


filename <- sprintf(
    "../../results/%s/figs/julia_%s_dataset=%s.pdf",
    dir, dir, dataset
)
makeDirIfMissing(filename)

# Save to file
pdf(file=filename, width=8.5, height=2.7)
grid.arrange(
  kernel.plot, marginal.plots,
  widths = unit.c(unit(0.4, "npc"), unit(0.55, "npc")),
  nrow=1)
dev.off()

