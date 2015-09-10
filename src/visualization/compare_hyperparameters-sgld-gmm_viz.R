# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)
library(coda)

source('results_data_utils.R')

### ess plots
dir <- "compare_hyperparameters-sgld-gmm"
d <- 2
seed <- NULL
solver <- "gurobi"
distname <- "sgld-gmm"

sgld.dat <- concatDataList(
    dir=dir,
    distname=distname,
    d=d,
    seed=seed
)

diag.df <- ldply(sgld.dat, function(res) {
    x <- do.call(cbind, res$X)
    ess <- effectiveSize(x)
    data.frame(
        n=res$n,
        seed=res$seed,
        measure=c(res$objectivevalue, ess),
        diagnostic=rep(c('diagnostic = Spanner Stein', 'diagnostic = ESS'), each=2),
        dimension=paste0('dimension ', 1:length(res$objectivevalue)),
        epsilon=res$epsilon,
        solver=res$solver
    )
})

trial.df <- ddply(
    diag.df,
    .(n, diagnostic, epsilon, solver, seed),
    function (df) {
        if (df$diagnostic[1] == "diagnostic = stein") {
            c("metric"=sum(df$measure))
        } else {
            c("metric"=median(df$measure))
        }
    })

summary.df <- ddply(
    trial.df,
    .(n, diagnostic, epsilon, solver),
    function (df) {
        sd <- sd(df$metric)
        med <- median(df$metric)
        S <- length(df$metric)
        z.alpha <- abs(qnorm(0.05/2))
        c(
            "numseeds"=S,
            "median"=med,
            "mean"=mean(df$metric),
            "sd"=sd,
            "low"=med - z.alpha*sd/sqrt(S),
            "high"=med + z.alpha*sd/sqrt(S)
        )
    })

summary.df <- subset(summary.df, numseeds > 10)
summary.df <- summary.df[order(summary.df$diagnostic, summary.df$epsilon), ]
n <- diag.df$n[1]
numseeds <- median(summary.df$numseeds)

filename <- sprintf(
    "../../results/%s/figs/julia_sgld-gmm_diagnostic_n=%d.pdf",
    dir, n
)
makeDirIfMissing(filename)

pdf(file=filename, width=3, height=3.15)
ggplot(data=summary.df, aes(x=epsilon, y=log(median))) +
  geom_point(aes(color=diagnostic, shape=diagnostic)) +
  geom_path(aes(color=diagnostic)) +
  facet_wrap(~ diagnostic, scales="free_y", ncol=1) +
  labs(x=expression(paste("Step size, ", epsilon)), y="Log median diagnostic") +
  scale_x_log10(breaks=10^(-4:-2)) +
  scale_y_continuous(breaks=c(1,1.5,2,2.5,3)) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=8), legend.position="none")
dev.off()

# Prepare data for contour plot
SIGMA2Y <- 2.0
SIGMA2X1 <- 10.0
SIGMA2X2 <- 1.0
GMM.logpdf <- function(x, y) {
    log.prior.lik <- dnorm(x[1], sd=sqrt(SIGMA2X1), log=T) +
        dnorm(x[2], sd=sqrt(SIGMA2X2), log=T)
    log.likelihood <- log(0.5) + log(
        dnorm(y, mean=x[1], sd=sqrt(SIGMA2Y)) +
        dnorm(y, mean=x[1] + x[2], sd=sqrt(SIGMA2Y))
    )
    log.prior.lik + sum(log.likelihood)
}
gen.posterior.df <- function(y, xlim=c(-2.5, 3), ylim=c(-5, 4), n=120) {
    xrange <- seq(xlim[1], xlim[2], length.out=n)
    yrange <- seq(ylim[1], ylim[2], length.out=n)
    points <- expand.grid(x1=xrange, x2=yrange)
    logdens <- apply(points, 1, function (x.point) {
      GMM.logpdf(x.point, y)
    })
    cbind(points, logdens=logdens)
}

# SGLD
sgld.seed <- 7
dcast(
    subset(trial.df, seed == sgld.seed),
    epsilon ~ diagnostic,
    value.var = "metric")
contour.breaks <- -c(189, 188, 187.65)

y.sgld <- concatDataList(
    dir="compare_hyperparameters-sgld-gmm-y",
    numsamples=100,
    x="\\[0.0,1.0\\]"
)[[1]][['y']]

sgld.examples <- concatDataList(
    dir=dir,
    distname=distname,
    solver=solver,
    epsilon="(5.0e-5|0\\.005|0\\.05)",
    d=d,
    seed=sgld.seed
)
sgld.xs.df <- ldply(sgld.examples, function(res) {
    data.frame(
        distname=res$distname,
        x1=res$X[[1]],
        x2=res$X[[2]],
        epsilon=res$epsilon,
        seed=res$seed
    )
})

logdens.df <- gen.posterior.df(y.sgld,
                               xlim=range(sgld.xs.df$x1),
                               ylim=range(sgld.xs.df$x2))

filename <- sprintf(
    "../../results/%s/figs/julia_diagnostic_contour_distname=%s_n=%d_seed=%s.pdf",
    dir, distname, n, sgld.seed
)
makeDirIfMissing(filename)

epsilon_labeller <- function(variable, values) {
  sci.values <- sprintf("%.0e", values)
  sapply(sci.values, function(sci.value) {
      bquote(paste("Step size, ", epsilon, " = ", .(sci.value)))
  })
}

pdf(file=filename, width=6, height=3)
ggplot() +
    geom_point(aes(x=x1, y=x2), data=sgld.xs.df, color="black", size=1) +
    stat_contour(
        aes(x=x1, y=x2, z=logdens, color=..level..),
        size=0.4,
        binwidth=20,
        breaks=contour.breaks,
        data=logdens.df) +
    facet_grid(. ~ epsilon, labeller=epsilon_labeller) +
    labs(x=expression(x[1]), y=expression(x[2])) +
    scale_x_continuous(breaks=seq(from=-2.5, to=3, by=0.5),
                       labels=c("", "-2", "", "-1", "", "0", "", "1", "", "2", "", "3")) +
    scale_y_continuous(breaks=seq(from=-4, to=4, by=1)) +
    scale_color_gradient(name="Log posterior",
                         low="green", high="red") +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          legend.position = "none")
dev.off()
