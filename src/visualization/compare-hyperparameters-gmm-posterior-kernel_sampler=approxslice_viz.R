# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
library(coda)
library(hexbin)

source('results_data_utils.R')

### ess plots
dir <- "compare-hyperparameters-gmm-posterior"
discrepancytype <- "inversemultiquadric"
n <- 1000
d <- 2
seed <- NULL
distname <- "gmm-posterior"

sampler.dat <- concatDataList(
    dir=dir,
    distname=distname,
    discrepancytype=discrepancytype,
    sampler="approxslice",
    n=n,
    d=d,
    seed=seed
)

diag.df <- ldply(sampler.dat, function(res) {
    x <- do.call(cbind, res$X)
    ess <- effectiveSize(x)
    ess.avg <- mean(ess)
    data.frame(
        n=nrow(x),
        seed=res$seed,
        metric=c(res$objectivevalue, ess.avg),
        solvetime=res$solvetime,
        ncores=res$ncores,
        diagnostic=rep(c('KSD (lower is better)', 'ESS (higher is better)')),
        epsilon=res$epsilon
    )
})

summary.df <- ddply(
    diag.df,
    .(diagnostic, epsilon),
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

summary.df <- subset(summary.df, numseeds >= 10)
summary.df <- summary.df[order(summary.df$diagnostic, summary.df$epsilon), ]
numseeds <- median(summary.df$numseeds)

format_eps <- get_epsilon_labeller(variable="")
eps.breaks <- sort(unique(summary.df$epsilon))
eps.labels <- format_eps(eps.breaks)
summary.df$epsilon <- factor(
    summary.df$epsilon,
    levels=eps.breaks,
    labels=as.char(eps.breaks))

diagnostic.plt <- ggplot(data=summary.df, aes(x=epsilon, y=log(median))) +
  geom_point(aes(color=diagnostic, shape=diagnostic)) +
  geom_path(aes(color=diagnostic, group=diagnostic)) +
  facet_wrap(~ diagnostic, scales="free_y", ncol=1) +
  labs(x=expression(paste("Tolerance parameter, ", epsilon)), y="Log median diagnostic") +
  scale_x_discrete(
      breaks=as.char(eps.breaks),
      labels=parse(text=eps.labels)
  ) +
  scale_y_continuous(breaks=seq(-1.0, 5.0, by=0.5)) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        axis.text = element_text(size=7),
        axis.title = element_text(size=10),
        strip.background = element_rect(fill="white",
            color="black", size=1),
        legend.text = element_text(size=8), legend.position="none")

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

# SAMPLER
sampler.seed <- 7
## dcast(
##     subset(trial.df, seed == sampler.seed),
##     epsilon ~ diagnostic,
##     value.var = "metric")

y.sampler <- concatDataList(
    dir="compare-hyperparameters-gmm-posterior-y",
    numsamples=100,
    x="\\[0.0,1.0\\]"
)[[1]][['y']]

sampler.examples <- concatDataList(
    dir=dir,
    distname=distname,
    discrepancytype=discrepancytype,
    sampler="approxslice",
    n=n,
    epsilon="(0.0|0.01|0.1)",
    d=d,
    seed=sampler.seed
)

format_eps <- get_epsilon_labeller()
sampler.xs.df <- ldply(sampler.examples, function(res) {
    n <- length(res$X[[1]])
    epsilon <- res$epsilon
    eps.and.n.label <- paste0(
        format_eps(epsilon),
        "~(n == ", n, ")")
    data.frame(
        distname=res$distname,
        x1=res$X[[1]],
        x2=res$X[[2]],
        epsilon=epsilon,
        n=n,
        seed=res$seed,
        eps.and.n.label=eps.and.n.label
    )
})

eps.values <- daply(sampler.xs.df, .(eps.and.n.label), function (df) {
    df$epsilon[1]
})
eps.values <- sort(eps.values)
sampler.xs.df$eps.and.n.label <- factor(
    sampler.xs.df$eps.and.n.label,
    levels=names(eps.values),
    labels=names(eps.values))

# x1
x1.range <- range(sampler.xs.df$x1)
x1.diff <- diff(x1.range)
x1.range[1] <- x1.range[1] - x1.diff * 0.05
x1.range[2] <- x1.range[2] + x1.diff * 0.05
# x2
x2.range <- range(sampler.xs.df$x2)
x2.diff <- diff(x2.range)
x2.range[1] <- x2.range[1] - x2.diff * 0.05
x2.range[2] <- x2.range[2] + x2.diff * 0.05

loghex.df <- ddply(sampler.xs.df,
                   .(distname, eps.and.n.label, seed),
                   function (df) {
    hex.res <- hexbin(df$x1, df$x2, xbnds=x1.range, ybnds=x2.range)
    data.frame(
        x1=hex.res@xcm,
        x2=hex.res@ycm,
#        z=hex.res@count/nrow(df)
        z=log(hex.res@count/nrow(df))
    )
})

logdens.df <- gen.posterior.df(y.sampler,
                               xlim=range(sampler.xs.df$x1),
                               ylim=range(sampler.xs.df$x2))
contour.breaks <- quantile(logdens.df$logdens,
#                           c(0.985, 0.99, 0.994))
                           c(0.945, 0.975, 0.99))

scatter.plts <- ggplot() +
    ## stat_summary_hex(
    ##     aes(x=x1, y=x2, z=z),
    ##     color="grey",
    ##     data=loghex.df
    ## ) +
    ## stat_binhex(
    ##     aes(x=x1, y=x2),
    ##     data=sampler.xs.df,
    ##     bins=30
    ## ) +
    geom_point(
        aes(x=x1, y=x2),
        data=sampler.xs.df,
        color="black",
        alpha=0.2,
        size=1) +
    stat_contour(
        aes(x=x1, y=x2, z=logdens, color=..level..),
        size=0.4,
        binwidth=20,
        breaks=contour.breaks,
        data=logdens.df) +
    facet_grid(
        . ~ eps.and.n.label,
        labeller=labeller(.cols=label_parsed)
    ) +
    labs(x=expression(x[1]), y=expression(x[2])) +
    scale_x_continuous(breaks=seq(from=-2.5, to=3, by=0.5),
                       labels=c("", "-2", "", "-1", "", "0", "", "1", "", "2", "", "3")) +
    scale_y_continuous(breaks=seq(from=-4, to=4, by=1)) +
    scale_color_gradient(name="Log posterior",
                         low="green", high="red") +
    scale_fill_gradient(
        low="white",
        high="black",
        guide = guide_legend(title="Density")
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          strip.background = element_rect(fill="white",
              color="black", size=1),
          panel.grid.major = element_blank(),
#          panel.grid.minor = element_blank(),
          legend.position = "none")

filename <- sprintf(
    "../../results/%s/figs/julia_diagnostic_contour_distname=%s_sampler=approxslice_n=%d_seed=%s.pdf",
    dir, distname, n, sampler.seed
)
makeDirIfMissing(filename)

pdf(file=filename, width=8.5, height=2.5)
grid.arrange(
    diagnostic.plt,
    scatter.plts,
    widths=unit.c(unit(0.23, "npc"), unit(0.76, "npc")),
    nrow=1)
dev.off()
