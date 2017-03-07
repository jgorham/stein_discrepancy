# script for plotting gmm experiment
library(ggplot2)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
library(mnormt)

source('results_data_utils.R')

SAMPLER.LEVELS <- c('IID', 'FW', 'FCFW', 'QMC')

dir <- "convergence_rates-pseudosamplers-gmm"
sampler <- NULL
distname <- "2-gmm"
n <- NULL
d <- 2
gap <- NULL
wassersteinn <- 2000

dat <- concatDataList(
    dir=dir,
    distname=distname,
    sampler=sampler,
    n=n,
    d=d,
    gap=gap,
    wassersteinn=wassersteinn
)

sum.df <- ldply(dat, function(res) {
    data.frame(
        n=res$n,
        diagnostic=c('Stein discrepancy','Wasserstein'),
        metric=c(sum(res$objectivevalue), res$wasserstein),
        metric.lb=c(sum(res$objectivevalue), res$wasserstein_lb),
        metric.ub=c(sum(res$objectivevalue), res$wasserstein_ub),
        sampler=res$sampler,
        distname=res$distname,
        d=res$d,
        gap=res$gap
    )
})

sum.df <- sum.df[with(sum.df, order(sampler, distname, d, n, gap)), ]
sum.df$sampler <- factor(sum.df$sampler,
                         levels=SAMPLER.LEVELS)
sum.df <- subset(sum.df, gap > 1)
sum.df$gap <- factor(sum.df$gap,
                     levels=c(3,5,7),
                     labels=c('3', '5', '7'))
sum.df

custom_labeller <- function(variable, value) {
    if (variable == "diagnostic") {
        lapply(strwrap(as.character(value), width=11, simplify=FALSE),
               paste, collapse="\n")
    } else {
        lapply(
            as.character(value),
            function (gap) {bquote(gamma == .(gap))}
        )
    }
}

filename <- sprintf(
    "../../results/%s/figs/julia_%s_distname=%s_bivariate_metrics.pdf",
    dir, dir, distname
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=3)
ggplot(data=sum.df, aes(x=n, y=metric)) +
  geom_errorbar(aes(ymax=metric.ub, ymin=metric.lb, color=sampler)) +
  geom_point(aes(color=sampler, shape=sampler), size=1) +
  geom_path(aes(color=sampler, linetype=sampler)) +
  labs(x="Number of sample points, n",
       y="Diagnostic") +
  facet_grid(diagnostic ~ gap,
             scales="free_y",
             labeller=custom_labeller) +
  scale_x_log10(breaks=c(10, 30, 100, 300)) +
  scale_y_log10(
    breaks=round((10^(1/6))^(-12:6), 2)
  ) +
  scale_color_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID="red", FW="green", FCFW="blue", QMC="purple")
  ) +
  scale_linetype_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID=1, FW=2, FCFW=3, QMC=4)
  ) +
  scale_shape_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID=1, FW=2, FCFW=3, QMC=4)
  ) +
  theme_bw()
dev.off()

##########
# Extras #
##########
sum_bign.df <- subset(sum.df, n == 200)
ggplot(data=sum_bign.df, aes(x=gap, y=metric)) +
  geom_errorbar(aes(ymax=metric.ub, ymin=metric.lb, color=sampler), width=0.3) +
  geom_point(aes(color=sampler, shape=sampler), size=1) +
  geom_path(aes(color=sampler, linetype=sampler)) +
  labs(x="Distance between modes", y="Graph Stein discrepancy") +
  facet_wrap(~ diagnostic, scales="free_y", ncol=1) +
#  scale_x_log10(breaks=c(10, 30, 100, 300)) +
#  scale_y_log10() +
   scale_color_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID="red", FW="green", FCFW="blue", QMC="purple")
  ) +
  scale_linetype_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID=1, FW=2, FCFW=3, QMC=4)
  ) +
  scale_shape_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(IID=1, FW=2, FCFW=3, QMC=4)
  ) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=12))

#################
# Scatter plots #
#################
GAP <- 5
N <- 200
fw1.res <- Filter(function (res) {res$n == N && res$sampler == "FW" && res$gap == GAP}, dat)[[1]]
iid1.res <- Filter(function (res) {res$n == N && res$sampler == "IID" && res$gap == GAP}, dat)[[1]]
fcfw1.res <- Filter(function (res) {res$n == N && res$sampler == "FCFW" && res$gap == GAP}, dat)[[1]]
qmc1.res <- Filter(function (res) {res$n == N && res$sampler == "QMC" && res$gap == GAP}, dat)[[1]]

fw1.X <- do.call(cbind, fw1.res$X)
iid1.X <- do.call(cbind, iid1.res$X)
qmc1.X <- do.call(cbind, qmc1.res$X)
fcfw1.X <- do.call(cbind, fcfw1.res$X)
fcfw1.X <- fcfw1.X[fcfw1.res$q[[1]] >= 1e-4,]

samples2.df <- rbind(
    data.frame(
        x1=iid1.X[,1],
        x2=iid1.X[,2],
        sampler="iid"
    ),
    data.frame(
        x1=fw1.X[,1],
        x2=fw1.X[,2],
        sampler="fw"
    ),
    data.frame(
        x1=fcfw1.X[,1],
        x2=fcfw1.X[,2],
        sampler="fcfw"
    ),
    data.frame(
        x1=qmc1.X[,1],
        x2=qmc1.X[,2],
        sampler="qmc"
    )
)

gen_gmm_logdensity <- function(mus, sigmas, weights) {
    function (xx) {
        k <- length(weights)
        logwtdcomponents <- log(weights) +
            sapply(1:k, function (ii) {dmnorm(xx, mus[ii,], sigmas[[ii]], log=T)})
        maxterm <- max(logwtdcomponents)
        maxterm + log(sum(exp(logwtdcomponents - maxterm)))
    }
}
gmm_logdensity <- gen_gmm_logdensity(
    do.call(cbind, iid1.res$mus),
#    t(iid1.res$mus),
    lapply(iid1.res$sigmas, function (res) {
        ss <- do.call(cbind, res)
        ss <- 0.5 * (ss + t(ss))  # needed b/c numerical issues can make ss asymmetric
        ss
    }),
    iid1.res$distweights
)
gen_logdensity_df <- function(xlim=c(-6, 6), ylim=c(-6, 6), n=150) {
    xrange <- seq(xlim[1], xlim[2], length.out=n)
    yrange <- seq(ylim[1], ylim[2], length.out=n)
    points <- expand.grid(x1=xrange, x2=yrange)
    logdens <- apply(points, 1, function (x.point) {
        gmm_logdensity(x.point)
    })
    cbind(points, logdens=logdens)
}
logdensity.df <- gen_logdensity_df()

filename <- sprintf(
    "../../results/%s/figs/julia_%s_contour2d_distname=%s_n=200.pdf",
    dir, dir, distname
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=5)
ggplot() +
    geom_point(aes(x=x1, y=x2), data=samples2.df, color="black", size=1) +
    stat_contour(
        aes(x=x1, y=x2, z=logdens, color=..level..),
        size=0.4,
        binwidth=20,
        breaks=c(-6, -5, -4, -3.5, -3, -2.5, -2),
        data=logdensity.df) +
    facet_wrap(~ sampler, ncol=2) +
    labs(x=expression(x[1]), y=expression(x[2]),
         labeller=label_bquote(cols=paste("sampler = ", .(sampler)))) +
    scale_x_continuous(breaks=seq(from=-8, to=8, by=2)) +
    scale_y_continuous(breaks=seq(from=-8, to=8, by=2)) +
    scale_color_gradient(name="Log density",
                         low="green", high="red") +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"))
dev.off()

## fcfw1.X <- do.call(cbind, fcfw1.res$X)
## fcfw.dat <- data.frame(
##     x1=fcfw1.X[,1],
##     x2=fcfw1.X[,2],
##     w=fcfw1.res$q
## )
## ggplot() +
##     geom_point(aes(x=x1, y=x2, size=w), data=fcfw.dat, color="black") +
##     stat_contour(
##         aes(x=x1, y=x2, z=logdens, color=..level..),
##         size=0.4,
##         binwidth=20,
##         breaks=c(-6, -5, -4, -3.5, -3, -2.5, -2),
##         data=logdensity.df) +
##     labs(x=expression(x[1]), y=expression(x[2])) +
##     scale_color_gradient(name="Log density",
##                          low="green", high="red") +
##     theme_bw() +
##     theme(plot.margin = unit(c(0,0,0,0), "npc"))
