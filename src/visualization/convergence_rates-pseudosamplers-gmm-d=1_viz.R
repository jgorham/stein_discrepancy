# script for plotting gmm experiment
library(ggplot2)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
library(mnormt)

source('results_data_utils.R')

SAMPLER.LEVELS <- c(
    'IID'='IID',
    'FW'='FW',
    'FCFW'='FCFW',
    'QMC'='QMC'
)

dir <- "convergence_rates-pseudosamplers-gmm"
sampler <- NULL
distname <- "2-gmm"
n <- NULL
d <- 1
gap <- "(1|3|5)"

dat <- concatDataList(
    dir=dir,
    distname=distname,
    sampler=sampler,
    n=n,
    d=d,
    gap=gap
)

sum.df <- ldply(dat, function(res) {
    # |P{X} - Q{X}|
    emp.first.moment <- sum(res$X[[1]] * res$q[[1]])
    true.first.moment <- sum(res$distweights * res$mus[[1]])
    first.moment.error <- abs(true.first.moment - emp.first.moment)
    # |P{|X|} - Q{|X|}|
    distmus <- res$mus[[1]]
    K <- length(distmus)
    distweights <- res$distweights
    distsigmas <- res$sigmas
    true.first.abs.moment <- sum(
        distweights *
        sapply(1:K, function (ii) {
            distsigmas[ii]*sqrt(2/pi)*exp(-(distmus[ii]^2)/(2*distsigmas[ii]^2)) +
                distmus[ii] * (1 - 2 * pnorm(-distmus[ii] / distsigmas[ii]))
        })
    )
    emp.first.abs.moment <- sum(abs(res$X[[1]]) * res$q[[1]])
    first.abs.moment.error <- abs(true.first.abs.moment - emp.first.abs.moment)
    # 1-Lipschitz approximation of the function indicating
    # that x is within 1 standard deviation of the closest mode mu*,
    # i.e., h(x) = sig * (1 - |x-mu*|/sig)
    h <- function(x) {
      apply(sapply(seq_len(K), function(i)
        pmax(distsigmas[i] - abs(x-distmus[i]),0)),1,max)
    }
    # |P{h(X)} - Q{h(X)}|
    emp.h.moment <- sum(h(res$X[[1]]) * res$q[[1]])
    true.h.moment <-
      sum(res$distweights*
            sapply(distmus,function(mu)
              integrate(function(x) dnorm(x-mu)*h(x),-Inf,Inf)$value))
    h.moment.error <- abs(true.h.moment - emp.h.moment)
    # MMD error
    # (a) compute E_{PxP}{k(x, .)} for k with gaussian unit variance (beta)
    beta <- 1
    mmd1 <- sum(
        sapply(1:K, function(ii) {
            sapply(1:K, function (jj) {
                distweights[ii] * distweights[jj] * dnorm(
                    distmus[ii] - distmus[jj],
                    sd = sqrt(beta + distsigmas[ii] + distsigmas[[jj]])
                )
            })
        })
    )
    # (b) now compute the E_Q{mu_P(X)} part
    mmd2 <- -2 * sum(
        sapply(1:K, function(ii) {
            xterms <- dnorm(res$X[[1]], distmus[ii], sd=sqrt(distsigmas[ii] + 1))
            distweights[ii] * sum(xterms * res$q[[1]])
        })
    )
    # (c) finally compute the sample only part
    mmd3 <- sum(
        sapply(1:length(res$X[[1]]), function(ii) {
            xterms <- dnorm(res$X[[1]], mean=res$X[[1]][ii], sd=sqrt(beta))
            res$q[[1]][ii] * sum(xterms * res$q[[1]])
        })
    )
    # result
    data.frame(
        n=res$n,
        seed=res$seed,
        stein=sum(res$objectivevalue),
        sampler=res$sampler,
        distname=res$distname,
        d=res$d,
        gap=res$gap,
        wasserstein=res$wasserstein,
        mmd=sqrt(mmd1 + mmd2 + mmd3),
        first.moment.error=first.moment.error,
        first.abs.moment.error=first.abs.moment.error,
        htest=h.moment.error
    )
})

error.df <- melt(sum.df, id=c('n', 'seed', 'sampler', 'distname', 'd', 'gap'),
                 variable.name='metric.name', value.name='metric')
error.df <- ddply(error.df,
                  .(n, sampler, distname, d, gap, metric.name),
                  function (df) {
    data.frame(metric=median(df$metric))
})
error.df <- error.df[with(error.df, order(sampler, distname, d, n, gap)), ]
error.df$gap <- factor(error.df$gap,
                       levels=c(1,3,5,7),
                       labels=c('Delta==1', 'Delta==3', 'Delta==5', 'Delta==7'))
wass.df <- subset(error.df,
                  metric.name %in% c('stein', 'wasserstein', 'htest'))
wass.df$sampler <- factor(wass.df$sampler,
                          levels=names(SAMPLER.LEVELS),
                          labels=as.character(SAMPLER.LEVELS))
wass.df$metric.name <- factor(wass.df$metric.name,
                              levels=c('stein', 'wasserstein', 'htest'),
                              labels=c('Langevin~Stein', 'Wasserstein', 'abs(E[P]~h[1](Z)- E[Q]~h[1](X))'))

diagnostic.plot <- ggplot(wass.df, aes(x=n, y=metric)) +
    geom_point(aes(color=sampler, shape=sampler), size=1) +
    geom_path(aes(color=sampler, linetype=sampler)) +
    labs(x="Number of sample points, n",
         y="Median diagnostic") +
    facet_grid(
        metric.name ~ gap,
        scales="free_y",
        labeller=labeller(.cols=label_parsed, .rows=label_parsed)
    ) +
    scale_x_log10(
        breaks=c(10, 30, 100, 300)
    ) +
    scale_y_log10(
        breaks=function(limits) {
            log.ticks <- pretty(log(limits, 10), 4)
            round(10^log.ticks, max(1, abs(floor(log.ticks))))
        }
    ) +
    scale_color_manual(
        guide = guide_legend(title = "Metric"),
        values=c(IID="red", FW="green", FCFW="blue", QMC="purple")
    ) +
    scale_linetype_manual(
        guide = guide_legend(title = "Metric"),
        values=c(IID=1, FW=2, FCFW=3, QMC=4)
    ) +
    scale_shape_manual(
        guide = guide_legend(title = "Metric"),
        values=c(IID=1, FW=2, FCFW=3, QMC=4)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          axis.title = element_text(size=17),
          strip.text = element_text(size=11),
          legend.text = element_text(size=12))

## filename <- sprintf(
##     "../../results/%s/figs/julia_%s_distname_%s_univariate_metrics_panels.pdf",
##     dir, dir, distname
## )
## makeDirIfMissing(filename)
## pdf(file=filename, width=9, height=6)
## diagnostic.plot
## dev.off()

#################
# Plot of htest #
#################
gaps <- unique(sum.df$gap)
htest.df <- adply(gaps, 1, function (gap) {
    xx <- seq(from=-5, to=5, by=0.1)
    hfunc <- function(xx) {
        max(0, 1 - min(abs(xx - gap/2), abs(xx + gap/2)))
    }
    hh <- sapply(xx, hfunc)
    data.frame(
        gap=paste0("Delta==", gap),
        x=xx,
        h=hh
    )
})

htest.plot <- ggplot(htest.df, aes(x=x, y=h)) +
    geom_path(color="blue") +
    labs(x="x",
         y=expression(paste(h[1],"(x)"))
    ) +
    facet_wrap(
        ~ gap,
#        scales="free_x",
        labeller=label_parsed
    ) +
    scale_y_continuous(
        breaks=c(0,1)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0.08,0.205,0,0.033), "npc"),
          axis.title = element_text(size=17),
          plot.title = element_text(size=14)
    )

## filename <- sprintf(
##     "../../results/%s/figs/julia_%s_distname_%s_htest.pdf",
##     dir, dir, distname
## )
## makeDirIfMissing(filename)

## pdf(file=filename, width=9, height=2.5)
## htest.plot
## dev.off()

filename <- sprintf(
    "../../results/%s/figs/julia_%s_distname_%s_univariate_metrics.pdf",
    dir, dir, distname
)
makeDirIfMissing(filename)

pdf(file=filename, width=7, height=7.25)
grid.arrange(diagnostic.plot, htest.plot, ncol=1,
             heights = unit.c(unit(0.73, "npc"), unit(0.27, "npc")))
dev.off()

########################
# 1-d Stein optimal gs #
########################
SEED <- 7
D <- 1
GAP <- 5
N <- 200
fw1.res <- Filter(function (res) {
    res$n == N && res$sampler == "FW" && res$gap == GAP && res$d == D
}, dat)[[1]]
iid1.res <- Filter(function (res) {
    res$n == N && res$sampler == "IID" && res$gap == GAP && res$d == D && res$seed == SEED
}, dat)[[1]]
fcfw1.res <- Filter(function (res) {
    res$n == N && res$sampler == "FCFW" && res$gap == GAP && res$d == D
}, dat)[[1]]
qmc1.res <- Filter(function (res) {
    res$n == N && res$sampler == "QMC" && res$gap == GAP && res$d == D
}, dat)[[1]]

samples1.df <- rbind(
    data.frame(
        x1=iid1.res$X[[1]],
        q=iid1.res$q[[1]],
        sampler="IID"
    ),
    data.frame(
        x1=fw1.res$X[[1]],
        q=fw1.res$q[[1]],
        sampler="FW"
    ),
    data.frame(
        x1=fcfw1.res$X[[1]],
        q=fcfw1.res$q[[1]],
        sampler="FCFW"
    ),
    data.frame(
        x1=qmc1.res$X[[1]],
        q=qmc1.res$q[[1]],
        sampler="QMC"
    )
)
samples1.df$sampler <- factor(
    samples1.df$sampler,
    levels=SAMPLER.LEVELS)

gmm1.fun <- function (xx) {
    K <- length(iid1.res$mus[[1]])
    comps <- sapply(1:K, function (kk) {
        dnorm(xx, mean=iid1.res$mus[[1]][kk], sd = sqrt(iid1.res$sigmas[kk]))
    })
    comps %*% iid1.res$distweights
}

hist1d.plot <- ggplot(data=samples1.df) +
    geom_histogram(aes(x=x1, y=..density.., weight=q), color="black", fill="white",
                   binwidth=0.2) +
    stat_function(
        size=1.25,
        color="blue",
        fun=gmm1.fun
    ) +
    facet_wrap(~ sampler, ncol=2) +
    labs(
        x=expression(x)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0, 0, 0, 0.03), "npc"),
          plot.title = element_text(size=16),
          axis.title = element_text(size=16),
          text = element_text(size=13))

## filename <- sprintf(
##     "../../results/%s/figs/julia_%s_distname_%s_univariate_histogram_panels.pdf",
##     dir, dir, distname
## )
## makeDirIfMissing(filename)

## pdf(file=filename, width=4.5, height=3.6)
## hist1d.plot
## dev.off()

h.df <- rbind(
    data.frame(
        x=iid1.res$X[[1]],
        h=iid1.res$operatorg[[1]],
        sampler='IID'
    ),
    data.frame(
        x=fw1.res$X[[1]],
        h=fw1.res$operatorg[[1]],
        sampler='FW'
    ),
    data.frame(
        x=qmc1.res$X[[1]],
        h=qmc1.res$operatorg[[1]],
        sampler='QMC'
    ),
    data.frame(
        x=fcfw1.res$X[[1]],
        h=fcfw1.res$operatorg[[1]],
        sampler='FCFW'
    )
)
h.df <- h.df[with(h.df, order(sampler, x)),]
h.df$sampler <- factor(h.df$sampler,
                       levels=SAMPLER.LEVELS)

hopt1d.plot <- ggplot(data=h.df) +
    geom_point(aes(x=x, y=h), color="black", size=0.5) +
    geom_path(aes(x=x, y=h), color="black") +
    facet_wrap(~ sampler, ncol=2) +
    labs(x=expression(x),
         y=expression(paste(h^"*"~ (x) == (T ~ g^"*")(x)))) +
    theme_bw() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "npc"),
          plot.title = element_text(size=16),
          axis.title = element_text(size=15),
          text = element_text(size=13))

## filename <- sprintf(
##     "../../results/%s/figs/julia_%s_distname_%s_univariate_optimalh.pdf",
##     dir, dir, distname
## )
## makeDirIfMissing(filename)

## pdf(file=filename, width=4.5, height=3.6)
## hopt1d.plot
## dev.off()

filename <- sprintf(
    "../../results/%s/figs/julia_%s_distname_%s_univariate_histograms.pdf",
    dir, dir, distname
)
makeDirIfMissing(filename)

pdf(file=filename, width=4.5, height=6.8)
grid.arrange(hist1d.plot, hopt1d.plot, ncol=1)
dev.off()

#############
# Graveyard #
#############
sum.df <- sum.df[with(sum.df, order(sampler, distname, d, n, gap)), ]
sum.df <- subset(sum.df, gap > 1)
head(sum.df)

filename <- sprintf(
    "../../results/%s/figs/julia_%s_distname_%s_graphsteindiscrepancies.pdf",
    dir, dir, distname
)
makeDirIfMissing(filename)

pdf(file=filename, width=12, height=9)
ggplot(data=sum.df, aes(x=n, y=stein)) +
  geom_point(aes(color=sampler, shape=sampler), size=1) +
  geom_path(aes(color=sampler, linetype=sampler)) +
  labs(x="Number of sample points, n", y="Graph Stein discrepancy") +
  facet_wrap(~ gap, scales="free_y", ncol=1) +
  scale_x_log10(breaks=c(10, 30, 100, 300)) +
  scale_y_log10(
    breaks=c(0.05, 0.1, 0.2, 0.5, 1.0)
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
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=12))
dev.off()
