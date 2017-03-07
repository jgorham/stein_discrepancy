# script for multimodal_gmm_langevin_floor
library(ggplot2)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
library(mnormt)
library(scales)

source('results_data_utils.R')

SAMPLER.LEVELS <- c(
    'iid'='i.i.d. from P',
    'unimodal'='i.i.d. from single mode'
)

dir <- "multimodal_gmm_langevin_floor"
sampler <- NULL
discrepancytype <- "graph"
distname <- "2-gmm"
n <- NULL
d <- 1
gap <- "(1|3|5)"
parallelize <- NULL

dat <- concatDataList(
    dir=dir,
    distname=distname,
    sampler=sampler,
    discrepancytype=discrepancytype,
    n=n,
    d=d,
    gap=gap,
    parallelize=parallelize
)

sum.df <- ldply(dat, function(res) {
    data.frame(
        n=res$n,
        stein=sum(res$objectivevalue),
        sampler=res$sampler,
        distname=res$distname,
        d=res$d,
        gap=res$gap
    )
})

error.df <- melt(sum.df,
                 id=c('n', 'sampler', 'distname', 'd', 'gap'),
                 variable.name='metric.name', value.name='metric')
error.df <- subset(error.df, n <= 30000 | gap == 5)

error.df <- error.df[with(error.df, order(sampler, distname, d, n, gap)), ]
error.df$gap <- factor(error.df$gap,
                       levels=c(1,3,5,7),
                       labels=c('Delta==1', 'Delta==3', 'Delta==5', 'Delta==7'))
error.df$sampler <- factor(error.df$sampler,
                          levels=names(SAMPLER.LEVELS),
                          labels=as.character(SAMPLER.LEVELS))
error.df$metric.name <- factor(error.df$metric.name,
                              levels=c('stein', 'wasserstein'),
                              labels=c('Stein discrepancy', 'Wasserstein'))

diagnostic.plot <- ggplot(error.df, aes(x=n, y=metric)) +
    geom_point(aes(color=sampler, shape=sampler), size=1) +
    geom_path(aes(color=sampler, linetype=sampler)) +
    labs(x="Number of sample points, n",
         y="Langevin spanner\nStein discrepancy") +
    facet_grid(
        . ~ gap,
        labeller=label_parsed,
        scales="free_x"
    ) +
    scale_x_log10(
        breaks=c(10, 100, 1000, 10000, 100000),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_y_log10(
        breaks=function(limits) {
            log.ticks <- pretty(log(limits, 10), 4)
            round(10^log.ticks, max(1, abs(floor(log.ticks))))
        }
    ) +
    scale_color_manual(
        guide = guide_legend(title = "Sample"),
        values=c('i.i.d. from P'="red", 'i.i.d. from single mode'="blue")
    ) +
    scale_linetype_manual(
        guide = guide_legend(title = "Sample"),
        values=c('i.i.d. from P'=1, 'i.i.d. from single mode'=2)
    ) +
    scale_shape_manual(
        guide = guide_legend(title = "Sample"),
        values=c('i.i.d. from P'=1, 'i.i.d. from single mode'=2)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size=12),
          legend.text = element_text(size=12))

filename <- sprintf(
    "../../results/%s/figs/julia_%s_comparison.pdf",
    dir, dir
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=2)
diagnostic.plot
dev.off()

ddply(error.df, .(sampler, gap, metric.name), function (df) {
    fit <- lm(log(metric, 10) ~ log(n, 10), data=df)
    data.frame(
        rate=coef(fit)[2]
    )
})
