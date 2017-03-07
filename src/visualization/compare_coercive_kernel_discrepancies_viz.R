# compare_coercive_kernel_discrepancies_viz
library(ggplot2)
library(plyr)
library(reshape2)
library(gridExtra)
library(scales)

source('results_data_utils.R')

SAMPLESOURCES <- list(
    "gaussian"="i.i.d. from target P",
    "randomepspacking"="Off-target sample"
)
DISCREPANCYTYPES <- list(
    "gaussian"="Gaussian",
    "matern"="Matérn",
    "gaussianpower"="Gaussian Power",
    "inversemultiquadric"="Inverse Multiquadric"
)

### filename forms
dir <- "compare_coercive_kernel_discrepancies"
samplesource <- "(gaussian|randomepspacking)"
discrepancytype <- NULL
n <- NULL
d <- "(5|8|20)"
seed <- 7

dat <- concatDataList(
    dir=dir,
    samplesource=samplesource,
    discrepancytype=discrepancytype,
    n=n,
    d=d,
    seed=seed
)

bound.df <- ldply(dat, function(res) {
    data.frame(
        n=res$n,
        d=res$d,
        samplesource=res$samplesource,
        discrepancytype=res$discrepancytype,
        seed=res$seed,
        bound=res$steindiscrepancy,
        ncores=res$ncores,
        solvetime=res$solvetime
    )
})
#bound.df <- subset(bound.df, n <= 80000)
bound.df <- subset(bound.df, discrepancytype %in% names(DISCREPANCYTYPES))
bound.df <- bound.df[order(bound.df$n), ]
bound.df$samplesource <- factor(bound.df$samplesource,
                                levels=names(SAMPLESOURCES),
                                labels=unlist(SAMPLESOURCES))
bound.df$discrepancytype <- factor(bound.df$discrepancytype,
                                   levels=names(DISCREPANCYTYPES),
                                   labels=unlist(DISCREPANCYTYPES))
bound.df$dim <- factor(bound.df$d,
                       levels=c(5,8,20),
                       labels=c('d = 5', 'd = 8', 'd = 20'))

bound.plt <- ggplot(bound.df, aes(x=n, y=bound)) +
    geom_point(aes(color=dim, shape=dim), size=1) +
    geom_path(aes(color=dim, linetype=dim)) +
    facet_grid(
        discrepancytype ~ samplesource,
        scales="free_y"
    ) +
    labs(x="Number of sample points, n", y="Kernel Stein discrepancy") +
    scale_x_log10(
        breaks=10^seq(1, 5, by=1),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_y_log10(
        breaks=10^seq(-2, 4, by=1),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    guides(
        color = guide_legend(title="Dimension"),
        shape = guide_legend(title="Dimension"),
        linetype = guide_legend(title="Dimension")
    ) +
    theme_bw() +
    theme(
        strip.background = element_rect(fill="white",
            color="black", size=1),
        plot.margin = unit(c(0,0,0,0), "npc"))

filename <- sprintf(
    "../../results/%s/figs/julia_%s_diagnostics.pdf",
    dir, dir
)
makeDirIfMissing(filename)

pdf(file=filename, width=5, height=5)
bound.plt
dev.off()
