# compare_coercive_kernel_discrepancies_viz
library(ggplot2)
library(plyr)
library(reshape2)
library(gridExtra)
library(scales)

source('results_data_utils.R')

REWEIGHT.SCHEMES <- c(
    'None',
    'Gaussian KSD',
    'IMQ KSD',
    'Gaussian CF',
    'IMQ CF',
    'Gaussian CF (Normalized)',
    'IMQ (Normalized)'
)

### filename forms
dir <- "blackweights"
n <- 100
K <- 500

avg.err.df <- concatData(
    dir=dir,
    header.names=c('d', REWEIGHT.SCHEMES),
    prefix="matlab_standard_gaussian_mean",
    n=n
)
# HACK: I have no idea why R is dropping the last d in the tsv file
avg.err.df$d <- c(2,10, 50, 75, 100)
avg.err.df <- avg.err.df[,c('d', 'None', 'Gaussian KSD', 'IMQ KSD')]

std.err.df <- concatData(
    dir=dir,
    header.names=c('d', REWEIGHT.SCHEMES),
    prefix="matlab_standard_gaussian_std",
    n=n
)
std.err.df$d <- c(2,10, 50, 75, 100)
std.err.df <- std.err.df[,c('d', 'None', 'Gaussian KSD', 'IMQ KSD')]

avg.df <- melt(avg.err.df, id.vars=c("d"), value.name="MSE", variable.name="method")

ALPHA <- 0.05
low.errs <- qt(ALPHA/2, df=K-1) * std.err.df[,-1] / sqrt(K) + avg.err.df[,-1]
low.errs <- cbind(d=std.err.df[,1], low.errs)
hi.errs <- qt(1-ALPHA/2, df=K-1) * std.err.df[,-1] / sqrt(K) + avg.err.df[,-1]
hi.errs <- cbind(d=std.err.df[,1], hi.errs)

low.df <- melt(low.errs, id.vars="d", value.name="low", variable.name="method")
hi.df <- melt(hi.errs, id.vars="d", value.name="high", variable.name="method")
ci.df <- merge(low.df, hi.df, by=c("d", "method"))

all.df <- merge(avg.df, ci.df, by=c("d", "method"))
all.df <- all.df[order(all.df$method, all.df$d),]

err.plt <- ggplot(all.df, aes(x=d, y=MSE)) +
    geom_errorbar(aes(x=d, ymin=low, ymax=high, color=method)) +
    geom_path(aes(color=method, linetype=method), size=0.8) +
    geom_point(aes(color=method, shape=method), size=1.5) +
    labs(x="Dimension, d",
         y=expression(paste("Average MSE, ||", E[P]~Z - E[tilde(Q[n])]~X, "||"[2]^2 / d))) +
    scale_x_continuous(
        breaks=sort(unique(all.df$d))
    ) +
    scale_y_log10(
        breaks=10^seq(-6, 0, by=0.5),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_color_discrete(
        breaks=c('None', 'Gaussian KSD', 'IMQ KSD'),
        labels=c(
            expression(paste("Initial", ~ Q[n])),
            'Gaussian KSD',
            'IMQ KSD'
        )
    ) +
    scale_linetype_discrete(
        breaks=c('None', 'Gaussian KSD', 'IMQ KSD'),
        labels=c(
            expression(paste("Initial", ~ Q[n])),
            'Gaussian KSD',
            'IMQ KSD'
        )
    ) +
    scale_shape_discrete(
        breaks=c('None', 'Gaussian KSD', 'IMQ KSD'),
        labels=c(
            expression(paste("Initial", ~ Q[n])),
            'Gaussian KSD',
            'IMQ KSD'
        )
    ) +
    guides(
        color = guide_legend(title="Sample"),
        shape = guide_legend(title="Sample"),
        linetype = guide_legend(title="Sample")
    ) +
    theme_bw() +
    theme(
        legend.text.align = 0,
        axis.title = element_text(size=10),
        strip.background = element_rect(fill="white",
            color="black", size=1),
        plot.margin = unit(c(0,0,0,0), "npc"))

filename <- sprintf(
    "../../results/%s/figs/julia_%s_n=%d.pdf",
    dir, dir, n
)
makeDirIfMissing(filename)

pdf(file=filename, width=5, height=2.5)
err.plt
dev.off()
