# incorrect_target_divergence_viz
# script for plotting univariate convergence bounds
library(ggplot2)
library(plyr)
library(reshape2)
library(gridExtra)

source('results_data_utils.R')

GAUSSIAN <- "Gaussian"
STUDENTT <- "Scaled\nStudent's t"
DISTS <- c(GAUSSIAN, STUDENTT)

### filename forms
dir <- "sample_target_mismatch-gaussian_vs_t"
target <- "gaussian"
n <- NULL
d <- 1
seed <- 8

dat <- concatDataList(
    dir=dir,
    target=target,
    n=n,
    d=d,
    seed=seed
)

bound.df <- ldply(dat, function(res) {
    data.frame(
        n=res$n,
        d=res$d,
        seed=res$seed,
        bound=c(res$studentt_objectivevalue, res$gaussian_objectivevalue),
        sample=c(STUDENTT, GAUSSIAN)
    )
})
bound.df <- bound.df[order(bound.df$n), ]
bound.df$sample <- factor(bound.df$sample, levels=DISTS)

bound.plt <- ggplot(bound.df, aes(x=n, y=bound)) +
    geom_point(aes(color=sample, shape=sample), size=2) +
    geom_path(aes(color=sample, linetype=sample)) +
    labs(x="Number of sample points, n", y="Stein discrepancy") +
    scale_x_log10() +
    scale_y_log10(
        breaks=c(0.3, 0.1, 0.03, 0.01)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          legend.position = "none")

argmax.df <- ldply(dat, function(res) {
    if (!(res$n %in% c(300, 3000, 30000))) {
        return(data.frame())
    }
    data.frame(
        n=sprintf("n = %d", res$n),
        d=res$d,
        seed=res$seed,
        X=c(res$gaussian_X[[1]], res$studentt_X[[1]]),
        g=c(res$gaussian_g[[1]], res$studentt_g[[1]]),
        h=c(res$gaussian_objectivefunc[[1]], res$studentt_objectivefunc[[1]]),
        sample=rep(c(GAUSSIAN, STUDENTT), c(res$n, res$n))
    )
})
argmax.df$sample <- factor(argmax.df$sample, levels=DISTS)

# downsample the paths
argmax.df <- ddply(argmax.df, .(n, sample), function(df) {
    r <- range(df$X)
    pts <- seq(r[1], r[2], length.out=10000)
    idx <- sapply(pts, function(pt) {
      which.max(df$X >= pt)
    })
    df[unique(idx), ]
})

g.plt <- ggplot(data=argmax.df, aes(x=X, y=g)) +
    geom_point(aes(color=sample, shape=sample), size=0.6) +
    facet_grid(n ~ sample) +
    labs(x="x", y="g") +
    theme_bw() +
    scale_color_discrete(
        guide = guide_legend(title = "Sample", override.aes = list(size=2))
    ) +
    scale_shape_discrete(
        guide = guide_legend(title = "Sample", override.aes = list(size=2))
    ) +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size = 8))

# copied from hadley: https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g <- ggplotGrob(g.plt + theme(legend.position="right"))$grobs
legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
lwidth <- sum(legend$width)
g.plt <- g.plt + theme(legend.position = "none")

h.plt <- ggplot(data=argmax.df, aes(x=X, y=h)) +
    geom_point(aes(color=sample, shape=sample), size=0.6) +
    facet_grid(n ~ sample, scales="free_y") +
    labs(x="x", y=expression(paste(h == T[P], ~~ g))) +
    theme_bw() +
    theme(legend.position = "none",
          plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size = 8))

filename <- sprintf(
    "../../results/%s/figs/julia_sample_target_mismatch-gaussian_vs_t_target=%s_seed=%d.pdf",
    dir, target, seed
)

makeDirIfMissing(filename)
pdf(file=filename, width=10, height=3)
grid.arrange(bound.plt, g.plt, h.plt, legend, nrow=1,
  widths = unit.c(unit(0.44, "npc") - lwidth, unit(0.28, "npc"), unit(0.28, "npc"), lwidth))
dev.off()

# assess rates for bound.df
lm.fit <- lm(log(bound, 10) ~ log(n, 10),
             data=bound.df, subset = (sample == GAUSSIAN))
summary(lm.fit)
