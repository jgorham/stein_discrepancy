# script for plotting univariate convergence bounds
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)

source('results_data_utils.R')

### filename forms
dir <- "compare_discrepancies"
distname <- NULL
n <- NULL
d <- 1
seed <- NULL

dat <- concatDataList(
    dir=dir,
    distname=distname,
    n=n,
    d=d,
    seed=seed
)

base.df <- ldply(dat, function(res) {
    as.data.frame(res)
})
bound.df <- melt(base.df, id=c("distname", "n", "seed", "d"),
                 c("classicalobjective", "wassdist", "graphobjective"),
                 variable.name="discrepancy",
                 value.name="bound")
bound.df$distname <- factor(bound.df$distname,
                            levels=c("gaussian", "uniform"),
                            labels=c("Gaussian", "Uniform"))
bound.df <- bound.df[with(bound.df, order(distname, discrepancy, seed, n)), ]
bound.df <- transform(bound.df, seedname = paste("seed", "=", seed))

filename <- sprintf(
    "../../results/%s/figs/julia_%s_d=%s.pdf",
    dir, dir, d
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=2)
ggplot(data=bound.df, aes(x=n, y=bound)) +
  geom_point(aes(color=discrepancy, shape=discrepancy), size=1) +
  geom_path(aes(color=discrepancy, linetype=discrepancy)) +
  facet_grid(distname ~ seedname, scales="free_y") +
  labs(x="Number of sample points, n", y="Discrepancy value") +
  scale_x_log10() +
  scale_y_log10(
      breaks=c(0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
  ) +
  scale_color_discrete(
      labels = c(wassdist="Wasserstein", graphobjective="Complete graph Stein", classicalobjective="Classical Stein"),
      guide = guide_legend(title = "Discrepancy")
  ) +
  scale_shape_discrete(
      labels = c(wassdist="Wasserstein", graphobjective="Complete graph Stein", classicalobjective="Classical Stein"),
      guide = guide_legend(title = "Discrepancy")
  ) +
  scale_linetype_discrete(
      labels = c(wassdist="Wasserstein", graphobjective="Complete graph Stein", classicalobjective="Classical Stein"),
      guide = guide_legend(title = "Discrepancy")
  ) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        axis.text = element_text(size=8),
        legend.text = element_text(size=9),
        strip.text = element_text(size=8)
  )
dev.off()
