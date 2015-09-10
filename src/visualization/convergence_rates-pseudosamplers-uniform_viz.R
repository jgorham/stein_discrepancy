# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)

source('results_data_utils.R')

### ess plots
dir <- "convergence_rates-pseudosamplers-uniform"
sampler <- NULL
distname <- "uniform"
n <- NULL
d <- 1
trial <- NULL
seed <- 7

dat <- concatDataList(
    dir="convergence_rates-pseudosamplers-uniform",
    distname=distname,
    sampler=sampler,
    n=n,
    trial=trial,
    d=d,
    seed=seed
)

ps.df <- ldply(dat, function(res) {
    first.moment <- sum(res$X[[1]] * res$q[[1]])
    first.moment.error <- abs(first.moment - 0.5)
    data.frame(
        n=res$n,
        trial=res$trial,
        seed=res$seed,
        stein=res$objectivevalue,
        sampler=res$sampler,
        distname=res$distname,
        d=res$d
    )
})

ps.df <- subset(ps.df, sampler != "sobol" | log(n, 2) %% 1 == 0)
num.unif <- length(unique(ps.df[ps.df$sampler == 'independent', 'trial']))
sum.df <- ddply(ps.df, .(distname, sampler, n, d), function (df) {
  c("stein"=median(df$stein))
})
sum.df <- sum.df[with(sum.df, order(sampler, distname, n, d)), ]

fit.df <- ddply(ps.df, .(sampler, distname, d), function (df) {
    lm.fit <- lm(log(stein, 10) ~ log(n, 10), data=df)
    c('intercept'=coef(lm.fit)[['(Intercept)']],
      'slope'=coef(lm.fit)[['log(n, 10)']])
})

sampler.rates <- sapply(1:nrow(fit.df), function(i) {
    sampler <- as.character(fit.df$sampler[i])
    sampler <- capitalizeFirstInitial(sampler)
    rate <- round(fit.df$slope[i], 2)
    bquote(.(sampler) %prop% n^.(rate))
})

filename <- sprintf(
    "../../results/%s/figs/julia_%s_distname=%s_seed=%d_d=%d_numtrials=%d.pdf",
    dir, dir, distname, seed, d, num.unif
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=2)
ggplot(data=sum.df, aes(x=n, y=stein)) +
  geom_point(aes(color=sampler, shape=sampler), size=1) +
  geom_path(aes(color=sampler, linetype=sampler)) +
  labs(x="Number of sample points, n", y="Median\nStein discrepancy") +
  scale_x_log10() +
  scale_y_log10(breaks=c(0.3, 0.1, 0.03, 0.01, 0.003)) +
   scale_color_manual(
    guide = guide_legend(title = "Sampler"),
    values=c("red", "green", "blue"),
    breaks=fit.df$sampler,
    labels=sampler.rates
  ) +
  scale_linetype_manual(
    guide = guide_legend(title = "Sampler"),
    values=c(1,3,5),
    breaks=fit.df$sampler,
    labels=sampler.rates
  ) +
  scale_shape_manual(
    guide = guide_legend(title = "Sampler"),
    values=1:3,
    breaks=fit.df$sampler,
    labels=sampler.rates
  ) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=12))
dev.off()
