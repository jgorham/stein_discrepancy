# incorrect_target_divergence_viz
# script for plotting univariate convergence bounds
library(ggplot2)
library(plyr)
library(reshape2)
library(gridExtra)

source('results_data_utils.R')

UNIFORM <- "Uniform"
BETA <- "Beta"
DISTS <- c(UNIFORM, BETA)
ALPHA <- 5
GRID.N <- 1000
GRID.GRAD <- 100

### filename forms
dir <- "sample_target_mismatch-multivariate_uniform_vs_beta"
target <- "uniform"
n <- NULL
d <- 2
seed <- 7

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
        bound=c(sum(res$uniform_objectivevalue), sum(res$beta_objectivevalue)),
        sample=c(UNIFORM, BETA)
    )
})
bound.df <- bound.df[order(bound.df$n), ]
bound.df$sample <- factor(bound.df$sample, levels=DISTS)

bound.plt <- ggplot(bound.df, aes(x=n, y=bound)) +
    geom_point(aes(color=sample, shape=sample), size=2) +
    geom_path(aes(color=sample, linetype=sample)) +
    labs(x="Number of sample points, n", y="Stein discrepancy") +
    scale_x_log10(
        breaks=c(100, 300, 1000, 3000)
    ) +
    scale_y_log10(
        breaks=c(0.2, 0.1, 0.05)
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          legend.text = element_text(size = 8))

func.df <- ldply(dat, function(res) {
    if (!(res$n %in% c(GRID.N))) {
        return(data.frame())
    }
    h <- c(
        res$uniform_gradg[[1]][[1]] + res$uniform_gradg[[2]][[2]],
        res$beta_gradg[[1]][[1]] + res$beta_gradg[[2]][[2]]
    ) / res$n
    data.frame(
        n=sprintf("n = %d", res$n),
        d=res$d,
        seed=res$seed,
        X1=c(res$uniform_X[[1]], res$beta_X[[1]]),
        X2=c(res$uniform_X[[2]], res$beta_X[[2]]),
        g1=c(res$uniform_g[[1]], res$beta_g[[1]]),
        g2=c(res$uniform_g[[2]], res$beta_g[[2]]),
        h=h,
        sample=rep(c(UNIFORM, BETA), c(res$n, res$n))
    )
})
func.df$sample <- factor(func.df$sample, levels=DISTS)
func.df <- melt(func.df,
                id=c("n", "d", "seed", "X1", "X2", "sample"),
                c("g1", "g2", "h"),
                variable.name="func",
                value.name="func.value")

# interpolate the values
grid.df <- ddply(func.df, .(n, d, seed, sample, func), function (df) {
    s <- seq(0, 1, length.out=GRID.GRAD)
    X <- expand.grid(X1=s, X2=s)
    smooth.func <- apply(X, 1, function(pt) {
        d <- sqrt((df$X1 - pt[[1]])^2 + (df$X2 - pt[[2]])^2)
        func.value <- df$func.value
        if (df$func[1] == "g1" || df$func[1] == "g2") {
          func.value <- c(func.value, 0, 0)
          # add boundary part
          if (df$func[1] == "g1") {
            d <- c(d, pt[[1]], 1 - pt[[1]])
          } else {
            d <- c(d, pt[[2]], 1 - pt[[2]])
          }
        }
        w <- (1/d^ALPHA) / sum(1/d^ALPHA)
        w[is.nan(w)] <- 1
        sum(w * func.value)
    })
    data.frame(
        X1=X[,1],
        X2=X[,2],
        func.value=smooth.func
    )
})

func.labels <- list(
    g1=expression(g[1]),
    g2=expression(g[2]),
    h=expression(paste(h == T[P], ~~ g))
)
facet_labeller <- function(variable, value) {
  if (variable == "func") {
    return(func.labels[value])
  }
  as.character(value)
}

g.grid.df <- subset(grid.df, func %in% c("g1", "g2"))
h.grid.df <- subset(grid.df, func %in% c("h"))
g.func.df <- subset(func.df, func %in% c("g1", "g2"))
h.func.df <- subset(func.df, func %in% c("h"))
g.range <- range(g.func.df$func.value)
h.range <- range(h.func.df$func.value)

g.func.plt <- ggplot(data=g.grid.df, aes(x=X1, y=X2)) +
    geom_tile(aes(fill=func.value)) +
    geom_point(data=g.func.df, aes(color=func.value), size=0.8) +
    facet_grid(sample ~ func, labeller=facet_labeller) +
    scale_x_continuous(breaks=c(0, 0.5, 1.0)) +
    scale_color_gradient2(limits=g.range, low="red", high="blue", guide = FALSE) +
    scale_fill_gradient2(limits=g.range, low="red", high="blue", guide = guide_legend(title="g value")) +
    labs(x=expression(x[1]), y=expression(x[2])) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          axis.text = element_text(size = 7),
          legend.text = element_text(size = 8),
          strip.text = element_text(size = 8))

h.func.plt <- ggplot(data=h.grid.df, aes(x=X1, y=X2)) +
    geom_tile(aes(fill=func.value)) +
    geom_point(data=h.func.df, aes(color=func.value), size=0.8) +
    facet_grid(sample ~ func, labeller=facet_labeller) +
    scale_x_continuous(breaks=c(0, 0.5, 1.0)) +
    scale_color_gradient2(limits=h.range, low="red", high="blue", guide = FALSE) +
    scale_fill_gradient2(limits=h.range, low="red", high="blue", guide = guide_legend(title="h value")) +
    labs(x=expression(x[1]), y=expression(x[2])) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          axis.text = element_text(size = 7),
          legend.text = element_text(size = 8),
          strip.text = element_text(size = 8))

filename <- sprintf(
    "../../results/%s/figs/julia_sample_target_mismatch-multivariate_uniform_vs_beta_target=%s_d=%d_seed=%d.pdf",
    dir, target, d, seed
)

makeDirIfMissing(filename)
pdf(file=filename, width=10, height=3)
grid.arrange(bound.plt, g.func.plt, h.func.plt, nrow=1,
  widths = unit.c(unit(0.35, "npc"), unit(0.37, "npc"), unit(0.28, "npc")))
dev.off()
