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
    'iid'='i.i.d. from mixture\ntarget P',
    'unimodal'='i.i.d. from single\nmixture component'
)
DISCREPANCY.LEVELS <- c(
    'kernel'='IMQ KSD',
    'graph'='Graph Stein\ndiscrepancy',
    'wasserstein'='Wasserstein'
)
AXIS.TEXTSIZE <- 7
STRIP.TEXTSIZE <- 7
LEGEND.TEXTSIZE <- 6

dir <- "multimodal_gmm_langevin_floor"
sampler <- NULL
discrepancytype <- NULL
distname <- "2-gmm"
n <- NULL
d <- 1
gap <- 3

discrepancy.dat <- concatDataList(
    dir=dir,
    distname=distname,
    sampler=sampler,
    discrepancytype=discrepancytype,
    n=n,
    d=d,
    gap=gap,
    numcores=NULL
)

error.df <- ldply(discrepancy.dat, function(res) {
    data.frame(
        n=res$n,
        d=res$d,
        gap=res$gap,
        sampler=res$sampler,
        discrepancytype=res$discrepancytype,
        discrepancy=sum(res$objectivevalue)
    )
})

error.df <- error.df[with(error.df, order(sampler, discrepancytype, d, n, gap)), ]
error.df$d <- factor(error.df$d,
                     levels=sort(unique(error.df$d)),
                     labels=as.character(sort(unique(error.df$d))))
error.df$gap <- factor(error.df$gap,
                       levels=c(1,3,5,7),
                       labels=c('Delta==1', 'Delta==3', 'Delta==5', 'Delta==7'))
error.df$sampler <- factor(error.df$sampler,
                          levels=names(SAMPLER.LEVELS),
                          labels=as.character(SAMPLER.LEVELS))
error.df$discrepancytype <- factor(error.df$discrepancytype,
                                   levels=names(DISCREPANCY.LEVELS),
                                   labels=as.character(DISCREPANCY.LEVELS))

diagnostic.plot <- ggplot(error.df, aes(x=n, y=discrepancy)) +
    geom_point(aes(color=discrepancytype, shape=discrepancytype), size=1) +
    geom_path(aes(color=discrepancytype, linetype=discrepancytype)) +
    labs(x="Number of sample points, n",
         y="Discrepancy value") +
    facet_wrap(
        ~ sampler,
        scales="free_x"
    ) +
    scale_x_log10(
        breaks=10^seq(1, 4.5, by=1),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_y_log10(
        breaks=10^seq(-3, 0, by=0.5),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_color_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'='red',
            'IMQ KSD'='blue',
            'Wasserstein'='darkgreen'
        )
    ) +
    scale_linetype_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'=1,
            'IMQ KSD'=2,
            'Wasserstein'=3
        )
    ) +
    scale_shape_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'=1,
            'IMQ KSD'=2,
            'Wasserstein'=3
        )
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          legend.position = "left",
          legend.margin = unit(0, "cm"),
          strip.background = element_rect(fill="white",
              color="black", size=1),
          axis.text = element_text(size=AXIS.TEXTSIZE),
          axis.title = element_text(size=AXIS.TEXTSIZE+1),
          legend.title = element_text(size=LEGEND.TEXTSIZE+1),
          legend.text = element_text(size=LEGEND.TEXTSIZE),
          strip.text = element_text(size=STRIP.TEXTSIZE)
    )

ddply(error.df, .(d, gap, sampler, discrepancytype), function (df) {
    fit <- lm(log(discrepancy, 10) ~ log(n, 10), data=df)
    data.frame(slope=coef(fit)[[2]])
})

## Timing plot ##
timing.dat <- concatDataList(
    dir=dir,
    distname=distname,
    sampler="iid",
    discrepancytype="(graph|kernel)",
    n=NULL,
    d="(1|4)",
    gap=3,
    numcores="(1|4)"
)
timing.df <- ldply(timing.dat, function(res) {
    timing <- NULL
    if (res$discrepancytype == "kernel") {
        if (
            (res$ncores == 1 && res$d == 1) ||
            (res$ncores == 4 && res$d == 4)
        ) {
            timing <- res$solvetime
        }
    } else if (res$discrepancytype == "graph") {
        if (
            res$ncores == 1 && res$d == 1
        ) {
            timing <- res$edgetime + sum(res$solvetime)
        } else if (res$ncores == 1 && res$d == 4) {
            timing <- res$edgetime + max(res$solvetime)
        }
    }
    if (is.null(timing)) {
        return(data.frame())
    }
    data.frame(
        n=res$n,
        d=res$d,
        gap=res$gap,
        sampler=res$sampler,
        discrepancytype=res$discrepancytype,
        timing=timing
    )
})

timing.df <- subset(timing.df, n >= 10)
timing.df$sampler <- factor(timing.df$sampler,
                            levels=names(SAMPLER.LEVELS),
                            labels=as.character(SAMPLER.LEVELS))
timing.df$gap <- factor(timing.df$gap,
                        levels=c(1,3,5,7),
                        labels=c('Delta==1', 'Delta==3', 'Delta==5', 'Delta==7'))
timing.df$dim <- factor(timing.df$d,
                        levels=sort(unique(timing.df$d)),
                        labels=paste0("d = ", sort(unique(timing.df$d)), "\n"))
timing.df$discrepancytype <- factor(timing.df$discrepancytype,
                                    levels=names(DISCREPANCY.LEVELS),
                                    labels=as.character(DISCREPANCY.LEVELS))

timing.df <- timing.df[with(timing.df, order(sampler, discrepancytype, d, gap, n)), ]

timing.plt <- ggplot(timing.df, aes(x=n, y=timing)) +
    geom_point(aes(color=discrepancytype, shape=discrepancytype), size=1) +
    geom_path(aes(color=discrepancytype, linetype=discrepancytype)) +
    facet_wrap(
        ~ dim,
        scales="free_x"
    ) +
    labs(x="Number of sample points, n",
         y="Computation time (sec)",
         color="Discrepancy",
         shape="Discrepancy",
         linetype="Discrepancy") +
    scale_x_log10(
        breaks=10^seq(1, 4.5, by=1),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_y_log10(
        breaks=10^seq(-3, 3, by=1),
        labels=trans_format('log10',math_format(10^.x))
    ) +
    scale_color_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'='red',
            'IMQ KSD'='blue'
        )
    ) +
    scale_linetype_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'=1,
            'IMQ KSD'=2
        )
    ) +
    scale_shape_manual(
        guide = guide_legend(title = "Discrepancy"),
        values=c(
            'Graph Stein\ndiscrepancy'=1,
            'IMQ KSD'=2
        )
    ) +
    theme_bw() +
    theme(plot.margin = unit(c(0,0,0,0), "npc"),
          strip.background = element_rect(fill="white",
              color="black", size=1),
          axis.text = element_text(size=AXIS.TEXTSIZE),
          axis.title = element_text(size=AXIS.TEXTSIZE+1),
          legend.position = "none",
          strip.text = element_text(size=STRIP.TEXTSIZE)
    )

## Optimal g and h plots ##
OPT.N <- c(1000)
opt.df <- ldply(discrepancy.dat, function(res) {
    if (
        res$discrepancytype != "kernel" ||
        res$d != 1 ||
        !(res$n %in% OPT.N) ||
        res$ncores != 12
    ) {
        return(data.frame())
    }

    X <- do.call(cbind, res$X)[,1]
    gap <- res$gap
    kernel_func <- function(xx, yy) {
        (1.0 + norm(xx - yy, type="2")^2)^(-0.5)
    }
    gradx_kernel_func <- function(xx, yy) {
        -(1.0 + norm(xx - yy, type="2")^2)^(-1.5) * (xx - yy)
    }
    grady_kernel_func <- function(xx, yy) {
        gradx_kernel_func(yy, xx)
    }
    gradxy_kernel_func <- function(xx, yy) {
        d <- length(xx)
        r2 <- norm(xx - yy, type="2")^2
        d * (1.0 + r2)^(-1.5) - 3 * r2 * (1 + r2)^(-2.5)
    }
    score_function <- function(xx) {
        d <- length(xx)
        neg.mean <- c(-gap/2, rep(0,d-1))
        pos.mean <- c(gap/2, rep(0,d-1))
        wneg <- dnorm(xx - neg.mean)
        wpos <- dnorm(xx - pos.mean)
        wneg / (wneg + wpos) * (neg.mean - xx) + wpos / (wneg + wpos) * (pos.mean - xx)
    }
    kernel_func0 <- function(xx, yy) {
        bx <- score_function(xx)
        by <- score_function(yy)
        sum(bx * by) * kernel_func(xx, yy) +
            sum(bx * grady_kernel_func(xx, yy)) +
            sum(by * gradx_kernel_func(xx, yy)) +
            gradxy_kernel_func(xx, yy)
    }
    xrange <- seq(
        from=min(X),
        to=max(X),
        length.out=500
    )
    g <- sapply(xrange, function (xi) {
        vals <- sapply(X, function (xj) {
            kernel_func(xj, xi)
        })
        mean(vals)
    })
    h <- sapply(xrange, function (xi) {
        vals <- sapply(X, function (xj) {
            kernel_func0(xj, xi)
        })
        mean(vals)
    })
    data.frame(
        n=res$n,
        d=res$d,
        gap=res$gap,
        sampler=res$sampler,
        discrepancytype=res$discrepancytype,
        x=rep(xrange, 2),
        value=c(g, h),
        func=c(rep('g', 500), rep('h == T[P] ~~ g', 500))
    )
})

opt.df$sampler <- factor(opt.df$sampler,
                         levels=names(SAMPLER.LEVELS),
                         labels=as.character(SAMPLER.LEVELS))
opt.df$discrepancytype <- factor(opt.df$discrepancytype,
                                 levels=names(DISCREPANCY.LEVELS),
                                 labels=as.character(DISCREPANCY.LEVELS))
#opt.df <- opt.df[with(opt.df, order(discrepancytype, sampler, gap, n, d, x)),]

## hist.df <- ldply(discrepancy.dat, function(res) {
##     if (
##         res$discrepancytype != "kernel" ||
##         res$d != 1 ||
##         !(res$n %in% OPT.N) ||
##         res$ncores != 4
##     ) {
##         return(data.frame())
##     }
##     data.frame(
##         n=res$n,
##         d=res$d,
##         gap=res$gap,
##         sampler=res$sampler,
##         discrepancytype=res$discrepancytype,
##         x=res$X[[1]]
##     )
## })
## hist.df$sampler <- factor(hist.df$sampler,
##                           levels=names(SAMPLER.LEVELS),
##                           labels=as.character(SAMPLER.LEVELS))

gh.plt <- ggplot(opt.df, aes(x=x, y=value)) +
    geom_path(aes(color=func)) +
    facet_grid(
        func ~ sampler,
        labeller=labeller(
            .rows=label_parsed,
            .cols=label_value
        ),
        scales="free"
    ) +
    scale_x_continuous(
        breaks=c(-3,0,3),
        labels=c(3,0,3)
    ) +
    labs(
        x="x",
        y=""
    ) +
    theme_bw() +
    theme(legend.position = "none",
          strip.background = element_rect(fill="white",
              color="black", size=1),
          plot.margin = unit(c(0,0,0,0), "npc"),
          axis.text = element_text(size=AXIS.TEXTSIZE),
          axis.title = element_text(size=AXIS.TEXTSIZE+1),
          strip.text = element_text(size=STRIP.TEXTSIZE)
     )

## hist.plt <- ggplot(hist.df, aes(x=x)) +
##     geom_histogram(color="black", fill="white") +
##     facet_grid(
##         . ~ sampler,
##         scales="free_x"
##     ) +
##     labs(
##         x="x",
##         y="count",
##         title=""
##     ) +
##     theme_bw() +
##     theme(legend.position = "none",
##           plot.margin = unit(c(0,0.07,0,0), "npc"),
##           strip.background = element_blank(),
##           strip.text.x = element_blank())

filename <- sprintf(
    "../../results/%s/figs/julia_%s_comparison.pdf",
    dir, dir
)
makeDirIfMissing(filename)

pdf(file=filename, width=8.5, height=2.4)
grid.arrange(
  diagnostic.plot, timing.plt, gh.plt,
  widths = unit.c(unit(0.39, "npc"), unit(0.28, "npc"), unit(0.33, "npc")),
  nrow=1)
dev.off()

## g.plt <- ggplot(opt.df, aes(x=X, y=g)) +
##     geom_point(aes(color=discrepancytype, shape=sampler)) +
##     facet_grid(discrepancytype ~ sampler, scales="free") +
##     labs(x="x", y="g",
##          title=sprintf("Optimal discriminating functions (n=%s)", OPT.N)) +
##     scale_color_manual(
##         guide = guide_legend(title = "Discrepancy"),
##         values=c(
##             'Graph Stein discrepancy'='red',
##             'Kernel Stein discrepancy'='blue'
##         )
##     ) +
##     scale_shape_discrete(
##         guide = guide_legend(title = "Sample", override.aes = list(size=2))
##     ) +
##     theme_bw() +
v##     theme(legend.position = "none",
##           plot.margin = unit(c(0,0,0,0), "npc"),
##           strip.text = element_text(size = 8))

## h.plt <- ggplot(data=opt.df, aes(x=X, y=h)) +
##     geom_point(aes(color=discrepancytype, shape=sampler)) +
##     facet_grid(discrepancytype ~ sampler, scales="free") +
##     labs(x="x", y=expression(paste(h == T[P], ~~ g)),
##          title=sprintf("Optimal discriminating functions (n=%s)", OPT.N)) +
##     scale_color_manual(
##         guide = guide_legend(title = "Discrepancy"),
##         values=c(
##             'Graph Stein discrepancy'='red',
##             'Kernel Stein discrepancy'='blue'
##         )
##     ) +
##     scale_shape_discrete(
##         guide = guide_legend(title = "Sample", override.aes = list(size=2))
##     ) +
##     theme_bw() +
##     theme(legend.position = "none",
##           plot.margin = unit(c(0,0,0,0), "npc"),
##           strip.text = element_text(size = 8))

## filename <- sprintf(
##     "../../results/%s/figs/julia_%s_optimal_g_and_h.pdf",
##     dir, dir
## )
## makeDirIfMissing(filename)

## pdf(file=filename, width=10, height=5)
## grid.arrange(g.plt, h.plt, ncol=2)
## dev.off()
