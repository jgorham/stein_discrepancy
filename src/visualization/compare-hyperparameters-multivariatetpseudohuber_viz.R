# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(plyr)
library(reshape2)
library(coda)
library(MASS)
library(hexbin)
library(gridExtra)
library(grid)
library(stringr)

source('results_data_utils.R')

### ess plots
dir <- "compare-hyperparameters-multivariatetpseudohuber-approxwasserstein"
distname <- "multivariatetpseudohuber"
n <- NULL
sampler <- "runsgrld"
batchsize <- 30
d <- 4
seed <- NULL
nu <- "10.0"
delta <- "0.1"
epsilon <- NULL

sgrld.dat <- concatDataList(
    dir=dir,
    distname=distname,
    n=n,
    sampler=sampler,
    epsilon=epsilon,
    nu=nu,
    delta=delta,
    batchsize=batchsize,
    d=d,
    seed=seed
)

diag.df <- ldply(sgrld.dat, function(res) {
    betas <- do.call(cbind, res$betas)
    ess <- effectiveSize(betas)
    marginal2d.wass.info <- res$marginal2d_wasserstein
    marginal2d.wass <- unlist(lapply(marginal2d.wass.info, '[[', 'wasserstein'))

    data.frame(
        n=res$n,
        seed=res$seed,
        measure=c(
            sum(res$objectivevalue),
            median(ess),
            median(marginal2d.wass)),
        diagnostic=c(
            'Riemannian Stein discrepancy',
            'ESS',
            'Surrogate ground truth'),
        epsilon=res$epsilon,
        solvetime=c(mean(res$solvertime), 0, 0)
    )
})

diag.df$diagnostic <- factor(
    diag.df$diagnostic,
    levels=c('ESS', 'Riemannian Stein discrepancy', 'Surrogate ground truth'))

summary.df <- subset(diag.df, epsilon < 1e-1)
summary.df <- summary.df[order(summary.df$diagnostic, summary.df$epsilon), ]
n <- diag.df$n[1]

eps.levels <- sort(unique(summary.df$epsilon))
eps.labels <- sapply(eps.levels, function (eps.level) {
    eps.sci <- round(log(as.numeric(as.character(eps.level)), 10))
    bquote(10 ^ .(eps.sci))
})

label_wrap <- function(variable, value) {
  lapply(strwrap(as.character(value), width=40, simplify=FALSE),
    paste, collapse="\n")
}

diagnostic.plot <- ggplot(data=summary.df, aes(x=epsilon, y=log(measure))) +
  geom_point(aes(color=diagnostic, shape=diagnostic)) +
  geom_path(aes(color=diagnostic)) +
  facet_wrap(~ diagnostic, scales="free_y", labeller=label_wrap) +
  labs(x=expression(paste("Step size, ", epsilon)), y="Log diagnostic") +
  scale_x_log10(
    breaks=eps.levels,
    labels=eps.labels
  ) +
  scale_y_continuous() +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        legend.text = element_text(size=8),
#        strip.text = element_text(size=8),
        axis.text = element_text(size=8),
        legend.position="none")

filename <- sprintf(
    "../../results/%s/figs/julia_%s_diagnostics.pdf",
    dir, dir, n
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=1.8)
diagnostic.plot
dev.off()

#################################
# Prepare data for contour plot #
#################################
n.marla <- "2000000"
marla.seed <- 7
epsilon.marla <- "0.01"

marla.betas <- concatDataList(
    dir="compare-hyperparameters-multivariatetpseudohuber-MARLA",
    distname=distname,
    n=n.marla,
    epsilon=epsilon.marla,
    nu=nu,
    delta=delta,
    d=d,
    seed=marla.seed
)[[1]]

## ldply(marla.betas, function (res) {
##     data.frame(
##         epsilon=res$epsilon,
##         ar=res$acceptance_ratio
##     )
## })

marla.df <- data.frame(
    distname=marla.betas$distname,
    beta1=marla.betas$betas[[1]],
    beta2=marla.betas$betas[[2]],
    beta3=marla.betas$betas[[3]],
    beta4=marla.betas$betas[[4]],
    seed=marla.betas$seed
)

sgrld.qs <- list()
for (ii in 1:4) {
    for (jj in 1:4) {
        if (ii >= jj) {
            next
        }
        sgrld.betas <- ldply(
            Filter(function (res) {res$epsilon < 1e-1}, sgrld.dat),
            function (res) {
                cbind(res$betas[[ii]], res$betas[[jj]])
            })
        sgrld.quants <- apply(sgrld.betas, 2, quantile, c(0.02, 0.98))
        sgrld.qs <- c(sgrld.qs, list(sgrld.quants))
    }
}

HEXBIN.EPSILONS <- c(1e-2, 1e-4, 1e-7)
sgrld.hexbin.df <- ldply(
    Filter(function (res) {any(abs(res$epsilon - HEXBIN.EPSILONS) < 1e-10)}, sgrld.dat),
    function (res) {
        res.df <- data.frame()
        kk <- 0
        for (ii in 1:4) {
            for (jj in 1:4) {
                if (ii >= jj) {
                    next
                }
                kk <- kk + 1
                col1 <- paste0('beta', ii)
                col2 <- paste0('beta', jj)

                bad.i <- (
                    res$betas[[ii]] <= sgrld.qs[[kk]][1,1] |
                    res$betas[[ii]] >= sgrld.qs[[kk]][2,1] |
                    res$betas[[jj]] <= sgrld.qs[[kk]][1,2] |
                    res$betas[[jj]] >= sgrld.qs[[kk]][2,2])

                hex.res <- hexbin(res$betas[[ii]][!bad.i],
                                  res$betas[[jj]][!bad.i])
                res.df <- rbind(
                    res.df,
                    data.frame(
                        xvar=ii,
                        yvar=jj,
                        x=hex.res@xcm,
                        y=hex.res@ycm,
                        z=hex.res@count/length(res$betas[[ii]]),
                        diffusion="SGRLD",
                        epsilon=res$epsilon,
                        diffusionepsilon=paste("SGLRD", res$epsilon)
                    )
                )
            }
        }
        res.df
    })

# compute density
marla.hexbin.df <- data.frame()
for (ii in 1:4) {
    for (jj in 1:4) {
        if (ii >= jj) {
            next
        }
        col1 <- paste0('beta', ii)
        col2 <- paste0('beta', jj)

        hex.res <- hexbin(marla.df[,col1], marla.df[,col2])
        marla.hexbin.df <- rbind(
            marla.hexbin.df,
            data.frame(
                xvar=ii,
                yvar=jj,
                x=hex.res@xcm,
                y=hex.res@ycm,
                z=hex.res@count/sum(hex.res@count),
                diffusion="MARLA",
                epsilon=0,  # hack so its not used
                diffusionepsilon="MARLA"
            )
        )
    }
}

hexbin.df <- rbind(marla.hexbin.df, sgrld.hexbin.df)

hexbin.df$epsilon <- factor(hexbin.df$epsilon,
                            levels=c(0, 1e-2, 1e-4, 1e-7),
                            labels=c("", "epsilon == 10^-2", "epsilon == 10^-4", "epsilon == 10^-7"))

hexbin.df$diffusion <- factor(hexbin.df$diffusion,
                                     levels=c("MARLA", "SGRLD"),
                                     labels=c("MARLA", "SGRLD"))
hexbin.df$xvar <- factor(hexbin.df$xvar,
                         levels=1:3,
                         labels=c("x == beta[1]", "x == beta[2]", "x == beta[3]"))
hexbin.df$yvar <- factor(hexbin.df$yvar,
                         levels=2:4,
                         labels=c("y == beta[2]", "y == beta[3]", "y == beta[4]"))

hex.plot <- ggplot(data=hexbin.df) +
    stat_summary_hex(
        aes(x=x, y=y, z=z), color="grey"
    ) +
    stat_summary_hex(
        aes(x=x, y=y, z=z), color="grey"
    ) +
    facet_grid(epsilon + diffusion ~ xvar + yvar,
               labeller=label_parsed,
               scales="free_x") +
    scale_fill_gradient(
        low="white",
        high="black",
        guide = guide_legend(title="Density")) +
    theme_bw() +
    theme(
#        strip.text.y = element_text(size=6),
        plot.margin = unit(c(0,0,0,0), "npc"))

filename <- sprintf(
    "../../results/%s/figs/julia_%s_hexbin_marginal2d_scatter.pdf",
    dir, dir
)
makeDirIfMissing(filename)

pdf(file=filename, width=8, height=4.5)
hex.plot
dev.off()

###########################
##### 1d/2d marginals #####
###########################
marg1d.df <- ldply(sgrld.dat, function (res) {
    betais <- unlist(lapply(res$marginal1d_wasserstein, '[[', 'betai'))
    wasserstein <- unlist(lapply(res$marginal1d_wasserstein, '[[', 'wasserstein'))
    data.frame(
        sampler=res$sampler,
        epsilon=res$epsilon,
        covariate=betais,
        wasserstein=wasserstein
    )
})
marg1d.df <- subset(marg1d.df, epsilon < 1e-1)
marg1d.df$covariate <- factor(marg1d.df$covariate, levels=1:4)
marg1d.df$epsilon <- factor(marg1d.df$epsilon)

marg1d.plot <- ggplot(data=marg1d.df) +
    geom_bar(aes(x=epsilon, y=wasserstein, fill=covariate), stat="identity", position="dodge") +
    labs(x=expression(paste("Step size, ", epsilon)), y="Marginal Wasserstein") +
    theme_bw()

marg2d.df <- ldply(sgrld.dat, function (res) {
    betais <- unlist(lapply(res$marginal2d_wasserstein, '[[', 'betai'))
    betajs <- unlist(lapply(res$marginal2d_wasserstein, '[[', 'betaj'))
    wasserstein <- unlist(lapply(res$marginal2d_wasserstein, '[[', 'wasserstein'))
    data.frame(
        sampler=res$sampler,
        epsilon=res$epsilon,
        covariate1=betais,
        covariate2=betajs,
        wasserstein=wasserstein
    )
})
marg2d.df <- subset(marg2d.df, epsilon < 1e-1)
marg2d.df$epsilon <- factor(marg2d.df$epsilon,
                            levels=sort(unique(marg2d.df$epsilon)),
                            labels=as.character(sort(unique(marg2d.df$epsilon))))
marg2d.df$covariates <- paste(marg2d.df$covariate1, marg2d.df$covariate2, sep=",")
marg2d.df$covariates <- factor(marg2d.df$covariates)
cov.levels <- levels(marg2d.df$covariates)
guide.labels <- lapply(cov.levels, function (cov.level) {
    dims <- strsplit(cov.level, ',')[[1]]
    bquote(paste("(", beta[.(dims[1])], ",", beta[.(dims[2])], ")"))
})
names(guide.labels) <- cov.levels
eps.levels <- levels(marg2d.df$epsilon)
eps.labels <- sapply(eps.levels, function (eps.level) {
    eps.sci <- round(log(as.numeric(as.character(eps.level)), 10))
    bquote(10 ^ .(eps.sci))
})

marg2d.plot <- ggplot(data=marg2d.df) +
    geom_bar(aes(x=epsilon, y=wasserstein, fill=covariates), stat="identity", position="dodge") +
    scale_x_discrete(
        breaks=eps.levels,
        labels=eps.labels
    ) +
    scale_fill_discrete(
        labels=guide.labels,
        guide = guide_legend(title="Variates")
    ) +
    labs(x=expression(paste("Step size, ", epsilon)), y="Wasserstein (approx)") +
    theme_bw() +
    theme(legend.title = element_text(size=8),
          legend.text = element_text(size=8),
          legend.position = "none")

filename <- sprintf(
    "../../results/%s/figs/julia_%s_marginal2d_wasserstein.pdf",
    dir, dir
)
makeDirIfMissing(filename)

pdf(file=filename, width=3.5, height=3.25)
marg2d.plot
dev.off()

##########################################################
###################### GRAVEYARD #########################
##########################################################
marla.1d.df <- melt(
    marla.df,
    id.vars=c(),
    measure.vars=c('beta1', 'beta2', 'beta3', 'beta4'),
    variable.name="coeff",
    value.name="value")

ggplot(data=marla.1d.df) +
    geom_histogram(aes(x=value)) +
    facet_wrap(~ coeff) +
    labs(x="Coefficient value", y="Count") +
    theme_bw() +
    theme(legend.position = "none")

sgrld.betas.df <- ldply(
    Filter(function (res) {res$seed == 7}, sgrld.dat),
    function (res) {
        data.frame(
            epsilon=res$epsilon,
            distname=res$distname,
            beta1=res$betas[[1]],
            beta2=res$betas[[2]],
            beta3=res$betas[[3]],
            beta4=res$betas[[4]]
        )
    })

marla.betas.df <- data.frame()
for (ii in 1:4) {
    for (jj in 1:4) {
        if (ii >= jj) {
            next
        }
        col1 <- paste0('beta', ii)
        col2 <- paste0('beta', jj)

        marla.betas.df <- rbind(
            marla.betas.df,
            data.frame(
                xvar=col1,
                yvar=col2,
                x=marla.df[,col1],
                x=marla.df[,col2],
                diffusion="MARLA"
            )
        )
    }
}

sgrld.betas.df <- ldply(
    Filter(function (res) {res$seed == 7 && res$epsilon <= 2e-5}, sgrld.dat),
    function (res) {
        res.df <- data.frame()
        for (ii in 1:4) {
            for (jj in 1:4) {
                if (ii >= jj) {
                    next
                }
                col1 <- paste0('beta', ii)
                col2 <- paste0('beta', jj)
                res.df <- rbind(
                    res.df,
                    data.frame(
                        xvar=col1,
                        yvar=col2,
                        x=res$betas[[ii]],
                        y=res$betas[[jj]]
                    )
                )
            }
        }
        cbind(
            res.df,
            seed=res$seed,
            epsilon=res$epsilon,
            diffusion="SGRLD"
        )
    })

sgrld.dens.df <- ldply(
    Filter(function (res) {res$seed == 7 && res$epsilon <= 2e-5}, sgrld.dat),
    function (res) {
        res.df <- data.frame()
        for (ii in 1:4) {
            for (jj in 1:4) {
                if (ii >= jj) {
                    next
                }
                col1 <- paste0('beta', ii)
                col2 <- paste0('beta', jj)

                dens.res <- kde2d(res$betas[[ii]], res$betas[[jj]], n=kde.n)
                res.df <- rbind(
                    res.df,
                    data.frame(
                        xvar=col1,
                        yvar=col2,
                        x=rep(dens.res$x, each=kde.n),
                        y=rep(dens.res$y, kde.n),
                        z=as.vector(dens.res$z),
                        diffusion="SGRLD"
                    )
                )
            }
        }
        cbind(
            res.df,
            seed=res$seed,
            epsilon=res$epsilon
        )
    })

# compute density
marla.dens.df <- data.frame()
kde.n <- 12
for (ii in 1:4) {
    for (jj in 1:4) {
        if (ii >= jj) {
            next
        }
        col1 <- paste0('beta', ii)
        col2 <- paste0('beta', jj)

        dens.res <- kde2d(marla.df[,col1], marla.df[,col2], n=kde.n)
        marla.dens.df <- rbind(
            marla.dens.df,
            data.frame(
                xvar=col1,
                yvar=col2,
                x=rep(dens.res$x, each=kde.n),
                y=rep(dens.res$y, kde.n),
                z=as.vector(dens.res$z),
                diffusion="MARLA"
            )
        )
    }
}
