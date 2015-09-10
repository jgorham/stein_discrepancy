# This file contains the implementation of various
# Monte Carlo samplers.

using SteinDistributions
using Distributions: MvNormal, TDist, ccdf

# This function generates a lambda function
# in the form of a power law. This is useful
# for computing the stepsize for algorithms
# that require some changing step size.
function epsilon_generator(;a=1.0, b=1.0, gamma=0.6)
    t::Int64 -> a * (b + t) ^ -gamma;
end

# A composition of the int and floor functions.
function intf(x::Float64)
    int(floor(x))
end

# This function runs the approximate Metropolis-Hastings
# sampling procedure as described in Korattikara, Chen,
# and Welling.
#
# Inputs:
# d                - the target distribution to approximate a sample from
# theta0           - the starting point for the sampler
# numiter          - the number of iterations to run the sampler (i.e. number of points)
# numlikelihood    - the number of likelihood evaluations to make before halting
# batchsize        - the number of observations to evaluate before doing the significance test
# epsilon          - the threshold for assessing the significance test
# proposalvariance - the variance of the mean-zero Gaussian proposal distribution
#
# Returns:
# (samples, number of gradient evaluations)
function runapproxmh(d::SteinDistribution,
                     theta0::Array{Float64};
                     numiter=Inf,
                     numlikelihood=Inf,
                     batchsize=10,
                     epsilon=0.1,
                     proposalvariance=1e-2)

    if isinf(numiter) && isinf(numlikelihood)
        error("Need to specify numlikelihood or numiter")
    end

    L = numdatapoints(d)
    p = length(theta0)
    t = 0
    noise = MvNormal(p, 1)
    thetas = Array(Float64, 0, p)
    theta_t = theta0
    numlikelihoods = Int64[]
    cumlikelihoods = 0

    while t < numiter && cumlikelihoods < numlikelihood
        significancelevel = 1.0
        currbatchsize = 0
        ldiffmean = 0.0
        perm = randperm(L)

        delta = sqrt(proposalvariance) .* vec(rand(noise, 1))
        prop_t = theta_t + delta

        logu = log(rand())
        lprior = logprior(d, prop_t) - logprior(d, theta_t)
        mu0 = (logu - lprior) / L

        while (currbatchsize < L) && (significancelevel > epsilon)
            currbatchsize = min(currbatchsize + batchsize, L)
            ldiffs = (
                loglikelihood(d, prop_t; idx=perm[1:currbatchsize]) -
                loglikelihood(d, theta_t; idx=perm[1:currbatchsize])
            )
            ldiffmean = mean(ldiffs)
            lstd = std(ldiffs) *
                sqrt(1.0/currbatchsize - (currbatchsize-1.0)/(currbatchsize*(L-1)))

            tstat = (ldiffmean - mu0) / lstd
            tdist = TDist(currbatchsize - 1)

            significancelevel = ccdf(tdist, abs(tstat))
        end

        cumlikelihoods += currbatchsize
        t += 1
        push!(numlikelihoods, cumlikelihoods)

        if ldiffmean > mu0
            theta_t = prop_t
        end
        thetas = [thetas; theta_t']
    end
    thetas, numlikelihoods
end

# This function creates samples via the method outlined in
# Teh and Welling called Stochastic Gradient Langevin Dynamics.
#
# Inputs:
# d                - the target distribution to approximate a sample from
# theta0           - the starting point for the sampler
# batchsize        - the size of the subsamples to evaluate the gradients
# numsweeps        - the number of passes to make over the entire dataset
# epsilonfunc      - the function generating epsilon as described in the algorithm
#
# Returns:
# (samples, number of gradient evaluations)
function runsgld(d::SteinDistribution,
                 theta0::Array{Float64};
                 batchsize=1,
                 numsweeps=100,
                 epsilonfunc=epsilon_generator())

    L = numdatapoints(d)
    p = length(theta0)
    noise = MvNormal(p, 1)
    t = 0
    batchratio = L / batchsize

    theta_t = theta0
    thetas = Array(Float64, numsweeps * intf(batchratio), p)

    for k = 1:numsweeps
        perm = randperm(L)
        for i = 1:intf(batchratio)
            j = batchsize * (i-1) + 1
            idx = perm[j:(j+batchsize-1)]

            epsilon_t = epsilonfunc(t)
            delta = (epsilon_t / 2) .* gradlogdensity(d, theta_t; idx=idx)
            delta += sqrt(epsilon_t) .* vec(rand(noise, 1))
            theta_t += delta
            t += 1
            thetas[t,:] = theta_t
        end
    end

    numgradients = numsweeps * intf(batchratio) * batchsize;
    thetas, numgradients;
end

# This function creates samples via Stochastic Gradient Fisher Scoring
# described in Korattikara, Welling and Ahn.
#
# Inputs:
# d                - the target distribution to approximate a sample from
# theta0           - the starting point for the sampler
# batchsize        - the size of the subsamples to evaluate the gradients
# numsweeps        - the number of passes to make over the entire dataset
# epsilon          - the parameter controlling the stepsizes and variance of noise
# kappa            - the function yielding kappa_t as described in the algorithm
#
# Returns:
# (samples, number of gradient evaluations)
function runsgfs(d::SteinDistribution,
                 theta0::Array{Float64};
                 batchsize=30,
                 numsweeps=100,
                 epsilon=1e-3,
                 kappa=(t -> 1/t))
    L = numdatapoints(d)
    p = length(theta0)
    gamma = (batchsize + L) / batchsize

    t = 1
    It = zeros(p, p)
    batchratio = L / batchsize

    theta_t = theta0;
    thetas = Array(Float64, numsweeps * intf(batchratio), p);

    for k = 1:numsweeps
        perm = randperm(L)
        for i = 1:intf(batchratio)
            j = batchsize * (i-1) + 1;
            idx = perm[j:(j+batchsize-1)]

            g = [gradloglikelihood(d, theta_t; idx=yidx)'
                 for yidx = idx]
            g = vcat(g...)
            V = cov(g)

            It = (1 - kappa(t)) .* It + kappa(t) .* V
            B = (gamma * L) .* It

            noise = MvNormal(B)
            eta = (2/sqrt(epsilon)) .* vec(rand(noise, 1))
            delta =
                2 .* inv(gamma * L * It + (4/epsilon) * B) *
                (gradlogdensity(d, theta_t; idx=idx) + eta)

            theta_t += delta
            thetas[t,:] = theta_t
            t += 1
        end
    end

    numgradients = numsweeps * intf(batchratio) * batchsize
    thetas, numgradients
end


# This function creates samples via Metropolis Adjusted Langevin Algorithm.
#
# Inputs:
# d                - the target distribution to approximate a sample from
# theta0           - the starting point for the sampler
# numiter          - the number of samples to create
# epsilonfunc      - the function computing epsilon, which controls the
#                    variance of the noise and step size
# Returns:
# (samples, number of gradient evaluations)
function runmala(d::SteinDistribution,
                 theta0::Array{Float64};
                 numiter=100,
                 epsilonfunc=epsilon_generator())

    L = numdatapoints(d)
    p = length(theta0)
    noise = MvNormal(p, 1)
    thetas = Array(Float64, numiter, p)
    theta_t = theta0

    for t = 1:numiter
        epsilon_t = epsilonfunc(t-1);
        delta = epsilon_t .* gradlogdensity(d, theta_t) ./ 2
        delta += sqrt(epsilon_t) .* vec(rand(noise, 1))

        prop_t = theta_t + delta
        logdensitydiff = logdensity(d, prop_t) - logdensity(d, theta_t)
        if (log(rand()) < logdensitydiff)
            theta_t = prop_t
        end
        thetas[t,:] = theta_t
    end
    thetas, numiter * L
end
