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
    round(Int, floor(x))
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
function runapproxmh(d::SteinPosterior,
                     theta0::Array{Float64};
                     numiter=Inf,
                     numlikelihood=Inf,
                     batchsize=10,
                     epsilon=0.1,
                     proposalvariance=1e-2)

    if isinf(numiter) && isinf(numlikelihood)
        error("Need to specify numlikelihood or numiter")
    end

    p = length(theta0)
    t = 0
    noise = MvNormal(p, 1)
    thetas = Array(Float64, 0, p)
    theta_t = theta0
    numlikelihoods = Int64[]
    cumlikelihoods = 0

    while t < numiter && cumlikelihoods < numlikelihood
        delta = sqrt(proposalvariance) .* vec(rand(noise, 1))
        prop_t = theta_t + delta
        u = rand()

        (acceptprop, currbatchsize) =
            _speedyproposaltest(d, prop_t, theta_t, u, epsilon, batchsize)

        t += 1
        cumlikelihoods += currbatchsize
        push!(numlikelihoods, cumlikelihoods)

        if acceptprop
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
function runsgld(d::SteinPosterior,
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

    numgradients = numsweeps * intf(batchratio) * batchsize
    thetas, numgradients
end

# This function creates samples via a variant of SGLD that uses the
# Riemannian-Langevin operator as opposed to the Langevin one.
# The update is of the form
#
# Q(.|x) = N(x + eps * [a(x) grad log p(x) + <grad, a^T>(x)], eps a(x))
#
# Inputs:
# d                          - the target distribution to approximate a sample from
# theta0                     - the starting point for the sampler
# volatility_covariance      - the matrix-valued function a = (1/2) sigma sigma^T
# grad_volatility_covariance - the vector-valued function grad_a = <grad, a^t>
# batchsize                  - the size of the subsamples to evaluate the gradients
# numsweeps                  - the number of passes to make over the entire dataset
# epsilonfunc                - the function generating epsilon as described in the algorithm
# Returns:
# (samples, number of gradient evaluations)
function runsgrld(d::SteinPosterior,
                  theta0::Array{Float64,1},
                  volatility_covariance::Function,
                  grad_volatility_covariance::Function;
                  batchsize=1,
                  numsweeps=100,
                  epsilonfunc=epsilon_generator())
    L = numdatapoints(d)
    p = length(theta0)
    t = 0
    batchratio = L / batchsize

    theta_t = theta0
    thetas = Array(Float64, numsweeps * intf(batchratio), p)

    for k = 1:numsweeps
        perm = randperm(L)
        for i = 1:intf(batchratio)
            j = batchsize * (i-1) + 1
            idx = perm[j:(j+batchsize-1)]
            # get the mass matrix ready
            epsilon_t = epsilonfunc(t)
            a = volatility_covariance(theta_t)
            grad_a = grad_volatility_covariance(theta_t)
            # compute change!
            delta = a * gradlogdensity(d, theta_t; idx=idx) + grad_a
            delta *= epsilon_t
            # next compute the variance term
            noise = MvNormal(a)
            delta += sqrt(epsilon_t) * vec(rand(noise, 1))
            theta_t += delta
            t += 1
            thetas[t,:] = theta_t
        end
    end

    numgradients = numsweeps * intf(batchratio) * batchsize
    thetas, numgradients
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
# diagonal         - if true, run SGFS-d (see the paper)
# kappa            - the function yielding kappa_t as described in the algorithm
#
# Returns:
# (samples, number of gradient evaluations)
function runsgfs(d::SteinPosterior,
                 theta0::Array{Float64};
                 batchsize=100,
                 numsweeps=100,
                 epsilon=1e-3,
                 diagonal=false,
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

            g = Array(Float64, 0, p)
            for yidx in idx
                g = vcat(g, gradloglikelihood(d, theta_t; idx=yidx))
            end
            V = cov(g)

            It = (1 - kappa(t)) .* It + kappa(t) .* V
            B = gamma * L .* It

            noise = MvNormal(B)
            eta = (2/sqrt(epsilon)) .* vec(rand(noise, 1))
            if diagonal
                Dvec = gamma * L * diag(It) + (4/epsilon) * diag(B)
                Dinv = diagm(1 ./ Dvec)
            else
                Dinv = inv(gamma * L * It + (4/epsilon) * B)
            end

            delta =
                2 .* Dinv *
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
# d           - the target distribution to approximate a sample from
# theta0      - the starting point for the sampler
# numiter     - the number of samples to create
# epsilonfunc - the function computing epsilon, which controls the
#               variance of the noise and step size
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

# This function creates samples via Metropolis Adjusted Langevin Algorithm
# on a manifold. This can be thought of as a discretization of Riemannian
# Langevin diffusions. The update is of the form
#
# Q(.|x) = N(x + eps * [a(x) grad log p(x) + <grad, a^T>(x)], eps a(x))
#
# Inputs:
# d                          - the target distribution to approximate a sample from
# theta0                     - the starting point for the sampler
# volatility_covariance      - the matrix-valued function a = (1/2) sigma sigma^T
# grad_volatility_covariance - the vector-valued function grad_a = <grad, a^t>
# numiter                    - the number of samples to create
# epsilonfunc                - the function computing epsilon, which controls the
#                              variance of the noise and step size
# Returns:
# (samples, number of gradient evaluations, acceptance ratio)
function runmarla(d::SteinDistribution,
                  theta0::Array{Float64,1},
                  volatility_covariance::Function,
                  grad_volatility_covariance::Function;
                  numiter::Int=100,
                  epsilonfunc::Function=epsilon_generator(),
                  verbose::Bool=false)

    L = numdatapoints(d)
    p = length(theta0)
    thetas = Array(Float64, numiter, p)
    theta_t = theta0
    numaccepts = 0

    for t = 1:numiter
        epsilon_t = epsilonfunc(t-1)::Float64
        a = volatility_covariance(theta_t)::Array{Float64,2}
        grad_a = grad_volatility_covariance(theta_t)::Array{Float64,1}
        # first compute the drift part
        delta = a * gradlogdensity(d, theta_t)::Array{Float64,1} + grad_a
        delta *= epsilon_t
        # next compute the variance term
        noise = MvNormal(a)
        delta += sqrt(epsilon_t) * vec(rand(noise, 1))
        # then define the new proposal
        prop_t = theta_t + delta
        logdensitydiff = logdensity(d, prop_t) - logdensity(d, theta_t)
        if (log(rand()) < logdensitydiff)
            theta_t = prop_t
            numaccepts += 1
        end
        thetas[t,:] = theta_t

        if verbose && (t % 10000) == 0
            println("Finished iteration $t. Continuing.")
        end
    end
    acceptance_ratio = (numaccepts / numiter)
    thetas, 2 * numiter * L, acceptance_ratio
end


# This function creates samples via the approx slice sampler:
#
# http://jmlr.org/proceedings/papers/v33/dubois14.pdf
#
# Inputs:
# d                - the target distribution to approximate a sample from
# theta0           - the starting point for the sampler
# epsilon - the threshold for the sequential test to stop
# numiter - the number of points to sample before halting
# numlikelihood - the maximum number of likelihoods to use before halting
# batchsize - the size of each batch included during the sequential test
# w - the initial length of the line between L and R
# Vmax - the max number of times the initial [L, R] will be extended
# Returns:
# (samples, number of gradient evaluations)
function runapproxslice(d::SteinDistribution,
                        theta0::Array{Float64,1};
                        epsilon=0.1,
                        numiter=Inf,
                        numlikelihood=Inf,
                        batchsize=30,
                        w=1.0,
                        Vmax=100)
    if isinf(numiter) && isinf(numlikelihood)
        error("Need to specify numlikelihood or numiter")
    end

    p = length(theta0)
    thetas = Array(Float64, 0, p)
    theta_t = theta0
    numlikelihoods = Int64[]
    cumlikelihoods = 0
    t = 0

    while t < numiter && cumlikelihoods < numlikelihood
        u, wbar, vbar = [rand() for _ in 1:3]
        # pick some direction
        j = round(Int, ceil(p * rand()))
        z = zeros(p); z[j] = 1
        # now initialize the boundaries: L and R
        L = theta_t - w * wbar * z
        R = L + w * z
        # set VL and VR, needed for detailed balance
        VL = intf(Vmax * vbar)
        VR = Vmax - 1 - VL
        # search for the slice
        onslice = true
        while VL > 0 && onslice
            (onslice, likelihoods)  =
                _speedyproposaltest(d, L, theta_t, u, epsilon, batchsize)
            cumlikelihoods += likelihoods
            VL -= 1
            L -= (w * z)
        end
        onslice = true
        while VR > 0 && onslice
            (onslice, likelihoods)  =
                _speedyproposaltest(d, R, theta_t, u, epsilon, batchsize)
            cumlikelihoods += likelihoods
            VR -= 1
            R += (w * z)
        end
        # now pick the next candidate!
        while true
            eta = rand()
            thetaprop = L + eta * (R - L)
            (onslice, likelihoods) =
                _speedyproposaltest(d, thetaprop, theta_t, u, epsilon, batchsize)
            cumlikelihoods += likelihoods
            if onslice
                theta_t = thetaprop
                break
            end
            if dot(thetaprop, z) < dot(theta_t, z)
                L = thetaprop
            else
                R = thetaprop
            end
        end
        t += 1
        thetas = [thetas; theta_t']
        push!(numlikelihoods, cumlikelihoods)
    end

    thetas, numlikelihoods
end


# This does the sequential test to that runs until the new proposal
# (thetaprop) is significantly different than the acceptance threshold.  The
# sequential test to see if the probability P(thetaptop) / P(theta) > u,
# where u is drawn from Unif([0,1]).
#
# @params
# @d         - the distribution to be sampled from
# @thetaprop - the theta proposed to be on the slice
# @theta     - the current theta defining the slice
# @u         - the Unif([0,1]) that is the height in the sampler
# @epsilon   - the significance level of the test to cut short
#              the complete computation. If 0.0, this is exact.
# @batchsize - the number of samples to add at each test
# @returns: (is ratio larger, number of likelihood evals)
function _speedyproposaltest(d::SteinPosterior,
                             thetaprop::Array{Float64,1},
                             theta::Array{Float64,1},
                             u::Float64,
                             epsilon::Float64,
                             batchsize::Int)
    N = numdatapoints(d)
    logu = log(u)
    lprior = logprior(d, thetaprop) - logprior(d, theta)
    mu0 = (logu - lprior) / N

    significancelevel = 1.0
    currbatchsize = 0
    ldiffmean = 0.0
    perm = randperm(N)

    while (currbatchsize < N) && (significancelevel > epsilon)
        currbatchsize = min(currbatchsize + batchsize, N)
        ldiffs = Float64[]
        for yidx in perm[1:currbatchsize]
            push!(
                ldiffs, (
                    loglikelihood(d, thetaprop; idx=yidx) -
                    loglikelihood(d, theta; idx=yidx)
                )
            )
        end
        ldiffmean = mean(ldiffs)
        lstd = std(ldiffs) * sqrt(1.0/currbatchsize - 1.0/N)

        tstat = (ldiffmean - mu0) / lstd
        tdist = TDist(currbatchsize - 1)

        significancelevel = ccdf(tdist, abs(tstat))
    end

    ldiffmean > mu0, currbatchsize
end
