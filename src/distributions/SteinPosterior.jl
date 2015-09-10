# SteinPosterior
# An abstract subtype of SteinDistribution representing a posterior
# distribution p(x) = pi(x | y_1,...,y_L) induced by a likelihood
# pi(y_1,...,y_L|x) over datapoints y_L and a prior pi(x)
abstract SteinPosterior <: SteinDistribution

# Returns the number of datapoints on which this posterior is conditioned
function numdatapoints(d::SteinPosterior)
    size(d.y, 1)
end

# Returns the mini-batch approximation
# log pi(x) + (L/length(idx)) log pi(x | d.y[idx])
# to the log posterior density
function logdensity(d::SteinPosterior,
                    x::Array{Float64, 1};
                    idx=1:numdatapoints(d))
    batchratio = numdatapoints(d)/length(idx)
    logprior(d,x) + batchratio * loglikelihood(d,x;idx=idx)
end

# For each sample point xs[i,:], returns the mini-batch approximation to the log posterior density
function logdensity(d::SteinPosterior,
                    xs::Array{Float64, 2};
                    idx=1:numdatapoints(d))

    logdensities = [logdensity(d, vec(xs[i,:]); idx=idx)
                    for i=1:size(xs, 1)]
    logdensities
end

# For each sample point xs[i,:] returns log likelihood gradient
# grad_{x} log pi(d.y[idx] | xs[i,:])
# over the batch of datapoints d.y[idx]
function gradloglikelihood(d::SteinPosterior,
                           xs::Array{Float64, 2};
                           idx=1:numdatapoints(d))
    gradloglikelihoods = [
        gradloglikelihood(d, vec(xs[i,:]); idx=idx)'
        for i=1:size(xs, 1)
    ]
    vcat(gradloglikelihoods...)
end

# Returns the mini-batch approximation
# grad_{x} log pi(x) + (L/length(idx)) grad_{x} log pi(x | d.y[idx])
# to the gradient of the log posterior density
function gradlogdensity(d::SteinPosterior,
                        x::Array{Float64, 1};
                        idx=1:numdatapoints(d))
    batchratio = numdatapoints(d)/length(idx)
    gradlogprior(d,x) + batchratio .* gradloglikelihood(d, x; idx=idx)
end

# For each sample point xs[i,:] returns the mini-batch approximation
# to the gradient of the log posterior density
function gradlogdensity(d::SteinPosterior,
                        xs::Array{Float64, 2};
                        idx=1:numdatapoints(d))
    gradlogdensities = [
        gradlogdensity(d, vec(xs[i,:]); idx=idx)'
        for i=1:size(xs, 1)
    ]
    vcat(gradlogdensities...)
end
