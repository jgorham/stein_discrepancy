# Bayesian Huber regression posterior abstract type
# The Huber loss is of the form
#
# log p(x) \prop (-1/2)*x^2 I{|x|<=delta} - delta*(|x| - delta/2) I{|x|>delta}

abstract SteinHuberLossRegressionPosterior <: SteinPosterior

function getnormalizingconstant(delta::Float64, p::Int)
    # volume of unit ball in p dimensions
    unitball = pi^(p/2) / gamma(p/2 + 1)
    (univariate_int, error) = quadgk(
        r -> r^(p-1) * exp(
            ifelse(abs(r) <= delta, -0.5 * r^2, delta * (delta / 2.0 - abs(r)))
        ),
        0,
        Inf
    )
    p * unitball * univariate_int
end

# each row of betas is assumed to be a beta sample
function loglikelihood(d::SteinHuberLossRegressionPosterior,
                       beta::Array{Float64,1};
                       idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    delta = d.delta
    p = length(beta)
    n = size(X, 1)
    C = getnormalizingconstant(delta, 1)

    residuals = y .- (X * beta)
    loghuber = zeros(n)
    for ii in 1:n
        residual = residuals[ii]
        if abs(residual) <= delta
            loghuber[ii] = -0.5 * residual^2
        else
            loghuber[ii] = -delta * (abs(residual) - delta / 2.0)
        end
    end
    -log(C) + sum(loghuber)
end

function gradloglikelihood(d::SteinHuberLossRegressionPosterior,
                           beta::Array{Float64,1};
                           idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    delta = d.delta
    p = length(beta)
    n = size(X, 1)

    residuals = y .- (X * beta)
    pregradients = zeros(n)
    for ii in 1:n
        residual = residuals[ii]
        if abs(residual) <= delta
            pregradients[ii] = residual
        else
            pregradients[ii] = delta * sign(residual)
        end
    end
    gradients = X' * pregradients
    vec(gradients)
end

