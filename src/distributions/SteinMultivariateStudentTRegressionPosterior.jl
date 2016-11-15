# Bayesian multivariate student-t regression posterior abstract type

abstract SteinMultivariateStudentTRegressionPosterior <: SteinPosterior

# each row of betas is assumed to be a beta sample
function loglikelihood(d::SteinMultivariateStudentTRegressionPosterior,
                       beta::Array{Float64,1};
                       idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    nu = d.nu
    # Extract the number of observations
    m = size(X, 1)
    mus = y .- (X * beta)
    sigmainv_mus = \(d.sigma[idx,idx], mus)
    tpart = (mus' * sigmainv_mus)[1]

    logcons = lgamma(0.5 * (m + nu)) - lgamma(nu / 2) - (m/2)*log(nu*pi) -
        0.5 * logdet(d.sigma[idx,idx])
    logtterms = -0.5 * (m + nu) * log(1 + tpart / nu)

    logcons + logtterms
end

function gradloglikelihood(d::SteinMultivariateStudentTRegressionPosterior,
                           beta::Array{Float64,1};
                           idx=1:length(d.y))
    X = d.X[idx,:]
    y = d.y[idx]
    nu = d.nu
    # Extract the number of observations
    m = size(X, 1)
    # mus is a N x b matrix
    mus = y .- (X * beta)
    sigmainv_mus = \(d.sigma[idx,idx], mus)
    tpart = (mus' * sigmainv_mus)[1]

    tdistpart = 0.5 * (1 + m / nu) / (1 + tpart / nu)
    gradtpart = 2 * X' * sigmainv_mus
    # each column of gradients is the gradient for a given beta
    gradients = tdistpart .* gradtpart
    vec(gradients)
end
