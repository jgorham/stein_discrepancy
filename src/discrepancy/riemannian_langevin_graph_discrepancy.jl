using SteinDistributions

# Computes graph Stein discrepancy based on the Riemannian-Langevin operator.
# Our parametrization is of the form
#
# dXt = b(Xt) dt + sigma(Xt) dBt
#
# where Xt is the diffusion process and Bt is d-dimensional Brownian Motion.
# Since this is a Riemannian-Langevin diffusion, b and sigma are linked via
# the relationship
#
# a(Xt) = (1/2) sigma(Xt) sigma(Xt)^T,
# grad a(Xt) = <grad, a^T> (e.g. the grad operator applied to each column of M^T),
# b(Xt) = a(Xt) grad log p(Xt) + grad a(Xt).
#
# This ensures that the diffusion has p as its invariant distribution.
#
# In this case, the infinitesmal operator has the form
#
# T g = 2 <g, b> + 2 <a, grad g>.
#
# Args:
#  sample - SteinDiscrete object representing a sample
#  target - SteinDistribution object representing target distribution
#  volatility_covariance - the matrix-valued function a = (1/2) sigma sigma^T
#  grad_volatility_covariance - the vector-valued function grad a = <grad, a^t>
#  solver - optimization program solver supported by JuMP
function riemannian_langevin_graph_discrepancy(sample::SteinDiscrete,
                                               target::SteinDistribution;
                                               volatility_covariance=nothing,
                                               grad_volatility_covariance=nothing,
                                               solver=nothing)
    # make sure solver is defined
    if isa(solver, AbstractString)
        solver = getsolver(solver)
    end
    isa(solver, AbstractMathProgSolver) ||
        error("Must specify solver of type String or AbstractMathProgSolver")
    ## Extract inputs
    points = sample.support
    weights = sample.weights
    n = length(weights)
    d = size(points,2)
    # make sure the volatility matrices are non-nothing
    if isa(volatility_covariance, Void)
        error("For riemannian langevin methods, you must supply the volatility_covariance")
    end
    if isa(grad_volatility_covariance, Void)
        error("For riemannian langevin methods, you must supply the grad_volatility_covariance")
    end
    # Now prepare the coefficients for g and grad g
    gobjcoefficients = Array(Float64, n, d)
    gradgobjcoefficients = Array(Float64, n, d, d)
    gradlogdensities = gradlogdensity(target, points)
    for i in 1:n
        xi = vec(points[i,:])
        wi = weights[i]
        gradlogpxi = vec(gradlogdensities[i,:])
        a = volatility_covariance(xi)
        grad_a = grad_volatility_covariance(xi)
        # Objective coefficients for each g(x_i)
        gobjcoefficients[i,:] = 2 * wi * (a * gradlogpxi + grad_a)
        # Objective coefficients for each grad g(x_i)
        gradgobjcoefficients[i,:,:] = 2 * wi * a
    end
    # solve the optimization problem
    result = affine_graph_discrepancy(sample,
                                      target,
                                      gobjcoefficients,
                                      gradgobjcoefficients,
                                      solver,
                                      "riemannian-langevin")
    result
end
