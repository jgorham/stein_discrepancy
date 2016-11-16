using SteinDistributions

# Computes graph Stein discrepancy based on the Langevin operator.
# Args:
#  sample - SteinDiscrete object representing a sample
#  target - SteinDistribution object representing target distribution
#  solver - optimization program solver supported by JuMP
function langevin_graph_discrepancy(sample::SteinDiscrete,
                                    target::SteinDistribution;
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
    # Objective coefficients for each g(x_i)
    gradlogdensities = gradlogdensity(target, points)
    gobjcoefficients = broadcast(*, gradlogdensities, weights)
    # Objective coefficients for each grad g(x_i)
    gradgobjcoefficients = zeros(n, d, d)
    for j = 1:d
        gradgobjcoefficients[:,j,j] = weights
    end
    # solve the optimization problem
    result = affine_graph_discrepancy(sample,
                                      target,
                                      gobjcoefficients,
                                      gradgobjcoefficients,
                                      solver,
                                      "langevin")
    result
end
