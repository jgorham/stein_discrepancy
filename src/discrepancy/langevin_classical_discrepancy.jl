using SteinDistributions

# Computes classical Stein discrepancy based on the Langevin operator
# for univariate target distributions.
# Args:
#  sample - SteinDiscrete object representing a sample
#  target - SteinDistribution object representing target distribution
#  solver - optimization program solver supported by JuMP that can solve
#   a quadratically constrained quadratic program (QCQP)
function langevin_classical_discrepancy(sample::SteinDiscrete,
                                        target::SteinDistribution;
                                        solver=nothing)
    # make sure solver is defined
    if isa(solver, AbstractString)
        solver = getsolver(solver)
    end
    # we require that the solver is Gurobi or else the quadratic expressions
    # don't work
    try
        eval(Expr(:import,:Gurobi))
        isa(solver, Main.Gurobi.GurobiSolver) ||
            error("The solver used must be Gurobi for classical discrepancies.")
    catch
        error("Gurobi must be installed to solve the classical discrepancy.")
    end
    # get primary objects
    points = sample.support
    weights = vec(sample.weights)
    n = length(weights)
    d = size(points,2)
    # check if d is 1
    (d == 1) || error("The classical discrepancy only works for univariate distributions.")
    # Objective coefficients for each g(x_i)
    gradlogdensities = gradlogdensity(target, points)
    gobjcoefficients = vec(weights .* gradlogdensities)
    gradgobjcoefficients = weights
    # solve the optimization problem
    result = affine_classical_discrepancy(sample,
                                          target,
                                          gobjcoefficients,
                                          gradgobjcoefficients,
                                          solver,
                                          "langevin")
    result
end
