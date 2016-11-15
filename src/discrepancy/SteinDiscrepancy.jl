# Functionality for computing Stein discrepancy
module SteinDiscrepancy

export
# Utility function for getting the solver
getsolver,
# Main method for computing Stein discrepancy
stein_discrepancy,
# Result objects returned by affine stein discrepancy methods
AffineDiscrepancyResult,
# Function for computing spanner edges
getspanneregdes,
# Function computing univariate Wasserstein distancen
wasserstein1d,
# Function computing Wasserstein distance for discrete distributions
wassersteindiscrete,
# Function approximating Wasserstein distance for multivariate Wasserstein
approxwasserstein

# Include files defining methods
include("getsolver.jl")
# Spanner files and utilities
include("spanner.jl")
include("l1spanner.jl")
include("mincostflow.jl")
# graph related discrepancies
include("AffineDiscrepancyResult.jl")
include("affine_graph_discrepancy.jl")
include("langevin_graph_discrepancy.jl")
include("riemannian_langevin_graph_discrepancy.jl")
# classical related discrepancies
include("affine_classical_discrepancy.jl")
include("langevin_classical_discrepancy.jl")
# utilities / addons
include("wasserstein.jl")
# the main ONE
include("stein_discrepancy.jl")
end # end module
