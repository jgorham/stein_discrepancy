# Functionality for computing Stein discrepancy
module SteinDiscrepancy

export
# Main method for computing Stein discrepancy
stein_discrepancy,
# Result objects returned by stein_discrepancy
LangevinDiscrepancyResult,
# Function for computing spanner edges
getspanneregdes,
# Function computing univariate Wasserstein distance
wasserstein1d,
# method for extending a langevin graph result
extend_langevin_graph_result,
# method for extending a langevin classical result
extend_langevin_classical_result


# Include files defining methods
include("spanner.jl")
include("LangevinDiscrepancyResult.jl")
include("langevin_graph_discrepancy.jl")
include("langevin_classical_discrepancy.jl")
include("stein_discrepancy.jl")
include("wasserstein.jl")
include("extend_langevin_graph_result.jl")
include("extend_langevin_classical_result.jl")

end # end module
