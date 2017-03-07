using SteinDistributions: SteinDistribution

# Computes Stein discrepancy between a weighted sample and a target distribution
#
# Args:
#   points - n x d array of sample points
#   weights - n x 1 array of real-valued weights associated with sample points
#     (default: equal weights)
#   target - SteinDistribution representing target probability distribution
#   operator - string in {"langevin", "riemannian-langevin"} indicating the Stein operator
#     to use (default: "langevin")
#   method - One of {"graph", "classical", "kernel"}, this uses different
#     methods in order to construct the different discrepancies.

function stein_discrepancy(; points=[],
                           weights=fill(1/size(points,1), size(points,1)),
                           target=nothing,
                           method="graph",
                           operator="langevin",
                           kwargs...)

    # Check arguments
    isempty(points) && error("Must provide non-empty array of sample points")
    isa(target, SteinDistribution) ||
        error("Must specify target of type SteinDistribution")

    if method in ["graph", "classical"]
        # Form weighted sample object
        sample = SteinDiscrete(points, weights);
        # call appropriate graph discrepancy
        if method == "graph"
            if operator == "langevin"
                langevin_graph_discrepancy(sample, target; kwargs...)
            elseif operator == "riemannian-langevin"
                riemannian_langevin_graph_discrepancy(sample, target; kwargs...)
            else
                error("unrecognized operator: $(operator)")
            end
        elseif method == "classical"
            if operator == "langevin"
                langevin_classical_discrepancy(sample, target; kwargs...)
            else
                error("unrecognized operator: $(operator)")
            end
        end
    elseif method == "kernel"
        if operator == "langevin"
            langevin_kernel_discrepancy(points, weights, target;
                                        kwargs...)
        else
            error("unrecognized operator: $(operator)")
        end
    else
        error("unrecognized method: $(method)")
    end
end
