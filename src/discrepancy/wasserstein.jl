# Computes the Wasserstein distance between a univariate discrete distribution
# and a univariate distribution with an implemented function 'cdf'.
#
# Args:
#   Q - univariate SteinDiscrete distribution; support of Q must be sorted
#   P - univariate SteinDistribution implementing the function 'cdf'
#
# Returns:
#   Tuple of Wasserstein distance and upper bound on numerical integration
#   error.
function wasserstein1d(Q::SteinDiscrete, P::SteinDistribution)
    X = Q.support;
    if size(X,2) != 1
        error("function only defined by univariate distributions")
    end
    if !issorted(X)
        error("Support of Q must be sorted")
    end
    q = Q.weights;
    n = length(q);

    # Find the extent of the domains of Q and P
    lower = min(X[1], supportlowerbound(P,1));
    upper = max(X[n], supportupperbound(P,1));

    # Compute \int_t |Q(t) - P(t)| dt where Q(t) is the cdf of Q and P(t)
    # is the cdf of P
    # First compute contribution of the interval (lower, x_1)
    (integral, max_error) = quadgk(t->cdf(P,t), lower, X[1]);
    # Cumulative Q weight
    weight = 0;
    for i in 1:(n-1)
        # Compute integral contributed by each (x_i, x_{i+1}] interval
        weight = weight + q[i];
        (estimate, error) = quadgk(t->abs(cdf(P,t) - weight), X[i], X[i+1]);
        integral = integral + estimate;
        max_error = max_error + error;
    end
    # Add contribution of (x_n, upper)
    (estimate, error) = quadgk(t->1-cdf(P,t), X[n], upper);
    integral = integral + estimate;
    max_error = max_error + error;
    (integral, max_error)
end

# Computes the Wasserstein distance between a univariate discrete distribution
# (represented by points and weights) and a univariate distribution with an 
# implemented function 'cdf'.
#
# Args:
#   points - n x 1 array of support points
#   weights - n x 1 array of real-valued weights associated with support points
#   target - univariate SteinDistribution implementing the function 'cdf'
#
# Returns:
#   Tuple of Wasserstein distance and upper bound on numerical integration
#   error.
function wasserstein1d(; points=[], 
                       weights=fill(1/size(points,1), size(points,1)), 
                       target=None)
    # Check arguments
    isempty(points) && error("Must provide non-empty array of support points")

    wasserstein1d(SteinDiscrete(points, weights), target)
end
