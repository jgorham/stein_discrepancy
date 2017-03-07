# compare_coercive_kernel_discrepancies
#
# This experiment highlights the need to properly account
# for the possibility that distributions could diverge
# infinity and look suitable vis-a-vis the KSD.

# parallelize now!
addprocs(CPU_CORES - 1)

using Distributions: Exponential
using Iterators: product

using SteinDistributions:
    SteinGaussian,
    SteinScaleLocationStudentT
using SteinKernels:
    SteinGaussianKernel,
    SteinMaternRadialKernel,
    SteinGaussianPowerKernel,
    SteinInverseMultiquadricKernel
using SteinDiscrepancy:
    langevin_kernel_discrepancy

include("experiment_utils.jl")

# dimension of the problem
@parseintcli d "d" "dimension" 8
# random number generator seed
@parseintcli seed "s" "seed" 7
# set the sample distribution {gaussian, studentt, epspacking, randomepspacking}
@parsestringcli samplesource "Q" "samplesource" "gaussian"
# set the discrepancy
@parsestringcli discrepancytype "k" "discrepancytype" "inversemultiquadric"

# specify the target distribution
target = SteinGaussian(d)
# set up student t distribution
df = 10.0
studentt = SteinScaleLocationStudentT(df, 0.0, sqrt((df-2.0)/df))
# set the kernel
if discrepancytype == "matern"
    kernel = SteinMaternRadialKernel()
elseif discrepancytype == "gaussian"
    kernel = SteinGaussianKernel()
elseif discrepancytype == "gaussianpower"
    kernel = SteinGaussianPowerKernel(2.0)
elseif discrepancytype == "inversemultiquadric"
    kernel = SteinInverseMultiquadricKernel(0.5)
end
# Sample sizes at which optimization problem will be solved
ns = vcat(100:100:1000, 2000:1000:10000, 20000:10000:100000)
# the radii to use for the lattace generation
rs = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03,
      0.02, 0.015, 0.012, 0.01, 0.0085, 0.007]

function packsphere3d(r::Float64)
    # we'll put the lattice in [-1, 1]^3 \cap B(0,1)
    I = ceil(1 / r)
    J = ceil(2 / (r * sqrt(3)))
    K = ceil(3 / (r * sqrt(6)))
    packing = Array(Float64, 0, 3)
    for (i,j,k) in product(0:I, 0:J, 0:K)
        pt = [2*i + mod(j + k, 2),
              sqrt(3)*(mod(k,2)/3.0 + j),
              2*sqrt(6)/3*k]
        pt = pt - ones(3)
        pt = pt .* r
        if norm(pt) <= 1
            packing = vcat(packing, pt')
        end
    end
    packing
end

function packsphererandom(n::Int, d::Int; deltan = n::Int -> 2*log(n))
    packing = Array(Float64, 0, d)
    @assert (d >= 3)

    delta = deltan(n)
    r = delta * n^(1/d)
    ii = 0
    depth = Exponential(1.0)

    while size(packing, 1) < n && (ii <= 100*n)
        ii += 1
        x0 = rand(target, 1)
        x0 = r * x0 / (norm(x0)^2 + rand(depth))^(0.5)
        shouldadd = true
        for jj in 1:size(packing,1)
            if norm(packing[jj:jj,:] - x0) <= delta
                shouldadd = false
                break
            end
        end
        if shouldadd
            packing = vcat(packing, x0)
        end
    end
    if (ii > 100*n)
        error(@sprintf("For n=%d the packing failed.", n))
    end
    packing
end

println("Beginning optimization")
continueloop = true
idx = 0
while continueloop
    idx += 1
    if samplesource == "gaussian"
        ii = ns[idx]
        X = @setseed seed rand(target, ii)
        continueloop = (idx < length(ns))
    elseif samplesource == "studentt"
        ii = ns[idx]
        Xvec = @setseed seed rand(studentt, ii*d)
        X = reshape(Xvec, d, ii)'
        continueloop = (idx < length(ns))
    elseif samplesource == "epspacking"
        @assert (d == 3)
        rr = rs[idx]
        X = packsphere3d(rr)
        ii = size(X,1)
        X = sqrt(ii / log(ii)) .* X
        continueloop = (idx < length(rs))
    elseif samplesource == "randomepspacking"
        ii = ns[idx]
        X = @setseed seed packsphererandom(ii, d)
        continueloop = (idx < length(ns))
    end
    # set the weights
    weights = ones(ii) ./ ii
    kernelresult = langevin_kernel_discrepancy(X,
                                               weights,
                                               target;
                                               kernel=kernel)
    kerneldiscrepancy = sqrt(kernelresult.discrepancy2)
    solvetime = kernelresult.solvetime
    # save data
    instance_data = Dict{Any, Any}(
        "samplesource" => samplesource,
        "discrepancytype" => discrepancytype,
        "n" => ii,
        "d" => d,
        "seed" => seed,
        "steindiscrepancy" => kerneldiscrepancy,
        "solvetime" => solvetime,
        "ncores" => nprocs(),
    )
    save_json(
        instance_data;
        dir="compare_coercive_kernel_discrepancies",
        samplesource=samplesource,
        discrepancytype=discrepancytype,
        n=ii,
        d=d,
        seed=seed,
    )
end
println("COMPLETE!")
