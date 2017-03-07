# multimodal_gmm_langevin_floor
#
# Considers the stein discrepancy for a 2 mixture GMM with gap between modes
# gap > 0. It compares a bad distribution [which only samples from one mode]
# and the actual iid sequence.

include("experiment_utils.jl")

# this controls the seed used for the approximating the wasserstein metric
@parseintcli seed "s" "seed" 7
# the gap between the two modes
@parseintcli gap "g" "gap" 3
# should we use multiple cores?
@parseintcli numcores "n" "numcores" 1
# the dimension of the space
@parseintcli d "d" "dimension" 1
# the gap between the two modes
@parsestringcli discrepancytype "t" "discrepancytype" "kernel"
# the gap between the two modes
@parsestringcli sampler "q" "sampler" "iid"
# parallelize it
if (numcores > 1)
    if (numcores > CPU_CORES)
        error(@sprintf("Requested %d cores, only %d available.", numcores, CPU_CORES))
    end
    addprocs(numcores - 1)
    @assert (numcores == nprocs())
end

using Distributions: MvNormal

using SteinDistributions: SteinGMM
using SteinKernels: SteinInverseMultiquadricKernel
using SteinDiscrepancy: stein_discrepancy, wasserstein1d

# select largest n
maxn = 30000
# Select an optimization problem solver
if discrepancytype == "graph"
#    solver = "clp"
    import Gurobi
    solver = Gurobi.GurobiSolver(Threads=numcores)
end
# the number of components
K = 2
# distribution sampled from
distname = "$K-gmm"
# target distribution
gmmmus = gap .* [-0.5, 0.5]
gmmmus = hcat(gmmmus, zeros(2, d-1))
gmmsigmas = eye(d)
gmmweights = [0.5, 0.5]
target = SteinGMM(gmmmus, gmmsigmas)
# Sample sizes at which optimization problem will be solved
ns = vcat([5],
          10:10:min(100,maxn),
          200:100:min(1000,maxn),
          2000:1000:min(10000,maxn),
          20000:10000:min(100000,maxn))

if sampler == "iid"
    X = @setseed rand(target, maxn)
elseif sampler == "unimodal"
    wrongtarget = MvNormal(vec(gmmmus[1,:]), gmmsigmas)
    X = @setseed rand(wrongtarget, maxn)'
end

println("Beginning optimization!")
for i in ns
    # initialize all the variables
    objectivevalue = nothing
    edgetime = nothing
    solvetime = nothing
    points = X[1:i,:]
    g = nothing
    gradg = nothing
    operatorg = nothing
    if discrepancytype == "graph"
        res = stein_discrepancy(points=X[1:i,:],
                                target=target,
                                solver=solver)
        objectivevalue = res.objectivevalue
        edgetime = res.edgetime
        solvetime = res.solvetime
        points = res.points
        g = res.g
        gradg = res.gradg
        operatorg = res.operatorg
    elseif discrepancytype == "kernel"
        kernel = SteinInverseMultiquadricKernel()
        res = stein_discrepancy(points=X[1:i,:],
                                target=target,
                                method="kernel",
                                kernel=kernel)
        objectivevalue = sqrt(res.discrepancy2)
        solvetime = res.solvetime
    elseif discrepancytype == "wasserstein"
        @assert (d == 1)
        (objectivevalue, error) = wasserstein1d(X[1:i,:], target=target)
    end

    println("\tn = $(i), objective = $(objectivevalue)")
    # Package and save results
    instance_data = Dict{Any, Any}(
        "seed" => seed,
        "mus" => gmmmus,
        "sigmas" => gmmsigmas,
        "distweights" => gmmweights,
        "sampler" => sampler,
        "gap" => gap,
        "distname" => distname,
        "X" => points,
        "g" => g,
        "gradg" => gradg,
        "operatorg" => operatorg,
        "n" => i,
        "d" => d,
        "objectivevalue" => objectivevalue,
        "discrepancytype" => discrepancytype,
        "edgetime" => edgetime,
        "solvetime" => solvetime,
        "ncores" => nprocs(),
    )

    save_json(
        instance_data;
        dir="multimodal_gmm_langevin_floor",
        distname=distname,
        sampler=sampler,
        discrepancytype=discrepancytype,
        n=i,
        d=d,
        gap=gap,
        seed=seed,
        numcores=nprocs(),
    )
end

println("COMPLETE!")
