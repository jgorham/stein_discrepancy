# Include this startup file prior to running Julia code

# Add project module locations to path
push!(LOAD_PATH, abspath(joinpath("src","bounds")))
push!(LOAD_PATH, abspath(joinpath("src","discrepancy")))
push!(LOAD_PATH, abspath(joinpath("src","distributions")))
push!(LOAD_PATH, abspath(joinpath("src","pseudosamplers")))
push!(LOAD_PATH, abspath(joinpath("src","kernels")))
push!(LOAD_PATH, abspath(joinpath("src","control_variates")))
push!(LOAD_PATH, abspath(joinpath("src","samplers")))
# Add directory that will house compiled spanner code to path
spannerpath = abspath(joinpath("src", "discrepancy", "spanner"))
push!(Libdl.DL_LOAD_PATH, spannerpath)
# Compile spanner code if not already compiled
if !isfile(joinpath(spannerpath,"libstein_spanner.so"))
    run(`make -C $spannerpath`)
end
function loadexp()
    cd("src/experiments")
    include("experiment_utils.jl")
    cd("../..")
end
