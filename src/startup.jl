# Include this startup file prior to running Julia code

# Add project module locations to path
push!(LOAD_PATH, abspath(joinpath("src","distributions")))
push!(LOAD_PATH, abspath(joinpath("src","samplers")))
