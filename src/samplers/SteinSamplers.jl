# Methods for drawing samples
module SteinSamplers

export
runapproxmh,
runsgld,
runsgfs,
runmala

# Include sampler implementations
include("samplers.jl")

end # end module
