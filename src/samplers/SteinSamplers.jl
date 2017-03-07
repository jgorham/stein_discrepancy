# Methods for drawing samples
module SteinSamplers

export
runapproxmh,
runsgld,
runsgrld,
runsgfs,
runmala,
runmarla,
runapproxslice

# Include sampler implementations
include("samplers.jl")

end # end module
