# Kernels for mean-zero RKHS
module SteinKernels

export
SteinKernel,
SteinParzenARKernel,
SteinChampionLenardMillsKernel,
SteinGaussianWeightedKernel,
SteinGaussianRectangularDomainKernel,
SteinGaussianPowerKernel,
SteinGaussianUnboundedPowerKernel,
SteinGaussianKernel,
SteinMaternTensorizedKernel,
SteinMaternRadialKernel,
SteinPolyharmonicSplineKernel,
SteinInverseMultiquadricKernel,
k,
gradxk,
gradyk,
gradxyk,
k0

include("SteinKernel.jl")
include("SteinTensorizedKernel.jl")
include("SteinGaussianWeightedKernel.jl")
include("SteinGaussianRectangularDomainKernel.jl")
include("SteinGaussianPowerKernel.jl")
include("SteinGaussianUnboundedPowerKernel.jl")
include("SteinGaussianKernel.jl")
include("SteinParzenARKernel.jl")
include("SteinChampionLenardMillsKernel.jl")
include("SteinPolyharmonicSplineKernel.jl")
include("SteinMaternTensorizedKernel.jl")
include("SteinMaternWeightedKernel.jl")
include("SteinMaternRadialKernel.jl")
include("SteinMaternPowerKernel.jl")
include("SteinMaternUnboundedPowerKernel.jl")
include("SteinInverseMultiquadricKernel.jl")

end # end module
