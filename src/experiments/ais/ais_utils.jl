using Optim

using SteinDistributions: SteinDistribution, gradlogdensity, logdensity, numdimensions

function getapproxmode(target::SteinDistribution;
                       x0::Array{Float64,1}=zeros(numdimensions(target)))
    function objective(x::Vector)
        -logdensity(target, x)
    end
    function gradient!(x::Vector, gradientres::Vector)
        gradient = gradlogdensity(target, x)
        for ii in 1:length(gradient)
            gradientres[ii] = -gradient[ii]
        end
    end
    res = optimize(objective,
                   gradient!,
                   x0,
                   method = LBFGS())
    res.minimum
end
