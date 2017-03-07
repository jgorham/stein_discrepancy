Notes about code:

This folder contains the implementation of Frank-Wolfe quadrature 
(FW-Quad Algorithm 1 in the paper) for a mixture of Gaussians, 
called 'kernel herding' here (though it is much more general as can use 
various FW optimization techniques such as line-search step-size, 
fully-corrective FW (FCFW), etc.).

- kernel_herding.m -> this is the main routine used by most experiments
                      [see its help for calling interface]

It has other versions:
- kernel_herding_before_acceleration.m -> a slower version which also 
        compute the objective 0.5*||mu(p) - mu(hat{p)_t||^2
        (see (3) in paper). This takes quadratic time in the 
        number of components of the mixture, and is not needed in SKH, and
        thus this is why it is not used in the UAV or synthetic PF experiment
        (kernel_herding.m skips this piece).
        It is used in the mixture of Gaussian experiment though as we plot
        the MMD error as a function of the number of iterations.

- kernel_herding_discrete.m -> this run FW on a mixed space (r,x), where r
        is a discrete variable which takes a finite number of states. It is
        used in the JMLS experiment. It allows to use a kernel which can
        distinguish different mixture components -- we use:
        k((r,x), (r',x')) = k_r(r,r')*RBF(x,x'); for JMLS, k_r(r,r') is 
        a Kronecker-delta function.

- kernel_herding_stable.m -> implements a more careful version of the FCFW
        algorithm to avoid numerical instabilities issues when the kernel is
        degenerate (this happens for example in the non-linear benchmark; 
        see Section C.2.1). We only needed it for the non-linear benchmark
        for big sigma2. This one also computes the objective in order to 
        allow early stopping (when MMD is almost zero).


== Helper files:

- RBF.m -> RBF kernel function
- mixture_kernel -> mixture of RBF kernel evaluations
- qrandgm -> produce quasi-random Sobol numbers from a mixture of Gaussians
             (used for the QMC implementation)
- safe_field -> to access fields of struct that might not exist (with a default value)
