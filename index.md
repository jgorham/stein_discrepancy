# SteinDiscrepancy.jl

## What is this so-called Stein discrepancy?

To improve the efficiency of Monte Carlo estimation, practitioners are
turning to biased Markov chain Monte Carlo procedures that trade off
asymptotic exactness for computational speed. The reasoning is sound: a
reduction in variance due to more rapid sampling can outweigh the bias
introduced. However, the inexactness creates new challenges for sampler and
parameter selection, since standard measures of sample quality like
effective sample size do not account for asymptotic bias. To address these
challenges, we introduce a new computable quality measure that quantifies
the maximum discrepancy between sample and target expectations over a large
class of test functions. This measure is what we are calling the
Stein discrepancy.

For a more detailed explanation, take a peek at the latest papers:

[Measuring Sample Quality with Diffusions](https://arxiv.org/abs/1611.06972),
[Measuring Sample Quality with Kernels](https://arxiv.org/abs/1703.01717).

These build on previous work from

[Measuring Sample Quality with Stein's Method](http://arxiv.org/abs/1506.03039)

and its companion paper

[Multivariate Stein Factors for a Class of Strongly Log-concave
Distributions](http://arxiv.org/abs/1512.07392).

These latter two papers are a more gentle introduction describing how the
Stein discrepancy bounds standard probability metrics like the
[Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).

## So how do I use it?

This software has been tested on Julia v0.5.0. There are currently two
broads classes of Stein discrepancies: graph and kernel.

### Graph Stein discrepancies

Computing the graph Stein discrepancy requires solving a linear program
(LP), and thus you'll need some kind of LP solver installed to use this
software. We use JuMP ([Julia for Mathematical
Programming](https://jump.readthedocs.org/en/latest/)) to interface with
these solvers; any of the supported JuMP LP solvers with do just fine.

Once you have an LP solver installed, computing our measure is easy.
First, clone this repo [or download it as a tarbell] in a directory of your
choosing. Next, you must have a target distribution in mind.  We represent
each target distribution as a class (specifically, a subclass of a
`SteinDistribution`) and encode all relevant information about the target
(like the gradient of its log density) in that class. Various examples of
target distributions can be found in the `src/distributions`. Feel free to add
your own!

Once you have this target in hand, the rest is easy. Here's a quick example
that you can run from the base directory (the parent directory of
src). After you fire up julia from the command line, the following commands
will compute the Langevin graph Stein discrepancy for a bivariate uniform sample:

```
# set up source paths, and compile C++ code
include("src/startup.jl")
# do the necessary imports
using SteinDistributions: SteinUniform
using SteinDiscrepancy: stein_discrepancy
# creates a uniform distribution on [0,1]^2
target = SteinUniform(2)
# generates 100 points
X = rand(target, 100)
# can be a string or a JuMP solver
solver = "clp"
result = stein_discrepancy(points=X, target=target, solver=solver, method="graph")
discrepancy = vec(result.objective_value)
```

The variable `discrepancy` here will encode the Stein discrepancy along each
dimension. The final discrepancy is just the sum of this vector.

### Kernel Stein discrepancies

Computing the kernel Stein discrepancies does not require a LP solver. In
lieu of a solver, you will need to specify a kernel function. Many common
kernels are already implemented in `src/kernels`, but if yours is not there
for some reason, feel free to inherit from the `SteinKernel` type and roll
your own.

With a kernel in hand, computing the kernel Stein discrepancy is easy:

```
# set up source paths if you havent already
include("src/startup.jl")
# do the necessary imports
using SteinDistributions: SteinGMM
using SteinKernels: SteinInverseMultiquadricKernel
using SteinDiscrepancy: stein_discrepancy
# setup up target distribution
target = SteinGMM([-0.5, 0.5])
# grab sample
X = rand(target, 500)
# create the kernel instance
kernel = SteinInverseMultiquadricKernel()
# compute the KSD2
result = stein_discrepancy(points=X, target=target, method="kernel", kernel=kernel)
# get the final ksd
ksd = sqrt(res.discrepancy2)
```

## Summary of the Code

All code is available in the src directory of the repo. Many examples
computing the stein_discrepancy are in the src/experiments directory. The
experiment multimodal_gmm_langevin_floor is a good first one to examine. To
see an example that goes beyond the Langevin Stein discrepancy, see the
experiment compare-hyperparameters-gmm-posterior.

### Contents of src

* startup.jl - Make sure all paths are configured correctly
* discrepancy - Code for computing Stein discrepancy
* distributions - Types representing probability distributions
* experiments - Code for running experiments
* samplers - Code implementing the samplers of study
* visualization - R scripts to create visualizations for the results

### Third-party software
The `src/experiments` directory contains many other people's software
to recreate the experiments from the papers. The list includes

* LR_HMC - This code was written by Ahn et al. and was used to recreate the
  MNIST experiment from Bayesian Posterior Sampling via Stochastic Gradient
  Fisher Scoring.
* skh - This code is a copy of Simon Lacoste-Julien and Francis Bach's code
  for sequential kernel herding.
* kernel_goodness_of_fit - This code is from Chwialkowski et al., and is a
  static fork of https://github.com/karlnapf/kernel_goodness_of_fit.
* blackweights - This code is from Qiang Liu and was used to compare kernels
  for the black box importance sampling experiment.

### Conventions

* Use lowercase file names (with underscores if needed for clarity) for scripts
* Use camel case with initial capital letter for defining types and modules
* Use lowercase (with underscores if needed for clarity) for variables and
  functions (unless there are other prevailing conventions like X representing
      a data matrix)

### Compiling Code in discrepancy/spanner directory

Our C++ code should be compiled when startup.jl is first invoked. However,
if this doesn't work for some reason, you can issue the following
commands to compile the code in discrepancy/spanner:

```
cd discrepancy/spanner
make
make clean
```

The last step isn't necessary, but it will remove some superfluous
files. If you want to kill everything made in the build process, just run

```
make distclean
```
