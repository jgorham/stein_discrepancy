# SteinDiscrepancy.jl

## What is this so-called Stein Discrepancy?

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

For a more detailed explanation, take a peek at the original paper,

[Measuring Sample Quality with Stein's Method](http://arxiv.org/abs/1506.03039).

To learn more about how the Stein discrepancy bounds standard probability metrics, 
like the [L1-Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric), see 

[Multivariate Stein Factors for Strongly Log-concave Distributions](http://arxiv.org/abs/1512.07392).

## So how do I use it?

Computing the vanilla Stein discrepancy requires solving a linear program (LP), and
thus you'll need some kind of LP solver installed to use this
software. We use JuMP ([Julia for Mathematical Programming](https://jump.readthedocs.org/en/latest/)) 
to interface with these solvers; any of the supported JuMP LP solvers with do just fine.

Assuming you have an LP solver installed, computing our measure is easy.
First, you must have a target distribution in mind. 
We represent each target distribution as a class (specifically, a
subclass of a `SteinDistribution`) and encode all relevant
information about the target (like the gradient of its log
density) in that class. 
Various examples of target distributions can be found in the
src/distributions.  Feel free to add your own!

Once you have this target in hand, the rest is easy. Here's a quick example
that you can run from the base directory (the parent directory of src):

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
result = stein_discrepancy(points=X, target=target, solver=solver)
discrepancy = vec(result.objective_value)
```

The variable `discrepancy` here will encode the Stein discrepancy along each
dimension. The final discrepancy is just the sum of this vector.

## Summary of the Code

All code is available in the src directory of the repo. Many examples
computing the stein_discrepancy are in the src/experiments directory
(the experiment sample_target_mismatch-multivariate_uniform_vs_beta
is a good first one to examine).

Make sure to include startup.jl so all the paths are properly set up.
To do so, after opening the julia REPL, enter `include("src/startup.jl")`
at the command prompt.

### Contents of src

* startup.jl - Adds project module locations to LOAD_PATH
* discrepancy - Code for computing Stein discrepancy
* distributions - Types representing probability distributions
* experiments - Code for running experiments
* samplers - Code implementing the samplers of study
* visualization - R scripts to create visualizations for the results

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
