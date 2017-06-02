# Stein discrepancy experiments

This repo contains many experiments that utilize the *Stein discrepancy*.  The
experiments here use the Julia package
[SteinDiscrepancy.jl](https://github.com/jgorham/SteinDiscrepancy.jl) and
implement all the experiments from the following papers:

[Measuring Sample Quality with Stein's Method](http://arxiv.org/abs/1506.03039),
[Measuring Sample Quality with Diffusions](https://arxiv.org/abs/1611.06972),
[Measuring Sample Quality with Kernels](https://arxiv.org/abs/1703.01717).

All the papers above introduce and explain the significance of the Stein
discrepancy.  The first paper has a more gentle introduction describing how
the Stein discrepancy bounds standard probability metrics like the
[Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).

## So how do I run these experiments?

The code has all been tested on Julia v0.5. There is a file `src/startup.jl`
which simply adds the paths of some modules to the `LOAD_PATH` variable.
You should either simlink this file to `~/.juliarc.jl` or add the lines
from this file to your current `.juliarc.jl`.

There is a REQUIRE file that demarcates all the necessary packages needed
to run all the experiments. All experiments should be run from
the base directory of this repo. For example, the command

```
julia src/experiments/multimodal_gmm_langevin_floor --gap=5 --seed=10
```

would run the experiment in the file
`src/experiments/multimodal_gmm_langevin_floor.jl` and pass in the
parameters gap and seed (defined in the script) to be 5 and 10 respectively.
All experiment results will be deposited in the directory `results`
in the base directory.

For more information, see [SteinDiscrepancy.jl](https://github.com/jgorham/SteinDiscrepancy.jl).