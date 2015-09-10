# Project Code

## Overview

All code in this folder is written assuming that it will be called from the
repository base directory, the parent directory of this folder.

Please include startup.jl before attempting to run other code.

## Contents

* startup.jl - Adds project module locations to LOAD_PATH
* discrepancy - Code for computing Stein discrepancy
* distributions - Types representing probability distributions
* experiments - Code for running experiments
* samplers - Code implementing the samplers of study
* visualization - R scripts to create visualizations for the results

## Conventions

* Use lowercase file names (with underscores if needed for clarity) for scripts
* Use camel case with initial capital letter for defining types and modules
* Use lowercase (with underscores if needed for clarity) for variables and 
  functions (unless there are other prevailing conventions like X representing
  a data matrix)

## Compiling Code in discrepancy/spanner directory
The code should be compiled when startup.jl is first invoked. However,
if this doesn't work for some reason, here's all you need to know for compiling
the code in discrepancy/spanner:

> cd discrepancy/spanner

> make

> make clean

The last step isn't necessary, but it will remove some superfluous
files. If you want to kill everything made in the build process, just run

> make distclean
