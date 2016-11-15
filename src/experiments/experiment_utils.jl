# experiment_utils
#
# A set of utility functions for running experiments.
# These are useful for removing ugly/repetitive code.

using DataFrames
using DataStructures
using JSON

# This is a utility function for parsing commandline arguments to a script.
# The doc uses the following inputs for this explanation: -s7 or --seed=7
# @arguments:
# arg:      the entire commandline argument to be parsed (e.g. -s7 or --seed=7)
# shortopt: the root for short options (in this example its 's')
# longopt:  the root for long option ('seed' in this case)
# @returns: (?string) The matching parsed argument as a string (e.g. '7') else Nothing
function matchcommandlinearg(arg::AbstractString, shortopt::AbstractString, longopt::AbstractString)
    shortmatch = match(Regex("^-$(shortopt)(.*)"), arg)
    if shortmatch != nothing
        return shortmatch.captures[1]
    end
    longmatch = match(Regex("^--$(longopt)=(.*)"), arg)
    if longmatch != nothing
        return longmatch.captures[1]
    end
end

# This is a utility function for saving a Dictionary of mixed types (instancedata)
# as a json in the given project. It constructs the pathname for the saved
# data as follows:
#
# ./results/<dir>/data/<prefix>_kwarg1=kwval1_..._kwargn=kwvaln.json
#
# where kwarg1=kwval1, ..., kwargn=kwvaln are all the keyword arguments passed in
# If any of the directories doesn't exists, it just creates that directory.
#
# CAREFUL! This will always construct the directories for the current dir.
function save_json(instancedata::Dict; prefix="julia", dir="misc", kwargs...)
    # first make sure the directory exists
    relpath_dir = joinpath(".", "results", dir, "data")
    if !isdir(relpath_dir)
        mkpath(relpath_dir)
    end
    # now build up the filename
    running_filename = prefix
    for (chunkkey, chunkvalue) in kwargs
        chunk = string(chunkkey, "=", chunkvalue)
        running_filename = string(running_filename, "_", chunk)
    end
    json_filename = string(running_filename, ".json")
    # and lets write it
    relpath_json_filename = joinpath(relpath_dir, json_filename);
    fh = open(relpath_json_filename, "w");
    write(fh, JSON.json(instancedata));
    close(fh);
end

# This is a utility function for laoding json data saved at filename
function load_json(filename)
    fh = open(filename);
    content_string = readlines(fh)[1];
    result = JSON.parse(content_string);
    close(fh);
    return result;
end

# This is a very specific macro for running try/catch blocks
# and printing a set of parameters to STDERR when something
# goes wrong.
#
# If it hits the catch portion, it will always continue (i.e.
# we assume this is always called within a loop).
macro trycatchcontinue(trycode, catchcode)
    quote
        try
            $(esc(trycode))
        catch err
            $(esc(catchcode))
            showerror(STDERR,err); Base.show_backtrace(STDERR,catch_backtrace())
            println()
            continue
        end
    end
end

# This is a macro that times the execution of a block of code.
# It returns a tuple with the amount of time as the first argument
# and the value returned by the code (if applicable) as the second.
# @arguments
# ex: Any block of code (an AST)
# @returns: (<time_delta>, <code_return_value>)
macro gettime(ex)
    quote
        local tstart = time();
        local val = $(esc(ex));
        local tend = time();
        (tend - tstart, val)
    end
end


# This is a macro that can be used to set the seed before executing code.
# It can be used in the following ways:
#
# @setseed <some_nondeterministic_code>
# @setseed <seednumber> <some_nondeterministic_code>
#
# In the former case, it searches the above scope to look for a value
# bound to the variable 'seed.'
macro setseed(args...)
    if length(args) == 1
        quote
            srand($(esc(seed)))
            $(esc(args[1]))
        end
    elseif length(args) == 2
        quote
            local seed = $(esc(args[1]))
            srand(seed)
            $(esc(args[2]))
        end
    else
        error("@setseed only accepts 1 or 2 arguments")
    end
end

# This is a private helper function for parsestringcli and parseintcli.
# It basically cycles through all the commandline args in ARGS and parses
# them according to the acceptable short and long options given in 'opts'
function _parsecli(opts...)
    for arg in ARGS
        climatch = matchcommandlinearg(arg, opts...)
        if climatch != nothing
           return climatch
        end
    end
    return nothing
end

# This is a utility function for parsing a commandline arg and binding
# that _string_ to the first argument here called 'variable.'
#
# @arguments
# variable: This is the symbol that the parsed cli chunk will be bound to.
# args:     The first two elements of this list must be the short and long
#           options for the commandline argument.
#           If there is a third element, it is the default in case no match is found
#
# Example usage: @parsestringcli foo "n" "num" "bar"
macro parsestringcli(variable, args...)
    if length(args) < 2 || length(args) > 3
        error("@parsecli only takes 2 possible flags and optional default")
    end
    local climatch = _parsecli(args[1], args[2])
    if climatch != nothing
        return :($(esc(variable)) = $(esc(climatch)))
    elseif length(args) == 3
        return :($(esc(variable)) = $(esc(args[3])))
    end
end

# This is a utility function for parsing a commandline arg and binding
# that _int_ to the first argument here called 'variable.'
#
# @arguments
# variable: This is the symbol that the parsed cli chunk will be bound to.
# args:     The first two elements of this list must be the short and long
#           options for the commandline argument.
#           If there is a third element, it is the default in case no match is found
#
# Example usage: @parseintcli seed "s" "seed" 8
macro parseintcli(variable, args...)
    if length(args) < 2 || length(args) > 3
        error("@parsecli only takes 2 possible flags and optional default")
    end
    local climatch = _parsecli(args[1], args[2])
    if climatch != nothing
        return :($(esc(variable)) = parse(Int, ($(esc(climatch)))))
    elseif length(args) == 3
        return :($(esc(variable)) = $(esc(args[3])))
    end
end
