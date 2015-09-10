# Computes Stein discrepancy between a weighted sample and a target distribution
#
# Args:
#   points - n x d array of sample points
#   weights - n x 1 array of real-valued weights associated with sample points
#     (default: equal weights)
#   target - SteinDistribution representing target probability distribution
#   solver - solver name recognized by getsolver or AbstractMathProgSolver for
#     solving Stein discrepancy program
#   operator - string in {"langevin"} indicating the Stein operator
#     to use (default: "langevin")
#   classical - if true and operator == "langevin", computes Langevin classical 
#     Stein discrepancy instead of graph discrepancy; only valid for d == 1 
#     (default: false)
function stein_discrepancy(; points=[],
                           weights=fill(1/size(points,1), size(points,1)),
                           target=None,
                           solver=None,
                           operator="langevin",
                           classical=false)
    # Check arguments
    isempty(points) && error("Must provide non-empty array of sample points")
    isa(target, SteinDistribution) ||
        error("Must specify target of type SteinDistribution")
    if isa(solver, String)
        solver = getsolver(solver)
    end
    isa(solver, AbstractMathProgSolver) ||
        error("Must specify solver of type String or AbstractMathProgSolver")

    # Form weighted sample object
    sample = SteinDiscrete(points, weights);

    # Solve Stein discrepancy optimization program
    if operator == "langevin"
        if classical
            langevin_classical_discrepancy(sample, target, solver);
        else
            langevin_graph_discrepancy(sample, target, solver);
        end
    else
        error("unrecognized operator: $(operator)")
    end
end

# Returns solver object associated with string name
#
# Args:
#  solver - Optimization problem solver name in
#   {"gurobi","gurobi_no_crossover","clp","glpk","ecos","scs"}
function getsolver(solver::String)
    solver = lowercase(solver)
    if solver == "gurobi"
        # Gurobi Options
        # BarConvTol: tolerance in relative primal to dual error for barrier method;
        #  default: 1e-8
        # Crossover: how to convert barrier solution into basic solution
        #  -1 (default): automatic setting; 0: disable crossover
        # Method: solution method
        #  -1 (default): automatic setting; 2: barrier method
        # NumericFocus: degree of numerical issue detection and management
        #  0 (default): automatic setting;
        #  Settings 1-3 imply increasing care in numerical computations
        # Presolve: controls the presolve level
        #  -1 (default): automatic setting
        #  0: disabled; 1: conservative; 2: aggressive
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver()
    elseif solver == "gurobi_no_crossover"
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver(Method=2,Crossover=0)
    elseif solver == "gurobi_one_thread"
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver(Threads=1)
    elseif solver == "clp"
        eval(Expr(:import,:Clp))
        # LogLevel: set to 1, 2, 3, or 4 for increasing output (default 0)
        # PresolveType: set to 1 to disable presolve
        # SolveType: choose the solution method:
        #  0 - dual simplex, 1 - primal simplex,
        #  3 - barrier with crossover to optimal basis,
        #  4 - barrier without crossover to optimal basis
        #  5 - automatic
        solver_object = Main.Clp.ClpSolver(SolveType=4,LogLevel=1)
    elseif solver == "glpk"
        eval(Expr(:import,:GLPKMathProgInterface))
        # GLPK Options
        # msg_level: verbosity level in {0,...,4}; defaults to 0 (no output)
        # presolve: presolve LP?; defaults to false
        solver_object = Main.GLPKMathProgInterface.GLPKSolverLP()
    elseif solver == "scs"
        eval(Expr(:import,:SCS))
        solver_object = Main.SCS.SCSSolver()
    elseif solver == "ecos"
        eval(Expr(:import,:ECOS))
        solver_object = Main.ECOS.ECOSSolver()
    else
        error("Unknown solver $solver_object requested.")
    end
end
