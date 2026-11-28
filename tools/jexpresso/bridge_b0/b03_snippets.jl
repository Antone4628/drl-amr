# The Julia side of bridge roadmap B0.3 (one process or two): the snippets a
# Python process calls through juliacall, and the same snippets run by a plain
# Julia process as the control.
#
# Included into Main by b03_child.py and b03_worker.py, and by a plain Julia
# process alike:
#
#     julia --project=. -e 'Base.include(Main, ARGS[1]); B03.main(ARGS[2:end])' b03_snippets.jl crash abort
#     julia --project=. -e 'Base.include(Main, ARGS[1]); B03.main(ARGS[2:end])' b03_snippets.jl spin 10
#     julia --project=. -e 'Base.include(Main, ARGS[1]); B03.main(ARGS[2:end])' b03_snippets.jl mpi
#     julia --project=. -e 'Base.include(Main, ARGS[1]); B03.main(ARGS[2:end])' b03_snippets.jl episodes AdvDiff advection2d_dg_insolve_native 2
#
# It is a probe: no package and no project of its own. MPI and
# PartitionedArrays are reached by their package ids, from the active
# project's manifest, and only by the functions that need them. Several of
# these functions end the process on purpose: `crash` is never called from a
# process that has run a case.

module B03

import Libdl
import LinearAlgebra

const MPI_ID = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")
const PARTITIONEDARRAYS_ID = Base.PkgId(Base.UUID("5a9dfac6-5c52-46f7-8278-5e2210713be9"), "PartitionedArrays")
const THROWN = "b03: thrown on purpose"
const THROWN_IN_WITH_MPI = "b03: thrown inside with_mpi"
const KINDS = ("throw", "throw_with_mpi", "abort", "null_read", "overflow", "mpi_abort", "exit", "hang")
const LIB_PATTERN = r"libmpi|libpmpi|libomp|libiomp|libgomp|openblas|libblas|liblapack|libgfortran|libquadmath|libgcc_s|libcrypto|libssl|libLLVM|libtorch|libc10|libshm|libpython|libjulia|libp4est|libsc\.|libstdc\+\+|libc\+\+|libunwind"i

# A package loaded after this module was defined is called through
# invokelatest, so that its methods are visible whatever world the caller
# was compiled in.
late(f, args...) = Base.invokelatest(f, args...)

function package(id::Base.PkgId, load::Bool)
    haskey(Base.loaded_modules, id) && return Base.loaded_modules[id]
    load || return nothing
    return Base.require(id)
end

# ---------------------------------------------------------------- the calls

@noinline empty() = nothing
@noinline add(a::Int, b::Int) = a + b

# A call that keeps the processor for `seconds` and returns its loop count.
# The safepoint is where a runtime that handles signals can deliver one.
function spin(seconds)
    stop = time() + Float64(seconds)
    n = 0
    while time() < stop
        n += 1
        n % 4096 == 0 && GC.safepoint()
    end
    return n
end

function gc_snapshot()
    g = Base.gc_num()
    return (Int(g.pause), Int(g.total_time), Int(g.full_sweep))
end

function mpi_init()
    MPI = package(MPI_ID, true)
    late(MPI.Init)
    return nothing
end

# ---------------------------------------------------------- the crash kinds

throw_error() = error(THROWN)

# PartitionedArrays' with_mpi, as Jexpresso's run.jl enters it. It initializes
# MPI if nothing has.
function throw_in_with_mpi()
    PA = package(PARTITIONEDARRAYS_ID, true)
    late(PA.with_mpi, _ -> error(THROWN_IN_WITH_MPI))
    return nothing
end

do_abort() = ccall(:abort, Cvoid, ())

@noinline null_read() = unsafe_load(Ptr{Int}(C_NULL))

# Recursion with no end, written so that it cannot be turned into a loop.
@noinline function deeper(n::Int)
    r = deeper(n + 1)
    return r > n ? r - n : r + n
end
overflow() = deeper(1)

function mpi_abort(code)
    MPI = package(MPI_ID, true)
    late(MPI.Init)
    late(MPI.Abort, MPI.COMM_WORLD, Int(code))
    return nothing
end

jl_exit(code) = exit(Int(code))

hang() = spin(1.0e9)

function crash(kind)
    k = String(kind)
    k == "throw" && return throw_error()
    k == "throw_with_mpi" && return throw_in_with_mpi()
    k == "abort" && return do_abort()
    k == "null_read" && return null_read()
    k == "overflow" && return overflow()
    k == "mpi_abort" && return mpi_abort(7)
    k == "exit" && return jl_exit(3)
    k == "hang" && return hang()
    error("b03: no such crash kind: $k")
end

# ------------------------------------------------------------- the reports
#
# Each report is one string of tab-separated lines, so that Python prints the
# text Julia would have printed and the two logs compare with `diff`.

function mpi_text()
    MPI = package(MPI_ID, false)
    MPI === nothing && return "mpi\tmodule_loaded\tfalse"
    lines = String["mpi\tmodule_loaded\ttrue"]
    initialized = late(MPI.Initialized)::Bool
    finalized = late(MPI.Finalized)::Bool
    push!(lines, "mpi\tinitialized\t$initialized")
    push!(lines, "mpi\tfinalized\t$finalized")
    if initialized && !finalized
        push!(lines, string("mpi\tthread_level\t", late(MPI.Query_thread)))
        push!(lines, string("mpi\tsize\t", late(MPI.Comm_size, MPI.COMM_WORLD)))
        push!(lines, string("mpi\trank\t", late(MPI.Comm_rank, MPI.COMM_WORLD)))
    end
    return join(lines, "\n")
end

function runtime_text()
    lines = String[
        string("runtime\tjulia_version\t", VERSION),
        string("runtime\tnthreads\t", Threads.nthreads()),
        string("runtime\tnthreads_interactive\t", Threads.nthreads(:interactive)),
        string("runtime\tngcthreads\t", Threads.ngcthreads()),
        string("runtime\thandle_signals\t", Base.JLOptions().handle_signals),
        string("runtime\tblas_threads\t", LinearAlgebra.BLAS.get_num_threads()),
        string("runtime\tpid\t", getpid()),
    ]
    return join(lines, "\n")
end

function libs_text()
    paths = Libdl.dllist()
    lines = String[string("libs_count\t", length(paths))]
    for path in sort(filter(p -> occursin(LIB_PATTERN, basename(p)), paths))
        push!(lines, "lib\t" * path)
    end
    return join(lines, "\n")
end

# ------------------------------------------------------------- the control

function say(io::IO, parts...)
    println(io, join(parts, '\t'))
    flush(io)
    return nothing
end

function first_line(err)
    text = sprint(showerror, err)
    return isempty(text) ? "" : String(first(split(text, '\n')))
end

function load_jexpresso()
    Core.eval(Main, :(using Jexpresso))
    return getfield(Main, :Jexpresso)
end

function main(args)
    io = stdout
    what = isempty(args) ? "" : String(args[1])
    say(io, "control", "julia", join(args, ' '))
    say(io, runtime_text())
    if what == "crash"
        kind = String(args[2])
        say(io, "ready", getpid())
        say(io, "crash", kind, "begin")
        try
            r = crash(kind)
            say(io, "crash", kind, "returned", repr(r))
        catch err
            say(io, "crash", kind, "raised", typeof(err), first_line(err))
        end
        say(io, "after", "add", add(1, 1))
    elseif what == "spin"
        seconds = parse(Float64, args[2])
        say(io, "ready", getpid())
        t0 = time()
        try
            n = spin(seconds)
            say(io, "spin_returned", round(time() - t0; digits = 2), n)
        catch err
            say(io, "spin_raised", typeof(err), round(time() - t0; digits = 2))
            rethrow()
        end
    elseif what == "mpi"
        say(io, "phase", "before_using")
        say(io, mpi_text())
        load_jexpresso()
        say(io, "phase", "after_using")
        say(io, mpi_text())
        say(io, libs_text())
    elseif what == "episodes"
        benchmark, case, n = String(args[2]), String(args[3]), parse(Int, args[4])
        jexpresso = load_jexpresso()
        say(io, "phase", "after_using")
        say(io, mpi_text())
        for i in 1:n
            say(io, "episode", i, "begin")
            late(jexpresso.run_case, benchmark, case)
            flush(stdout)
            say(io, "episode", i, "returned")
            say(io, "phase", "after_episode_$i")
            say(io, mpi_text())
        end
        say(io, libs_text())
    else
        error("b03: usage: crash <kind> | spin <seconds> | mpi | episodes <benchmark> <case> <n>")
    end
    say(io, "control_done", what)
    return nothing
end

end # module B03
