# Report what a Julia runtime is and what it has loaded.
#
# Bridge roadmap B0.1. The same file is included by a plain Julia process and
# by the Julia runtime that juliacall starts inside Python, so that the two
# reports have one format and can be compared with `diff`. It defines one
# function and runs nothing by itself.
#
#     include("b01_loaded.jl"); b01_report(stdout)
#
# Lines are tab-separated and start with a key:
#     julia_version, bindir, active_project, load_path, nthreads,
#     handle_signals, mpi_binary, gridapp4est: one value each
#     module: name, uuid, version, directory (one line per loaded module)
#     lib:    path (one line per loaded shared library whose name matches)

import Libdl

const B01_LIB_PATTERN = r"libmpi|libpmpi|libpython|libssl|libcrypto|libp4est|libsc\.|libjulia"i
const B01_MPIPREFERENCES = Base.PkgId(Base.UUID("3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"), "MPIPreferences")
const B01_GRIDAPP4EST = Base.PkgId(Base.UUID("c2c8e14b-f5fd-423d-9666-1dd9ad120af9"), "GridapP4est")

function b01_text(x)
    return x === nothing ? "-" : string(x)
end

function b01_report(io::IO)
    loaded = collect(Base.loaded_modules)
    println(io, "julia_version\t", VERSION)
    println(io, "bindir\t", Sys.BINDIR)
    println(io, "active_project\t", b01_text(Base.active_project()))
    println(io, "load_path\t", join(Base.load_path(), " | "))
    println(io, "nthreads\t", Threads.nthreads())
    println(io, "handle_signals\t", Base.JLOptions().handle_signals)

    mpi = "-"
    gridap = "-"
    rows = String[]
    for (id, m) in loaded
        push!(rows, join(("module", id.name, b01_text(id.uuid), b01_text(pkgversion(m)), b01_text(pkgdir(m))), "\t"))
        if id == B01_MPIPREFERENCES
            mpi = string(getfield(m, :binary))
        elseif id == B01_GRIDAPP4EST
            gridap = string(b01_text(pkgversion(m)), "\t", b01_text(pkgdir(m)))
        end
    end
    println(io, "mpi_binary\t", mpi)
    println(io, "gridapp4est\t", gridap)
    foreach(row -> println(io, row), sort(rows))
    for lib in sort(filter(l -> occursin(B01_LIB_PATTERN, basename(l)), Libdl.dllist()))
        println(io, "lib\t", lib)
    end
    flush(io)
    return nothing
end
