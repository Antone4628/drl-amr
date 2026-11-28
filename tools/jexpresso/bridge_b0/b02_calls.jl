# The Julia side of bridge roadmap B0.2 (call overhead): the functions a Python
# process calls through juliacall, and the same functions timed from Julia.
#
# Included into Main by b02_overhead.py and by a plain Julia process alike:
#
#     julia --project=. -e 'Base.include(Main, ARGS[1]); B02.baseline(stdout, false)' b02_calls.jl
#
# It is a probe: no package, no project, no dependency beyond Base. Nothing in
# it reaches Jexpresso. Every function that Python times returns either a
# Python-native value (nothing, a number, a tuple) or an Array.

module B02

const NS = (16, 64, 1024, 16384, 262144)
const KS = (1, 4, 8)
const ELTYPES = Dict{String,DataType}(
    "int8" => Int8,
    "int32" => Int32,
    "int64" => Int64,
    "float64" => Float64,
    "bool" => Bool,
)

# ---------------------------------------------------------------- the calls

@noinline empty() = nothing
@noinline scalar(x::Float64) = x + 1.0

# The patterned array: a[i, j, l] = i + 1000 j + 1000000 l in 1-based indices,
# with j = 1 and l = 1 where the array has no such axis. Integers narrower than
# the value wrap (two's complement), a Bool is the value's parity.
function pattern_int(I::CartesianIndex{N}) where {N}
    i = I[1]
    j = N >= 2 ? I[2] : 1
    l = N >= 3 ? I[3] : 1
    return i + 1000 * j + 1_000_000 * l
end
pattern_value(::Type{Bool}, I) = isodd(pattern_int(I))
pattern_value(::Type{T}, I) where {T<:Integer} = pattern_int(I) % T
pattern_value(::Type{T}, I) where {T<:AbstractFloat} = T(pattern_int(I))

function patterned(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    A = Array{T,N}(undef, dims)
    for I in CartesianIndices(A)
        A[I] = pattern_value(T, I)
    end
    return A
end

# Closures over one array, so that a timed call takes no argument and the
# array's type is concrete inside it.
make_pre(A) = () -> A
make_copy(A) = () -> copy(A)
make_fresh(::Type{T}, dims::NTuple{N,Int}) where {T,N} = () -> fill!(Array{T,N}(undef, dims), one(T))
make_zero_based(A::Array{Int32}) = () -> A .- Int32(1)

# getter("int32", false, n, k): a function returning one preallocated array.
# getter("int32", true, n, k): a function returning a newly allocated, filled one.
function getter(tname, fresh, dims...)
    T = ELTYPES[String(tname)]
    d = map(Int, dims)
    return fresh ? make_fresh(T, d) : make_pre(patterned(T, d))
end

# An index array as a glue routine would hold it: 1-based, 0 for absent.
function index_array(n, k)
    A = Matrix{Int32}(undef, Int(n), Int(k))
    for j in axes(A, 2), i in axes(A, 1)
        A[i, j] = (i + j) % 5 == 0 ? Int32(0) : Int32(i + j)
    end
    return A
end
zero_based(A) = A .- Int32(1)
index_getter(n, k) = make_pre(index_array(n, k))
zero_based_getter(n, k) = make_zero_based(index_array(n, k))

# Marks in the contract's vocabulary (-1 coarsen, 0 hold, +1 refine) made into
# the Vector{Cint} that adapt needs, in Jexpresso's (0 nothing, 1 refine,
# 2 coarsen). The vector is built here, never converted from the argument.
function marks_to_cint(marks)
    n = length(marks)
    flags = Vector{Cint}(undef, n)
    for i in 1:n
        m = marks[i]
        flags[i] = m == 1 ? Cint(1) : (m == -1 ? Cint(2) : Cint(0))
    end
    flags isa Vector{Cint} || error("marks_to_cint: not a Vector{Cint}")
    return flags
end
marks_len(marks) = length(marks_to_cint(marks))
function marks_report(marks)
    f = marks_to_cint(marks)
    return (string(typeof(marks)), length(f), count(==(Cint(0)), f), count(==(Cint(1)), f), count(==(Cint(2)), f))
end

# Nine arrays in the shapes of the contract's SolverState: level, neighbors,
# neighbor_wrap, family, can_refine, can_coarsen, centroid, h, one indicator.
function nine_arrays(n)
    n = Int(n)
    return (
        patterned(Int32, (n,)),
        patterned(Int32, (n, 4, 2)),
        patterned(Bool, (n, 4)),
        patterned(Int32, (n, 4)),
        patterned(Bool, (n,)),
        patterned(Bool, (n,)),
        patterned(Float64, (n, 2)),
        patterned(Float64, (n, 2)),
        patterned(Float64, (n, 1)),
    )
end
nine_getter(n) = make_pre(nine_arrays(n))
nine_singles(n) = map(make_pre, nine_arrays(n))

# ------------------------------------------- arrays kept by name (semantics)

const KEPT = Dict{String,Any}()

function keep(name, tname, dims...)
    KEPT[String(name)] = patterned(ELTYPES[String(tname)], map(Int, dims))
    return nothing
end
function keep_index(name, n, k)
    KEPT[String(name)] = index_array(n, k)
    return nothing
end
kept(name) = KEPT[String(name)]
kept_pointer(name) = UInt(pointer(KEPT[String(name)]))
kept_length(name) = length(KEPT[String(name)])
kept_read(name, lin) = KEPT[String(name)][Int(lin)]
function kept_write(name, lin, value)
    A = KEPT[String(name)]
    A[Int(lin)] = convert(eltype(A), value)
    return nothing
end

function bitvector(n)
    b = falses(Int(n))
    for i in eachindex(b)
        b[i] = isodd(i)
    end
    return b
end
function bitmatrix(n, k)
    b = falses(Int(n), Int(k))
    for j in axes(b, 2), i in axes(b, 1)
        b[i, j] = isodd(i + j)
    end
    return b
end
arrives_as(x) = string(typeof(x))

# ------------------------------------------------------ the collector, heap

function gc_snapshot()
    g = Base.gc_num()
    return (Int(g.pause), Int(g.total_time), Int(g.full_sweep))
end
function live_bytes()
    GC.gc(true)
    GC.gc(true)
    return Int(Base.gc_live_bytes())
end
# The number of slots PythonCall holds wrapped Julia values in; -1 where
# PythonCall is not loaded or keeps them elsewhere.
function wrapper_slots()
    try
        return length(getfield(Main, :PythonCall).JlWrap.Cjl.PYJLVALUES)
    catch
        return -1
    end
end

# ------------------------------------------------- the same calls from Julia

function summarize(samples::Vector{Float64})
    s = sort(samples)
    m = length(s)
    med = isodd(m) ? s[(m + 1) ÷ 2] : (s[m ÷ 2] + s[m ÷ 2 + 1]) / 2
    return med, s[1], s[m ÷ 4 + 1], s[(3 * m) ÷ 4 + 1]
end

# One cell, timed as b02_overhead.py times its own: the first call apart, an
# inner loop sized to target_ns, `repeats` samples of nanoseconds per call.
function timed(io::IO, label::String, f::F; target_ns::Int = 10_000_000, repeats::Int = 30) where {F}
    t0 = time_ns()
    Base.donotdelete(f())
    first_ns = Int(time_ns() - t0)
    inner = 1
    dt = 0
    while true
        t0 = time_ns()
        for _ in 1:inner
            Base.donotdelete(f())
        end
        dt = Int(time_ns() - t0)
        (dt >= target_ns ÷ 10 || inner >= 1 << 22) && break
        inner *= 2
    end
    inner = clamp(round(Int, inner * (target_ns / max(dt, 1))), 1, 10_000_000)
    samples = Vector{Float64}(undef, repeats)
    g0 = gc_snapshot()
    for r in 1:repeats
        t0 = time_ns()
        for _ in 1:inner
            Base.donotdelete(f())
        end
        samples[r] = Int(time_ns() - t0) / inner
    end
    g1 = gc_snapshot()
    med, mn, q1, q3 = summarize(samples)
    fields = (
        "cell", label, inner,
        round(med; digits = 1), round(mn; digits = 1), round(q1; digits = 1), round(q3; digits = 1),
        first_ns, g1[1] - g0[1], g1[2] - g0[2], g1[3] - g0[3],
    )
    println(io, join(fields, '\t'))
    flush(io)
    return nothing
end

dims_tag(dims) = join(dims, 'x')

function baseline(io::IO, floor_only::Bool)
    println(io, join(("info", "julia", VERSION, "threads", Threads.nthreads()), '\t'))
    println(io, join(("columns", "cell", "label", "inner", "median_ns", "min_ns", "q1_ns", "q3_ns",
                      "first_ns", "gc_pauses", "gc_ns", "gc_full"), '\t'))
    timed(io, "jl_empty", empty)
    timed(io, "jl_scalar", () -> scalar(1.5))
    if !floor_only
        for tname in ("int32", "float64"), n in NS, k in KS
            T = ELTYPES[tname]
            tag = string(tname, "_", dims_tag((n, k)))
            timed(io, "jl_fresh_" * tag, make_fresh(T, (n, k)))
            timed(io, "jl_copy_" * tag, make_copy(patterned(T, (n, k))))
        end
        for n in NS
            timed(io, string("jl_zb_", n, "x4"), make_zero_based(index_array(n, 4)))
            marks = Int8[(i % 3) - 1 for i in 0:(n - 1)]
            timed(io, string("jl_marks_", n), () -> marks_len(marks))
        end
    end
    println(io, join(("baseline_done", floor_only), '\t'))
    flush(io)
    return nothing
end

end # module B02
