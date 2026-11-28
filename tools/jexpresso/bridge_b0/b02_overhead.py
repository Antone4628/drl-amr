"""The probes of bridge roadmap B0.2: what a call from Python into Julia costs.

One mode per process, with the `rl-amr` environment active and the variables
of REQUIRED set:

    python b02_overhead.py floor
    python b02_overhead.py semantics
    python b02_overhead.py sweep
    python b02_overhead.py sweep --reduced --jexpresso
    python b02_overhead.py heap
    python b02_overhead.py libs --numpy absent
    python b02_overhead.py case --numpy before --case AdvDiff advection2d_dg_insolve_native

`floor` is the control: a Python no-op, the empty Julia call in its three
forms, and numpy's own copy of 16 MiB. `semantics` checks what an array is on
the Python side (a view or a copy, its dtype, shape, order and values) and
prints one `check` line per exact signal. `sweep` times the calls over the
element counts NS and the widths KS. `heap` returns 10^5 arrays and reads
Julia's live heap. `libs` lists the libraries loaded in the process. `case`
evaluates `Jexpresso.run_case` with numpy imported first (run it from the root
of the Jexpresso clone).

The Julia functions are b02_calls.jl, beside this file; a plain Julia process
times the same functions with `B02.baseline`.

juliacall is imported inside a function, never at import of this module, and
only after the variables are checked: without PYTHON_JULIACALL_EXE and
PYTHON_JULIACALL_PROJECT the import would hand control to JuliaPkg. numpy is
imported inside a function too, because where it is imported is an option.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CALLS_JL = HERE / "b02_calls.jl"

REQUIRED = (
    "PYTHON_JULIACALL_EXE",
    "PYTHON_JULIACALL_PROJECT",
    "JULIA_LOAD_PATH",
    "JULIA_CONDAPKG_BACKEND",
    "PYTHON_JULIACALL_STARTUP_FILE",
    "PYTHON_JULIACALL_THREADS",
)
REPORTED = REQUIRED + ("PYTHON_JULIACALL_HANDLE_SIGNALS", "PYTHON_JULIACALL_LIB", "PYTHON_JULIACALL_BINDIR")
EXPECTED_JULIACALL = "0.9.36"

NS = (16, 64, 1024, 16384, 262144)
KS = (1, 4, 8)
TIMED_TYPES = ("int32", "float64")
REDUCED_NS = (1024, 16384)
REDUCED_K = 4
CENSUS = (
    ("int8", (16, 4)),
    ("int32", (16, 4)),
    ("int64", (16, 4)),
    ("float64", (16, 4)),
    ("bool", (16, 4)),
    ("int32", (16,)),
    ("int32", (16, 4, 2)),
)
HEAP_CALLS = 100_000
HEAP_DIMS = (1024, 8)
HEAP_BOUND_BYTES = HEAP_CALLS * HEAP_DIMS[0] * HEAP_DIMS[1] * 8 // 100
COLUMNS = (
    "cell",
    "label",
    "inner",
    "median_ns",
    "min_ns",
    "q1_ns",
    "q3_ns",
    "first_ns",
    "gc_pauses",
    "gc_ns",
    "gc_full",
)


def say(*parts: object) -> None:
    """Print one tab-separated line and flush, so Python's lines and Julia's keep their order in a log."""
    print(*parts, sep="\t", flush=True)


# ------------------------------------------------------------------ timing


@dataclass
class Config:
    """How a cell is timed, and where Julia's collector is read."""

    target_ns: int
    repeats: int
    gcsnap: Callable[[], Any] | None = None


def measure(fn: Callable[[], Any], target_ns: int, repeats: int) -> tuple[int, float, float, float, float, int]:
    """Time `fn`: the first call apart, then `repeats` samples of an inner loop sized to `target_ns`.

    Returns the inner count, the median, minimum, lower and upper quartile in nanoseconds per call, and the
    first call's nanoseconds. Python's collector is off inside the samples.
    """
    t0 = time.perf_counter_ns()
    fn()
    first_ns = time.perf_counter_ns() - t0
    inner = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        dt = time.perf_counter_ns() - t0
        if dt >= target_ns // 10 or inner >= 1 << 22:
            break
        inner *= 2
    inner = max(1, min(10_000_000, round(inner * target_ns / max(dt, 1))))
    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for _ in range(inner):
                fn()
            samples.append((time.perf_counter_ns() - t0) / inner)
    finally:
        if was_enabled:
            gc.enable()
    samples.sort()
    m = len(samples)
    return inner, statistics.median(samples), samples[0], samples[m // 4], samples[min(m - 1, (3 * m) // 4)], first_ns


def cell(label: str, fn: Callable[[], Any], cfg: Config) -> None:
    """Time one cell and print its line, with the difference in Julia's collector counters across it."""
    g0 = tuple(cfg.gcsnap()) if cfg.gcsnap else (0, 0, 0)
    inner, med, low, q1, q3, first_ns = measure(fn, cfg.target_ns, cfg.repeats)
    g1 = tuple(cfg.gcsnap()) if cfg.gcsnap else (0, 0, 0)
    say(
        "cell",
        label,
        inner,
        f"{med:.1f}",
        f"{low:.1f}",
        f"{q1:.1f}",
        f"{q3:.1f}",
        first_ns,
        *(b - a for a, b in zip(g0, g1, strict=True)),
    )


def noop() -> None:
    return None


# -------------------------------------------------------------- the runtime


def start(numpy_mode: str):
    """Check the variables, import juliacall (numpy before or after it, or not at all), return Julia's Main."""
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise SystemExit("refusing to import juliacall; not set: " + ", ".join(missing))
    for name in REPORTED:
        say("env", name, os.environ.get(name, "<unset>"))
    say("python", sys.version.split()[0], sys.executable)
    say("cwd", Path.cwd())
    say("numpy_mode", numpy_mode)
    if numpy_mode == "before":
        import numpy

        say("numpy_imported", "before", numpy.__version__)
    say("numpy_loaded_before_juliacall", "numpy" in sys.modules)
    say("torch_imported_before", "torch" in sys.modules)
    t0 = time.perf_counter()
    import juliacall

    say("import_seconds", f"{time.perf_counter() - t0:.2f}")
    say("juliacall_version", juliacall.__version__)
    if numpy_mode == "after":
        import numpy

        say("numpy_imported", "after", numpy.__version__)
    return juliacall.Main


def load_b02(jl, with_jexpresso: bool):
    """Optionally `using Jexpresso`, then include b02_calls.jl into Main and return its module."""
    if with_jexpresso:
        t0 = time.perf_counter()
        jl.seval("using Jexpresso")
        say("using_jexpresso_seconds", f"{time.perf_counter() - t0:.2f}")
    say("jexpresso_loaded", with_jexpresso)
    jl.seval(f'Base.include(Main, raw"{CALLS_JL}")')
    say("pythoncall_version", jl.seval("string(pkgversion(PythonCall))"))
    return jl.B02


def finish(mode: str, failures: int | None = None) -> None:
    """The lines of signal E1 that close every mode."""
    import juliacall

    version_ok = juliacall.__version__ == EXPECTED_JULIACALL
    torch_absent = "torch" not in sys.modules
    juliapkg_absent = "juliapkg" not in sys.modules
    say("check", "E1", "juliacall_version", "OK" if version_ok else "FAIL", juliacall.__version__)
    say("check", "E1", "torch_absent", "OK" if torch_absent else "FAIL", torch_absent)
    say("check", "E1", "juliapkg_never_imported", "OK" if juliapkg_absent else "FAIL", juliapkg_absent)
    say("numpy_loaded_at_end", "numpy" in sys.modules)
    if failures is not None:
        failures += (not version_ok) + (not torch_absent) + (not juliapkg_absent)
        say("result", mode, "OK" if failures == 0 else "FAIL", failures)
    say("mode_done", mode)


# ------------------------------------------------------------------- floor


def empty_cells(jl, b02, cfg: Config) -> None:
    """The empty call in its three forms, and one scalar in and out."""
    empty = b02.empty
    scalar = b02.scalar
    cell("python_noop", noop, cfg)
    cell("empty_handle", empty, cfg)
    cell("empty_attr", lambda: jl.B02.empty(), cfg)
    cell("empty_seval", lambda: jl.seval("B02.empty()"), cfg)
    cell("scalar_handle", lambda: scalar(1.5), cfg)


def mode_floor(jl, b02, cfg: Config, numpy_mode: str) -> None:
    say("columns", *COLUMNS)
    empty_cells(jl, b02, cfg)
    if numpy_mode != "absent":
        import numpy as np

        a = np.ones(1 << 21, dtype=np.float64)
        cell("numpy_copy_16MiB", a.copy, cfg)


# --------------------------------------------------------------- semantics


class Checks:
    """Counts the exact signals that failed."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, signal: str, name: str, ok: bool, detail: object = "") -> None:
        ok = bool(ok)
        if not ok:
            self.failures += 1
        say("check", signal, name, "OK" if ok else "FAIL", detail)


def expected(np, tname: str, dims: tuple[int, ...]):
    """The patterned array of b02_calls.jl, built by numpy: i + 1000 j + 1000000 l, 1-based."""
    grids = np.indices(dims, dtype=np.int64) + 1
    j = grids[1] if len(dims) >= 2 else 1
    l = grids[2] if len(dims) >= 3 else 1
    value = grids[0] + 1000 * j + 1_000_000 * l
    if tname == "bool":
        return (value % 2) == 1
    return value.astype(np.dtype(tname))


def fortran_index(np, lin: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """The numpy index of Julia's 1-based linear index `lin`."""
    return tuple(int(i) for i in np.unravel_index(lin - 1, dims, order="F"))


def another(tname: str, current: object, candidate: int) -> object:
    """A value of the array's type that differs from `current`."""
    if tname == "bool":
        return not bool(current)
    return candidate if current != candidate else candidate + 1


def pointer_of(v) -> int:
    return int(v.__array_interface__["data"][0])


def attempt(name: str, fn: Callable[[], Any]) -> None:
    """Record what `fn` gives or raises; nothing here is a signal."""
    try:
        say("record", name, fn())
    except Exception as exc:
        say("record", name, "raised", type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "")


def check_census(np, b02, checks: Checks) -> None:
    """Signals E2, E3 and E4 for each element type and shape of CENSUS."""
    for tname, dims in CENSUS:
        name = f"{tname}_{'x'.join(map(str, dims))}"
        b02.keep(name, tname, *dims)
        x = b02.kept(name)
        v = np.asarray(x)
        want = expected(np, tname, dims)
        say("record", name, "python_type", type(x).__name__)
        say("record", name, "base_type", type(v.base).__name__)
        say(
            "record",
            name,
            "strides",
            v.strides,
            "F",
            v.flags.f_contiguous,
            "C",
            v.flags.c_contiguous,
            "W",
            v.flags.writeable,
        )
        checks.check("E2", f"{name}:dtype", v.dtype == want.dtype, v.dtype)
        checks.check("E2", f"{name}:shape", v.shape == tuple(dims), v.shape)
        checks.check("E2", f"{name}:values", v.shape == want.shape and bool(np.array_equal(v, want)))

        copied = x.to_numpy()
        checks.check("E4", f"{name}:copy_values", bool(np.array_equal(copied, want)))
        checks.check("E4", f"{name}:copy_pointer_differs", pointer_of(copied) != pointer_of(v))

        checks.check("E3", f"{name}:pointer", pointer_of(v) == int(b02.kept_pointer(name)), pointer_of(v))
        if len(dims) > 1:
            checks.check("E3", f"{name}:fortran_order", v.flags.f_contiguous and not v.flags.c_contiguous)
        first = fortran_index(np, 1, dims)
        last_lin = int(b02.kept_length(name))
        last = fortran_index(np, last_lin, dims)
        from_julia = another(tname, v[first].item(), 77)
        b02.kept_write(name, 1, from_julia)
        checks.check("E3", f"{name}:julia_write_seen_in_numpy", v[first].item() == from_julia, v[first].item())
        from_numpy = another(tname, v[last].item(), 78)
        v[last] = from_numpy
        checks.check("E3", f"{name}:numpy_write_seen_in_julia", b02.kept_read(name, last_lin) == from_numpy)

        second = fortran_index(np, 2, dims)
        before = copied[second].item()
        later = another(tname, before, 55)
        b02.kept_write(name, 2, later)
        checks.check(
            "E4", f"{name}:later_write_not_in_copy", copied[second].item() == before and v[second].item() == later
        )


def check_zero_based(np, b02, checks: Checks) -> None:
    """Signal E5: the conversion on the Julia side, and the in-place subtraction on a view."""
    b02.keep_index("idx", 16, 4)
    one_based = np.asarray(b02.kept("idx"))
    zb = np.asarray(b02.zero_based(b02.kept("idx")))
    say("record", "idx", "absent_places", int((one_based == 0).sum()), "of", one_based.size)
    checks.check("E5", "dtype", zb.dtype == np.dtype("int32"), zb.dtype)
    checks.check(
        "E5", "absent_is_minus_one", bool(np.array_equal(one_based == 0, zb == -1)) and bool((one_based == 0).any())
    )
    checks.check("E5", "others_one_less", bool(np.array_equal(zb, one_based - 1)))
    checks.check(
        "E5",
        "source_unchanged",
        pointer_of(zb) != pointer_of(one_based) and int(b02.kept_read("idx", 1)) == int(one_based.flat[0]),
    )

    b02.keep("throwaway", "int32", 16, 4)
    view = np.asarray(b02.kept("throwaway"))
    before = int(b02.kept_read("throwaway", 1))
    view -= 1
    after = int(b02.kept_read("throwaway", 1))
    checks.check("E5", "in_place_subtract_rewrites_julia", after == before - 1, f"{before}->{after}")


def check_marks(np, b02, checks: Checks) -> None:
    """Signal E6: int8 marks in, a Vector{Cint} built on the Julia side."""
    for dtype in ("int8", "int32", "int64"):
        marks = (np.arange(64) % 3 - 1).astype(dtype)
        arrived, n, holds, refines, coarsens = b02.marks_report(marks)
        say("record", f"marks_{dtype}", "arrives_as", arrived)
        ok = (
            int(n) == marks.size
            and int(holds) == int((marks == 0).sum())
            and int(refines) == int((marks == 1).sum())
            and int(coarsens) == int((marks == -1).sum())
        )
        signal = "E6" if dtype == "int8" else "E6+"
        checks.check(signal, f"marks_{dtype}", ok, f"n={n} hold={holds} refine={refines} coarsen={coarsens}")
    say("record", "float64_matrix", "arrives_as", b02.arrives_as(np.zeros((4, 2))))
    say("record", "c_ordered_int32", "arrives_as", b02.arrives_as(np.zeros((4, 2), dtype=np.int32, order="C")))


def record_others(np, b02) -> None:
    """What the source read predicts and no signal declares."""
    for name, make, want in (
        ("bitvector", lambda: b02.bitvector(16), np.arange(1, 17) % 2 == 1),
        ("bitmatrix", lambda: b02.bitmatrix(16, 4), (np.indices((16, 4)).sum(axis=0) + 2) % 2 == 1),
    ):
        x = make()
        say("record", name, "python_type", type(x).__name__)

        def as_array(x=x, want=want):
            v = np.asarray(x)
            return f"dtype={v.dtype} shape={v.shape} equal={bool(np.array_equal(v, want))} base={type(v.base).__name__}"

        def to_numpy(x=x, want=want):
            v = x.to_numpy()
            return f"dtype={v.dtype} shape={v.shape} equal={bool(np.array_equal(v, want))}"

        def to_numpy_nocopy(x=x):
            v = x.to_numpy(copy=False)
            return f"dtype={v.dtype} shape={v.shape}"

        attempt(f"{name}:asarray", as_array)
        attempt(f"{name}:to_numpy", to_numpy)
        attempt(f"{name}:to_numpy_copy_false", to_numpy_nocopy)

    getter = b02.getter("int32", False, 4, 16)
    transposed = np.asarray(getter()).T
    say(
        "record",
        "transposed_4x16",
        "shape",
        transposed.shape,
        "C",
        transposed.flags.c_contiguous,
        "equal",
        bool(np.array_equal(transposed, expected(np, "int32", (4, 16)).T)),
    )
    nine = b02.nine_getter(16)()
    say("record", "nine", "python_type", type(nine).__name__, "length", len(nine))
    say("record", "nine", "element_types", ",".join(type(a).__name__ for a in nine))
    say("record", "nine", "dtypes", ",".join(str(np.asarray(a).dtype) for a in nine))
    say("record", "nine", "shapes", ",".join(str(np.asarray(a).shape) for a in nine))
    say("record", "empty", "returns", repr(b02.empty()))
    say("record", "scalar", "returns", repr(b02.scalar(1.5)))
    say("record", "gc_snapshot", "python_type", type(b02.gc_snapshot()).__name__)
    attempt(
        "to_numpy_copy_false:shares",
        lambda: pointer_of(b02.kept("idx").to_numpy(copy=False)) == int(b02.kept_pointer("idx")),
    )


def mode_semantics(b02) -> int:
    import numpy as np

    checks = Checks()
    check_census(np, b02, checks)
    check_zero_based(np, b02, checks)
    check_marks(np, b02, checks)
    record_others(np, b02)
    return checks.failures


# ------------------------------------------------------------------- sweep


def array_cells(np, b02, cfg: Config, tname: str, dims: tuple[int, ...], reduced: bool) -> None:
    """One element type and shape: the wrapper, the view and the copies, on a preallocated and a fresh array."""
    tag = f"{tname}_{'x'.join(map(str, dims))}"
    pre = b02.getter(tname, False, *dims)
    cell(f"wrap_{tag}", pre, cfg)
    cell(f"view_{tag}", lambda: np.asarray(pre()), cfg)
    cell(f"copy_{tag}", lambda: pre().to_numpy(), cfg)
    if reduced:
        return
    view = np.asarray(pre())
    cell(f"ccopy_{tag}", lambda: np.ascontiguousarray(np.asarray(pre())), cfg)
    cell(f"npcopy_{tag}", lambda: view.copy(order="K"), cfg)
    cell(f"npccopy_{tag}", lambda: view.copy(order="C"), cfg)
    fresh = b02.getter(tname, True, *dims)
    cell(f"fresh_wrap_{tag}", fresh, cfg)
    cell(f"fresh_view_{tag}", lambda: np.asarray(fresh()), cfg)
    cell(f"fresh_copy_{tag}", lambda: fresh().to_numpy(), cfg)


def transposed_cell(np, b02, cfg: Config, tname: str, n: int, k: int) -> None:
    """Julia returns (k, n); Python reads the transpose, a C-ordered (n, k) view."""
    pre = b02.getter(tname, False, k, n)
    cell(f"tview_{tname}_{n}x{k}", lambda: np.asarray(pre()).T, cfg)


def zero_based_cells(np, b02, cfg: Config, n: int) -> None:
    """The 0-based pass on the Julia side against the same pass in numpy."""
    in_julia = b02.zero_based_getter(n, 4)
    one_based = b02.index_getter(n, 4)
    cell(f"zb_julia_{n}x4", lambda: np.asarray(in_julia()), cfg)
    cell(f"zb_numpy_{n}x4", lambda: np.asarray(one_based()) - 1, cfg)


def marks_cell(np, b02, cfg: Config, n: int) -> None:
    marks = (np.arange(n) % 3 - 1).astype(np.int8)
    marks_len = b02.marks_len
    cell(f"marks_in_{n}", lambda: marks_len(marks), cfg)


def nine_cells(np, b02, cfg: Config, n: int) -> None:
    """Nine arrays in one call against nine calls of one array each."""
    together = b02.nine_getter(n)
    singles = tuple(b02.nine_singles(n))
    cell(f"nine_one_call_{n}", lambda: [np.asarray(a) for a in together()], cfg)
    cell(f"nine_nine_calls_{n}", lambda: [np.asarray(f()) for f in singles], cfg)


def mode_sweep(jl, b02, cfg: Config, reduced: bool) -> None:
    import numpy as np

    say("columns", *COLUMNS)
    empty_cells(jl, b02, cfg)
    if reduced:
        for tname in TIMED_TYPES:
            for n in REDUCED_NS:
                array_cells(np, b02, cfg, tname, (n, REDUCED_K), True)
        return
    for tname in TIMED_TYPES:
        for n in NS:
            for k in KS:
                array_cells(np, b02, cfg, tname, (n, k), False)
    for n in NS:
        array_cells(np, b02, cfg, "int32", (n, 4, 2), False)
    for tname in TIMED_TYPES:
        for n in NS:
            for k in (4, 8):
                transposed_cell(np, b02, cfg, tname, n, k)
    for n in NS:
        zero_based_cells(np, b02, cfg, n)
    for n in NS:
        marks_cell(np, b02, cfg, n)
    for n in NS[:4]:
        nine_cells(np, b02, cfg, n)


# -------------------------------------------------------- heap, libs, case


def mode_heap(b02) -> int:
    """Signal E7: Julia's live heap after HEAP_CALLS returned arrays have been dropped."""
    import numpy as np

    checks = Checks()
    fresh = b02.getter("float64", True, *HEAP_DIMS)
    fresh()
    for phase, use in (("wrappers", lambda x: None), ("views", lambda x: np.asarray(x))):
        before = int(b02.live_bytes())
        slots_before = int(b02.wrapper_slots())
        t0 = time.perf_counter()
        for _ in range(HEAP_CALLS):
            use(fresh())
        seconds = time.perf_counter() - t0
        gc.collect()
        after = int(b02.live_bytes())
        slots_after = int(b02.wrapper_slots())
        say("record", f"heap_{phase}", "calls", HEAP_CALLS, "seconds", f"{seconds:.2f}")
        say("record", f"heap_{phase}", "live_before", before, "live_after", after, "difference", after - before)
        say("record", f"heap_{phase}", "slots_before", slots_before, "slots_after", slots_after)
        signal = "E7" if phase == "wrappers" else "E7+"
        checks.check(signal, f"live_heap_{phase}", abs(after - before) < HEAP_BOUND_BYTES, f"bound={HEAP_BOUND_BYTES}")
    return checks.failures


def use_blas() -> None:
    """One matrix product, so that numpy's BLAS is loaded and its threads have started."""
    import numpy as np

    a = np.ones((200, 200))
    say("numpy_matmul_trace", float((a @ a).trace()))


def mode_libs(jl, numpy_mode: str) -> None:
    if numpy_mode != "absent":
        use_blas()
    libs = sorted(str(p) for p in jl.seval("import Libdl; Libdl.dllist()"))
    say("libs_count", len(libs))
    for path in libs:
        say("lib", path)


def mode_case(jl, benchmark: str, case: str) -> None:
    """`Jexpresso.run_case` at Julia's top level, with numpy imported first and its BLAS used once."""
    for name in (benchmark, case):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"not a plain case name: {name!r}")
    cwd = Path.cwd()
    if not ((cwd / "Project.toml").is_file() and (cwd / "problems" / benchmark / case).is_dir()):
        raise SystemExit(f"run from the root of the Jexpresso clone: no problems/{benchmark}/{case} under {cwd}")
    use_blas()
    t0 = time.perf_counter()
    jl.seval("using Jexpresso")
    say("using_jexpresso_seconds", f"{time.perf_counter() - t0:.2f}")
    t1 = time.perf_counter()
    jl.seval(f'Jexpresso.run_case("{benchmark}", "{case}")')
    jl.seval("flush(stdout)")
    say("run_case_seconds", f"{time.perf_counter() - t1:.2f}")
    say("run_case", benchmark, case, "returned")


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("floor", "semantics", "sweep", "heap", "libs", "case"))
    parser.add_argument("--numpy", dest="numpy_mode", choices=("absent", "before", "after"), default="after")
    parser.add_argument("--jexpresso", action="store_true", help="`using Jexpresso` before anything is called")
    parser.add_argument("--reduced", action="store_true", help="sweep: the subset compared with and without Jexpresso")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--target-ms", type=float, default=10.0)
    parser.add_argument("--case", nargs=2, metavar=("BENCHMARK", "CASE"))
    args = parser.parse_args(argv)

    if args.numpy_mode == "absent" and args.mode not in ("floor", "libs"):
        raise SystemExit(f"mode {args.mode} needs numpy; --numpy absent is for floor and libs")
    if args.mode == "case" and (args.case is None or args.numpy_mode != "before"):
        raise SystemExit("mode case needs --case BENCHMARK CASE and --numpy before")

    say("mode", args.mode, "reduced", args.reduced, "repeats", args.repeats, "target_ms", args.target_ms)
    jl = start(args.numpy_mode)
    failures: int | None = None
    if args.mode == "case":
        mode_case(jl, *args.case)
    elif args.mode == "libs":
        if args.jexpresso:
            jl.seval("using Jexpresso")
        say("jexpresso_loaded", args.jexpresso)
        mode_libs(jl, args.numpy_mode)
    else:
        b02 = load_b02(jl, args.jexpresso)
        cfg = Config(target_ns=int(args.target_ms * 1_000_000), repeats=args.repeats, gcsnap=b02.gc_snapshot)
        if args.mode == "floor":
            mode_floor(jl, b02, cfg, args.numpy_mode)
        elif args.mode == "semantics":
            failures = mode_semantics(b02)
        elif args.mode == "sweep":
            mode_sweep(jl, b02, cfg, args.reduced)
        else:
            failures = mode_heap(b02)
    if args.numpy_mode == "absent":
        say("check", "numpy_absent", "numpy_never_imported", "OK" if "numpy" not in sys.modules else "FAIL")
    finish(args.mode, failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
