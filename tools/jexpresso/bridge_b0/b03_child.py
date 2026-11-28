"""The child processes of bridge roadmap B0.3: one process, with Julia inside it or not.

One mode per process, with the `rl-amr` environment active. Where `--imports`
names juliacall, the variables of REQUIRED must be set:

    python b03_child.py coexist --imports juliacall,numpy,torch
    python b03_child.py coexist --imports numpy,torch
    python b03_child.py coexist --imports juliacall,numpy,torch,sb3,ssl --case AdvDiff advection2d_dg_insolve_native
    python b03_child.py mpi
    python b03_child.py episodes --case AdvDiff advection2d_dg_insolve_native --n 2
    python b03_child.py spin --where julia --seconds 10
    python b03_child.py spin --where python --seconds 10 --imports ""
    python b03_child.py crash --kind abort
    python b03_child.py thread
    python b03_child.py floor --stages mpi,jexpresso,numpy,torch

`--imports` is the order in which the process imports juliacall, numpy, torch,
stable-baselines3 (`sb3`) and `ssl`; what is not named is not imported.
`coexist` runs one optimizer step of a small network when torch is imported,
with Julia calls before and after when juliacall is. `mpi` reports MPI's state
before and after `using Jexpresso`. `episodes` evaluates `Jexpresso.run_case`
n times in the one process (run it, and `coexist --case`, from the root of the
Jexpresso clone). `spin` prints a `ready` line and then keeps the processor,
in Julia or in Python, as a target for a signal. `crash` calls one snippet of
b03_snippets.jl that throws or ends the process; it refuses to run from the
clone. `thread` calls Julia from threads other than the main one. `floor`
times the empty call as MPI, Jexpresso, numpy and torch enter the process.

The Julia functions are b03_snippets.jl, beside this file; a plain Julia
process runs the same snippets with `B03.main`. b03_host.py starts a child,
sends it a signal and records how it ended.

juliacall is imported inside a function, never at import of this module, and
only after the variables are checked: without PYTHON_JULIACALL_EXE and
PYTHON_JULIACALL_PROJECT the import would hand control to JuliaPkg.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path

from b02_overhead import COLUMNS, Config, cell

HERE = Path(__file__).resolve().parent
SNIPPETS_JL = HERE / "b03_snippets.jl"

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
IMPORTABLE = ("juliacall", "numpy", "torch", "sb3", "ssl")
WATCHED = ("juliacall", "numpy", "torch", "stable_baselines3", "ssl", "juliapkg")
KINDS = ("throw", "throw_with_mpi", "abort", "null_read", "overflow", "mpi_abort", "exit", "hang")
THROWN = {"throw": "b03: thrown on purpose", "throw_with_mpi": "b03: thrown inside with_mpi"}
STAGES = ("mpi", "jexpresso", "numpy", "torch")
THREAD_CALLS = 10_000


def say(*parts: object) -> None:
    """Print one tab-separated line and flush, so Python's lines and Julia's keep their order in a log."""
    print(*parts, sep="\t", flush=True)


def first_line(text: object, limit: int = 160) -> str:
    """The first line of a message, cut to `limit` characters."""
    lines = str(text).splitlines()
    return lines[0][:limit] if lines else ""


def emit(text: object) -> None:
    """Print a report that Julia returned as one string of lines."""
    for line in str(text).splitlines():
        print(line, flush=True)


class Checks:
    """Counts the exact signals that failed."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, signal: str, name: str, ok: bool, detail: object = "") -> None:
        ok = bool(ok)
        if not ok:
            self.failures += 1
        say("check", signal, name, "OK" if ok else "FAIL", detail)


# ------------------------------------------------------------- the imports


def import_in_order(names: list[str]):
    """Import what `names` lists, in that order, and return Julia's Main if juliacall is among them."""
    jl = None
    for name in names:
        say("before_import", name, *(f"{m}={m in sys.modules}" for m in WATCHED))
        t0 = time.perf_counter()
        if name == "juliacall":
            missing = [v for v in REQUIRED if not os.environ.get(v)]
            if missing:
                raise SystemExit("refusing to import juliacall; not set: " + ", ".join(missing))
            for v in REPORTED:
                say("env", v, os.environ.get(v, "<unset>"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import juliacall
            for w in caught:
                say("import_warning", "juliacall", w.category.__name__, first_line(w.message))
            say("import_warnings", "juliacall", len(caught))
            say("juliacall_version", juliacall.__version__)
            jl = juliacall.Main
        elif name == "numpy":
            import numpy

            say("numpy_version", numpy.__version__)
        elif name == "torch":
            import torch

            say("torch_version", torch.__version__, "threads", torch.get_num_threads())
        elif name == "sb3":
            import stable_baselines3

            say("sb3_version", stable_baselines3.__version__)
        elif name == "ssl":
            import ssl

            ssl.create_default_context()
            say("ssl_version", ssl.OPENSSL_VERSION)
        say("imported", name, f"{time.perf_counter() - t0:.2f}")
    return jl


def load_b03(jl):
    """Include b03_snippets.jl into Main and return its module."""
    jl.seval(f'Base.include(Main, raw"{SNIPPETS_JL}")')
    return jl.B03


def use_jexpresso(jl) -> None:
    t0 = time.perf_counter()
    jl.seval("using Jexpresso")
    say("using_jexpresso_seconds", f"{time.perf_counter() - t0:.2f}")


def check_case(benchmark: str, case: str) -> None:
    for name in (benchmark, case):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"not a plain case name: {name!r}")
    cwd = Path.cwd()
    if not ((cwd / "Project.toml").is_file() and (cwd / "problems" / benchmark / case).is_dir()):
        raise SystemExit(f"run from the root of the Jexpresso clone: no problems/{benchmark}/{case} under {cwd}")


def run_case(jl, benchmark: str, case: str) -> None:
    t0 = time.perf_counter()
    jl.seval(f'Jexpresso.run_case("{benchmark}", "{case}")')
    jl.seval("flush(stdout)")
    say("run_case_seconds", f"{time.perf_counter() - t0:.2f}")
    say("run_case", benchmark, case, "returned")


def process_threads() -> object:
    """The number of operating-system threads in this process, where psutil can read it."""
    try:
        import psutil

        return psutil.Process().num_threads()
    except Exception as exc:
        return f"unread:{type(exc).__name__}"


# ----------------------------------------------------------------- coexist


def network_step(tag: str) -> None:
    """One optimizer step of a small network on the CPU, and one LAPACK call; the losses printed bit for bit."""
    import torch

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(64, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 4),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(4096, 64)
    y = torch.randn(4096, 4)
    loss0 = torch.nn.functional.mse_loss(net(x), y)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    loss1 = torch.nn.functional.mse_loss(net(x), y)
    r = torch.linalg.qr(torch.randn(200, 200, dtype=torch.float64)).R
    qr = float(r.diagonal().abs().sum())
    say("network", tag, "loss0", float(loss0).hex(), "loss1", float(loss1).hex(), "qr", qr.hex())
    say("network_decimal", tag, float(loss0), float(loss1), qr, "threads", torch.get_num_threads())


def mode_coexist(jl, args, checks: Checks) -> None:
    b03 = load_b03(jl) if jl is not None else None
    with_torch = "torch" in sys.modules
    say("threads", "at_start", process_threads())
    if b03 is not None:
        emit(b03.runtime_text())
        checks.check("C2", "julia_before", int(b03.add(20, 22)) == 42)
    if with_torch:
        network_step("first")
        say("threads", "after_network", process_threads())
    if b03 is not None:
        checks.check("C2", "julia_after", int(b03.add(40, 2)) == 42)
        empty = b03.empty
        for _ in range(1000):
            empty()
        say("empty_calls", 1000)
    if args.jexpresso or args.case:
        use_jexpresso(jl)
        say("threads", "after_using_jexpresso", process_threads())
    if args.case:
        run_case(jl, *args.case)
        if with_torch:
            network_step("after_case")
        checks.check("C3", "julia_after_case", int(b03.add(1, 41)) == 42)
    if b03 is not None:
        emit(b03.mpi_text())
        emit(b03.libs_text())
    say("threads", "at_end", process_threads(), "python_threads", threading.active_count())


# ------------------------------------------------------------ mpi, episodes


def mode_mpi(jl) -> None:
    b03 = load_b03(jl)
    emit(b03.runtime_text())
    say("phase", "before_using")
    emit(b03.mpi_text())
    use_jexpresso(jl)
    say("phase", "after_using")
    emit(b03.mpi_text())
    emit(b03.libs_text())


def mode_episodes(jl, benchmark: str, case: str, n: int) -> None:
    b03 = load_b03(jl)
    emit(b03.runtime_text())
    use_jexpresso(jl)
    say("phase", "after_using")
    emit(b03.mpi_text())
    for i in range(1, n + 1):
        say("episode", i, "begin")
        run_case(jl, benchmark, case)
        say("episode", i, "returned")
        say("phase", f"after_episode_{i}")
        emit(b03.mpi_text())
    emit(b03.libs_text())
    say("threads", "at_end", process_threads())


# ------------------------------------------------------- spin, crash, thread


def mode_spin(jl, where: str, seconds: float) -> None:
    """Keep the processor for `seconds`, in Julia or in Python, and say how the wait ended."""
    spin = None
    if where == "julia":
        b03 = load_b03(jl)
        emit(b03.runtime_text())
        spin = b03.spin
        spin(0.01)
    say("spin_where", where, seconds)
    say("ready", os.getpid())
    t0 = time.perf_counter()
    try:
        if spin is not None:
            spin(seconds)
        else:
            time.sleep(seconds)
        say("spin_returned", f"{time.perf_counter() - t0:.2f}")
    except BaseException as exc:
        say("spin_raised", type(exc).__name__, f"{time.perf_counter() - t0:.2f}", first_line(exc))
        raise
    say("after_spin", f"{time.perf_counter() - t0:.2f}")


def mode_crash(jl, kind: str, checks: Checks) -> None:
    import juliacall

    cwd = Path.cwd()
    if (cwd / "Project.toml").exists() or (cwd / "problems").exists():
        raise SystemExit(f"crash children do not run from a project directory: {cwd}")
    b03 = load_b03(jl)
    emit(b03.runtime_text())
    say("faulthandler_enabled", faulthandler.is_enabled())
    say("ready", os.getpid())
    say("crash", kind, "begin")
    try:
        r = b03.crash(kind)
        say("crash", kind, "returned", repr(r))
    except Exception as exc:
        text = str(exc)
        say("crash", kind, "raised", f"{type(exc).__module__}.{type(exc).__name__}", first_line(text, 200))
        if kind in THROWN:
            checks.check("K1", f"{kind}:is_julia_error", isinstance(exc, juliacall.JuliaError), type(exc).__name__)
            checks.check("K1", f"{kind}:carries_message", THROWN[kind] in text)
    if kind in THROWN:
        checks.check("K1", f"{kind}:julia_usable_after", int(b03.add(1, 1)) == 2)
        emit(b03.mpi_text())
    say("after_crash", kind, "process_alive")


def mode_thread(b03) -> None:
    """Calls into Julia from one thread other than the main one, then from two at once. Recorded, not declared."""
    add = b03.add
    results: dict[str, tuple] = {}

    def work(name: str) -> None:
        try:
            total = 0
            for i in range(THREAD_CALLS):
                total += int(add(i, 1))
            results[name] = ("ok", total, threading.get_ident())
        except Exception as exc:
            results[name] = ("raised", type(exc).__name__, first_line(exc))

    want = sum(i + 1 for i in range(THREAD_CALLS))
    say("record", "thread", "main_ident", threading.get_ident(), "want", want)
    emit(b03.runtime_text())
    say("threads", "before", process_threads())
    one = threading.Thread(target=work, args=("one",))
    one.start()
    one.join()
    say("record", "thread", "one", *results.get("one", ("no_result",)))
    pair = [threading.Thread(target=work, args=(name,)) for name in ("two_a", "two_b")]
    for t in pair:
        t.start()
    for t in pair:
        t.join()
    say("record", "thread", "two_a", *results.get("two_a", ("no_result",)))
    say("record", "thread", "two_b", *results.get("two_b", ("no_result",)))
    say("record", "thread", "main_after", int(add(1, 1)))
    say("threads", "after", process_threads())


# ------------------------------------------------------------------- floor


def mode_floor(jl, b03, stages: list[str], cfg: Config) -> None:
    """The empty call and one integer call, timed as each stage enters the process. Recorded, not declared."""
    empty = b03.empty
    add = b03.add
    say("columns", *COLUMNS)
    done = ["bare"]

    def cells() -> None:
        tag = "+".join(done)
        say("threads", tag, process_threads())
        cell(f"empty_{tag}", empty, cfg)
        cell(f"add_{tag}", lambda: add(1, 2), cfg)

    cells()
    for stage in stages:
        if stage == "mpi":
            b03.mpi_init()
        elif stage == "jexpresso":
            use_jexpresso(jl)
        elif stage == "numpy":
            import numpy

            say("numpy_version", numpy.__version__)
        elif stage == "torch":
            import torch

            say("torch_version", torch.__version__)
            network_step("floor")
        done.append(stage)
        cells()
    emit(b03.mpi_text())


# -------------------------------------------------------------------- main


def finish(mode: str, order: list[str], checks: Checks | None) -> None:
    """The lines of signal C1 that close every mode."""
    say("import_order", ",".join(order) if order else "<none>")
    say("modules_at_end", *(f"{m}={m in sys.modules}" for m in WATCHED))
    failures = checks.failures if checks is not None else None
    if "juliacall" in sys.modules:
        import juliacall

        version_ok = juliacall.__version__ == EXPECTED_JULIACALL
        juliapkg_absent = "juliapkg" not in sys.modules
        say("check", "C1", "juliacall_version", "OK" if version_ok else "FAIL", juliacall.__version__)
        say("check", "C1", "juliapkg_never_imported", "OK" if juliapkg_absent else "FAIL", juliapkg_absent)
        if failures is not None:
            failures += (not version_ok) + (not juliapkg_absent)
    if failures is not None:
        say("result", mode, "OK" if failures == 0 else "FAIL", failures)
    say("mode_done", mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("coexist", "mpi", "episodes", "spin", "crash", "thread", "floor"))
    parser.add_argument("--imports", default="juliacall,numpy", help="comma-separated, in the order to import")
    parser.add_argument("--case", nargs=2, metavar=("BENCHMARK", "CASE"))
    parser.add_argument("--jexpresso", action="store_true", help="coexist: `using Jexpresso` without a case")
    parser.add_argument("--n", type=int, default=2, help="episodes: how many run_case calls")
    parser.add_argument("--where", choices=("julia", "python"), default="julia")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--faulthandler", action="store_true", help="enable Python's faulthandler before any import")
    parser.add_argument("--stages", default="mpi,jexpresso,numpy,torch", help="floor: comma-separated, in order")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--target-ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    order = [name for name in args.imports.split(",") if name]
    unknown = [name for name in order if name not in IMPORTABLE]
    if unknown or len(set(order)) != len(order):
        raise SystemExit(f"--imports takes each of {', '.join(IMPORTABLE)} at most once; got {args.imports!r}")
    stages = [name for name in args.stages.split(",") if name]
    if [name for name in stages if name not in STAGES] or len(set(stages)) != len(stages):
        raise SystemExit(f"--stages takes each of {', '.join(STAGES)} at most once; got {args.stages!r}")
    with_julia = "juliacall" in order
    needs_julia = args.mode in ("mpi", "episodes", "crash", "thread", "floor") or (
        args.mode == "spin" and args.where == "julia"
    )
    if needs_julia and not with_julia:
        raise SystemExit(f"mode {args.mode} needs juliacall in --imports")
    if args.mode == "floor" and order != ["juliacall"]:
        raise SystemExit("mode floor takes --imports juliacall; numpy and torch enter as stages")
    if args.mode == "crash" and args.kind is None:
        raise SystemExit("mode crash needs --kind")
    if args.mode == "episodes" and args.case is None:
        raise SystemExit("mode episodes needs --case BENCHMARK CASE")
    if args.case and (args.mode not in ("coexist", "episodes") or not with_julia):
        raise SystemExit("--case is for coexist and episodes, with juliacall imported")
    if args.jexpresso and not with_julia:
        raise SystemExit("--jexpresso needs juliacall in --imports")
    if args.case:
        check_case(*args.case)

    if args.faulthandler:
        faulthandler.enable()
    say("mode", args.mode, "imports", args.imports or "<none>", "pid", os.getpid())
    say("python", sys.version.split()[0], sys.executable)
    say("cwd", Path.cwd())
    jl = import_in_order(order)

    checks: Checks | None = None
    if args.mode == "coexist":
        checks = Checks()
        mode_coexist(jl, args, checks)
    elif args.mode == "mpi":
        mode_mpi(jl)
    elif args.mode == "episodes":
        mode_episodes(jl, *args.case, args.n)
    elif args.mode == "spin":
        mode_spin(jl, args.where, args.seconds)
    elif args.mode == "crash":
        checks = Checks()
        mode_crash(jl, args.kind, checks)
    elif args.mode == "thread":
        mode_thread(load_b03(jl))
    else:
        b03 = load_b03(jl)
        cfg = Config(target_ns=int(args.target_ms * 1_000_000), repeats=args.repeats, gcsnap=b03.gc_snapshot)
        mode_floor(jl, b03, stages, cfg)
    finish(args.mode, order, checks)
    return 0 if checks is None or checks.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
