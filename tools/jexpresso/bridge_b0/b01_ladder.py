"""The rungs of bridge roadmap B0.1: Jexpresso run from Python through juliacall.

One rung per process, run from the root of the Jexpresso clone with the
`rl-amr` environment active and the variables of REQUIRED set:

    python b01_ladder.py identity
    python b01_ladder.py cfunction
    python b01_ladder.py load
    python b01_ladder.py run CompEuler theta_amr

`identity` imports juliacall and reports what Julia runtime it started.
`cfunction` runs the closure-@cfunction check inside that runtime.
`load` does `using Jexpresso` and reports every loaded module and library.
`run` evaluates `Jexpresso.run_case("<benchmark>", "<case>")` at Julia's top
level, the same expression the Julia runs pass to `julia -e`.

juliacall is imported inside a function, never at import of this module, and
only after the variables are checked: without PYTHON_JULIACALL_EXE and
PYTHON_JULIACALL_PROJECT the import would hand control to JuliaPkg, which
would pick its own Julia and build its own environment.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_JL = HERE / "b01_loaded.jl"
CFUNCTION_JL = HERE / "b01_cfunction.jl"

REQUIRED = (
    "PYTHON_JULIACALL_EXE",
    "PYTHON_JULIACALL_PROJECT",
    "JULIA_LOAD_PATH",
    "JULIA_CONDAPKG_BACKEND",
    "PYTHON_JULIACALL_STARTUP_FILE",
    "PYTHON_JULIACALL_THREADS",
)
REPORTED = REQUIRED + ("PYTHON_JULIACALL_HANDLE_SIGNALS", "PYTHON_JULIACALL_LIB", "PYTHON_JULIACALL_BINDIR")


def say(*parts: object) -> None:
    """Print one tab-separated line and flush, so Python's lines and Julia's keep their order in a log."""
    print(*parts, sep="\t", flush=True)


def start():
    """Check the variables, import juliacall, and return Julia's Main module."""
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise SystemExit("refusing to import juliacall; not set: " + ", ".join(missing))
    for name in REPORTED:
        say("env", name, os.environ.get(name, "<unset>"))
    say("python", sys.version.split()[0], sys.executable)
    say("cwd", Path.cwd())
    say("torch_imported_before", "torch" in sys.modules)
    t0 = time.perf_counter()
    import juliacall

    say("import_seconds", f"{time.perf_counter() - t0:.2f}")
    say("juliacall_version", juliacall.__version__)
    say("juliapkg_imported", "juliapkg" in sys.modules)
    prefix = os.environ.get("CONDA_PREFIX")
    say("conda_julia_env_exists", (Path(prefix) / "julia_env").exists() if prefix else "<no CONDA_PREFIX>")
    return juliacall.Main


def include(jl, path: Path) -> None:
    """Include a Julia file into Main. `Base.include` is spelled out: it does not depend on what Main exports."""
    jl.seval(f'Base.include(Main, raw"{path}")')


def report(jl) -> None:
    """Print the runtime report of b01_loaded.jl."""
    include(jl, REPORT_JL)
    jl.seval("b01_report(stdout)")


def rung_identity(jl) -> None:
    two = jl.seval("1 + 1")
    say("seval_1_plus_1", two, type(two).__name__)
    say("pythoncall_version", jl.seval("string(pkgversion(PythonCall))"))
    report(jl)


def rung_cfunction(jl) -> None:
    include(jl, CFUNCTION_JL)
    say("cfunction_rung", "returned")


def rung_load(jl) -> None:
    t0 = time.perf_counter()
    jl.seval("using Jexpresso")
    say("using_jexpresso_seconds", f"{time.perf_counter() - t0:.2f}")
    report(jl)


def rung_run(jl, benchmark: str, case: str) -> None:
    for name in (benchmark, case):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"not a plain case name: {name!r}")
    cwd = Path.cwd()
    if not ((cwd / "Project.toml").is_file() and (cwd / "problems" / benchmark / case).is_dir()):
        raise SystemExit(f"run from the root of the Jexpresso clone: no problems/{benchmark}/{case} under {cwd}")
    t0 = time.perf_counter()
    jl.seval("using Jexpresso")
    say("using_jexpresso_seconds", f"{time.perf_counter() - t0:.2f}")
    t1 = time.perf_counter()
    jl.seval(f'Jexpresso.run_case("{benchmark}", "{case}")')
    jl.seval("flush(stdout)")
    say("run_case_seconds", f"{time.perf_counter() - t1:.2f}")
    say("run_case", benchmark, case, "returned")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="rung", required=True)
    sub.add_parser("identity")
    sub.add_parser("cfunction")
    sub.add_parser("load")
    run = sub.add_parser("run")
    run.add_argument("benchmark")
    run.add_argument("case")
    args = parser.parse_args(argv)

    say("rung", args.rung)
    jl = start()
    if args.rung == "identity":
        rung_identity(jl)
    elif args.rung == "cfunction":
        rung_cfunction(jl)
    elif args.rung == "load":
        rung_load(jl)
    else:
        rung_run(jl, args.benchmark, args.case)
    say("rung_done", args.rung)
    return 0


if __name__ == "__main__":
    sys.exit(main())
