"""Bridge roadmap B0.3, shape B: Julia in a worker process, the parent talking to it over a pipe.

With the `rl-amr` environment active and the variables of REQUIRED set (the
worker inherits them):

    python b03_worker.py
    python b03_worker.py --torch-in-parent

The parent never imports juliacall. It spawns one worker, which imports
juliacall and then numpy and serves requests: nothing, one array, the nine
arrays of a state. The parent times each round trip at B0.2's element counts,
checks every array it receives against the pattern of b02_calls.jl, has the
worker abort, and starts a second one. With `--torch-in-parent` the parent
imports numpy and torch first, and the worker reports that it holds no torch.

An array crosses the pipe in two ways: pickled (`rt_pickle_`), and as its raw
bytes with the dtype and shape sent beside them (`rt_bytes_`).

The Julia functions are b02_calls.jl and b03_snippets.jl, beside this file.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

from b02_overhead import COLUMNS, Config, cell, expected

HERE = Path(__file__).resolve().parent
CALLS_JL = HERE / "b02_calls.jl"
SNIPPETS_JL = HERE / "b03_snippets.jl"

REQUIRED = (
    "PYTHON_JULIACALL_EXE",
    "PYTHON_JULIACALL_PROJECT",
    "JULIA_LOAD_PATH",
    "JULIA_CONDAPKG_BACKEND",
    "PYTHON_JULIACALL_STARTUP_FILE",
    "PYTHON_JULIACALL_THREADS",
)
NS = (16, 64, 1024, 16384)
KS = (4, 8)
TYPES = ("int32", "float64")
NINE = (
    ("int32", lambda n: (n,)),
    ("int32", lambda n: (n, 4, 2)),
    ("bool", lambda n: (n, 4)),
    ("int32", lambda n: (n, 4)),
    ("bool", lambda n: (n,)),
    ("bool", lambda n: (n,)),
    ("float64", lambda n: (n, 2)),
    ("float64", lambda n: (n, 2)),
    ("float64", lambda n: (n, 1)),
)
START_TIMEOUT = 180.0
REPLY_TIMEOUT = 60.0


def say(*parts: object) -> None:
    """Print one tab-separated line and flush."""
    print(*parts, sep="\t", flush=True)


# -------------------------------------------------------------- the worker


def serve(conn) -> None:
    """The worker: import juliacall, then numpy, and answer requests until told to stop."""
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        conn.send(("refused", "not set: " + ", ".join(missing)))
        return
    t0 = time.perf_counter()
    import juliacall
    import numpy as np

    jl = juliacall.Main
    jl.seval(f'Base.include(Main, raw"{CALLS_JL}")')
    jl.seval(f'Base.include(Main, raw"{SNIPPETS_JL}")')
    b02, b03 = jl.B02, jl.B03
    getters: dict[tuple, object] = {}
    nines: dict[int, object] = {}
    conn.send(
        (
            "ready",
            os.getpid(),
            juliacall.__version__,
            "torch" in sys.modules,
            "juliapkg" in sys.modules,
            round(time.perf_counter() - t0, 2),
        )
    )
    while True:
        request = conn.recv()
        op = request[0]
        if op == "empty":
            conn.send(None)
        elif op == "add":
            conn.send(int(b03.add(request[1], request[2])))
        elif op in ("pickle", "bytes"):
            key = request[1:]
            getter = getters.get(key)
            if getter is None:
                tname, n, k = key
                getter = getters[key] = b02.getter(tname, False, n, k)
            view = np.asarray(getter())
            if op == "pickle":
                conn.send(view)
            else:
                conn.send_bytes(view.T)
        elif op == "nine":
            n = request[1]
            getter = nines.get(n)
            if getter is None:
                getter = nines[n] = b02.nine_getter(n)
            conn.send([np.asarray(a) for a in getter()])
        elif op == "crash":
            b03.crash(request[1])
            conn.send(("crash_returned", request[1]))
        elif op == "stop":
            conn.send("bye")
            return
        else:
            conn.send(("unknown", op))


# -------------------------------------------------------------- the parent


def ask(conn, request: tuple, timeout: float = REPLY_TIMEOUT):
    conn.send(request)
    if not conn.poll(timeout):
        raise TimeoutError(f"no reply to {request[0]} in {timeout} s")
    return conn.recv()


def start_worker(ctx):
    """Spawn a worker and wait for its `ready` message."""
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=serve, args=(child_conn,))
    t0 = time.perf_counter()
    proc.start()
    child_conn.close()
    if not parent_conn.poll(START_TIMEOUT):
        proc.kill()
        raise SystemExit(f"the worker sent nothing in {START_TIMEOUT} s")
    message = parent_conn.recv()
    if message[0] != "ready":
        raise SystemExit(f"the worker did not start: {message}")
    say("worker_ready", *message[1:], "seconds_from_start", f"{time.perf_counter() - t0:.2f}")
    return proc, parent_conn, message


def count(failures: list[int], signal: str, name: str, ok: bool, detail: object = "") -> None:
    ok = bool(ok)
    if not ok:
        failures[0] += 1
    say("check", signal, name, "OK" if ok else "FAIL", detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--torch-in-parent", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--target-ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise SystemExit("the worker would refuse to import juliacall; not set: " + ", ".join(missing))
    say("mode", "worker", "torch_in_parent", args.torch_in_parent, "pid", os.getpid())
    say("python", sys.version.split()[0], sys.executable)
    say("cwd", Path.cwd())

    import numpy as np

    if args.torch_in_parent:
        import torch

        say("parent_torch", torch.__version__, "threads", torch.get_num_threads())
    failures = [0]
    cfg = Config(target_ns=int(args.target_ms * 1_000_000), repeats=args.repeats)
    ctx = multiprocessing.get_context("spawn")
    proc, conn, ready = start_worker(ctx)

    count(failures, "W2", "juliacall_absent_from_parent", "juliacall" not in sys.modules)
    count(failures, "W2", "torch_absent_from_worker", ready[3] is False, ready[3])
    count(failures, "W2", "parent_torch_as_asked", ("torch" in sys.modules) == args.torch_in_parent)
    say("record", "worker", "juliacall_version", ready[2], "juliapkg_imported", ready[4])

    for tname in TYPES:
        dtype = np.dtype(tname)
        for n in NS:
            for k in KS:
                want = expected(np, tname, (n, k))
                tag = f"{tname}_{n}x{k}"
                pickled = ask(conn, ("pickle", tname, n, k))
                count(failures, "W1", f"pickle_{tag}", pickled.dtype == dtype and bool(np.array_equal(pickled, want)))
                conn.send(("bytes", tname, n, k))
                raw = np.frombuffer(conn.recv_bytes(), dtype=dtype).reshape(k, n).T
                count(failures, "W1", f"bytes_{tag}", bool(np.array_equal(raw, want)))
    for n in NS:
        nine = ask(conn, ("nine", n))
        ok = len(nine) == len(NINE) and all(
            a.dtype == np.dtype(tname) and bool(np.array_equal(a, expected(np, tname, dims(n))))
            for a, (tname, dims) in zip(nine, NINE, strict=False)
        )
        count(failures, "W1", f"nine_{n}", ok)

    say("columns", *COLUMNS)

    def round_trip(request: tuple):
        conn.send(request)
        return conn.recv()

    def round_trip_bytes(request: tuple, dtype, n: int, k: int):
        conn.send(request)
        return np.frombuffer(conn.recv_bytes(), dtype=dtype).reshape(k, n).T

    cell("rt_empty", lambda: round_trip(("empty",)), cfg)
    cell("rt_add", lambda: round_trip(("add", 1, 2)), cfg)
    for tname in TYPES:
        dtype = np.dtype(tname)
        for n in NS:
            for k in KS:
                tag = f"{tname}_{n}x{k}"
                cell(f"rt_pickle_{tag}", lambda t=tname, n=n, k=k: round_trip(("pickle", t, n, k)), cfg)
                cell(
                    f"rt_bytes_{tag}",
                    lambda t=tname, d=dtype, n=n, k=k: round_trip_bytes(("bytes", t, n, k), d, n, k),
                    cfg,
                )
    for n in NS:
        cell(f"rt_nine_pickle_{n}", lambda n=n: round_trip(("nine", n)), cfg)

    conn.send(("crash", "abort"))
    t0 = time.perf_counter()
    try:
        reply = conn.recv()
        say("record", "abort", "worker_replied", reply)
        saw_end = False
    except (EOFError, ConnectionError) as exc:
        say("record", "abort", "pipe_ended", type(exc).__name__, f"{time.perf_counter() - t0:.2f}")
        saw_end = True
    proc.join(30.0)
    say("record", "abort", "worker_exitcode", proc.exitcode, "alive", proc.is_alive())
    count(failures, "W3", "pipe_ended", saw_end)
    count(failures, "W3", "worker_ended_by_signal", proc.exitcode is not None and proc.exitcode < 0, proc.exitcode)
    conn.close()

    proc2, conn2, _ = start_worker(ctx)
    count(failures, "W3", "second_worker_adds", ask(conn2, ("add", 40, 2)) == 42)
    second = ask(conn2, ("pickle", "float64", 1024, 8))
    count(failures, "W3", "second_worker_array", bool(np.array_equal(second, expected(np, "float64", (1024, 8)))))
    say("record", "stop", ask(conn2, ("stop",)))
    proc2.join(30.0)
    say("record", "stop", "worker_exitcode", proc2.exitcode)
    count(failures, "W3", "second_worker_exit_0", proc2.exitcode == 0, proc2.exitcode)

    count(failures, "W2", "juliacall_absent_from_parent_at_end", "juliacall" not in sys.modules)
    say("result", "worker", "OK" if failures[0] == 0 else "FAIL", failures[0])
    say("mode_done", "worker")
    return 0 if failures[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
