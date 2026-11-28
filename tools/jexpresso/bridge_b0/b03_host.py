"""The parent of bridge roadmap B0.3's child processes: start one, signal it, record how it ended.

    python b03_host.py --label abort_no --out records.txt --log abort_no.log --cwd scratch --timeout 60 -- \
        python b03_child.py crash --kind abort
    python b03_host.py --label int_julia --out records.txt --log int_julia.log --signal INT --after-ready 2 -- \
        python b03_child.py spin --where julia --seconds 10
    python b03_host.py --label kill_midsolve --out records.txt --log kill.log --signal KILL --after-frames 5 \
        --watch-dir output/AdvDiff/advection2d_dg_insolve_native --timeout 300 -- \
        python b03_child.py episodes --case AdvDiff advection2d_dg_insolve_native --n 1

The command after `--` is started as it stands (its environment is this
process's). The child's stdout goes to `--log` and its stderr to `--log` with
`.err` added. A signal is sent `--after-ready` seconds after the child prints
a line that starts with `ready`, or once a directory that appears under
`--watch-dir` holds `--after-frames` directories named `iter_*`. A child still
running at `--timeout` is killed. One record of tab-separated `host` lines is
appended to `--out` and printed.

This process never imports juliacall, and it exits 0 however the child ended.
"""

from __future__ import annotations

import argparse
import collections
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SIGNALS = {"INT": signal.SIGINT, "TERM": signal.SIGTERM, "KILL": signal.SIGKILL}
REPORTS = Path.home() / "Library" / "Logs" / "DiagnosticReports"
TAIL_LINES = 8
POLL_SECONDS = 0.05
REPORT_WAIT_SECONDS = 6.0


def report_names() -> set[str]:
    try:
        return {p.name for p in REPORTS.iterdir()}
    except OSError:
        return set()


def frame_count(directory: Path) -> int:
    try:
        return sum(1 for p in directory.iterdir() if p.is_dir() and p.name.startswith("iter_"))
    except OSError:
        return 0


def subdirectories(directory: Path) -> set[str]:
    try:
        return {p.name for p in directory.iterdir() if p.is_dir()}
    except OSError:
        return set()


def thread_count(pid: int) -> int | None:
    try:
        import psutil

        return psutil.Process(pid).num_threads()
    except Exception:
        return None


def signal_name(returncode: int | None) -> str:
    if returncode is None or returncode >= 0:
        return "-"
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def tail_of(path: Path, count: int) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-count:]
    except OSError:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path, help="the record file, appended to")
    parser.add_argument("--log", required=True, type=Path, help="the child's stdout; stderr goes beside it")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--signal", dest="signal_name", choices=tuple(SIGNALS))
    parser.add_argument("--after-ready", type=float)
    parser.add_argument("--watch-dir", type=Path)
    parser.add_argument("--after-frames", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("no command after --")
    by_frames = args.after_frames is not None
    if by_frames != (args.watch_dir is not None):
        raise SystemExit("--after-frames and --watch-dir go together")
    if (by_frames or args.after_ready is not None) and args.signal_name is None:
        raise SystemExit("--after-ready and --after-frames need --signal")
    if by_frames and args.after_ready is not None:
        raise SystemExit("one trigger: --after-ready or --after-frames")
    if args.cwd is not None and not args.cwd.is_dir():
        raise SystemExit(f"--cwd is not a directory: {args.cwd}")

    err_path = args.log.with_name(args.log.name + ".err")
    reports_before = report_names()
    watched_before = subdirectories(args.watch_dir) if by_frames else set()
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    state: dict[str, float | None] = {"ready": None}

    t0 = time.monotonic()
    with open(args.log, "w") as log_file, open(err_path, "w") as err_file:
        proc = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
            bufsize=1,
            errors="replace",
        )

        def read_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                tail.append(line.rstrip("\n"))
                if state["ready"] is None and line.split("\t", 1)[0].strip() == "ready":
                    state["ready"] = time.monotonic()

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        signal_at: float | None = None
        timed_out = False
        peak_threads: int | None = None
        new_directory = "-"
        frames_at_signal: int | None = None
        while proc.poll() is None:
            now = time.monotonic()
            count = thread_count(proc.pid)
            if count is not None:
                peak_threads = count if peak_threads is None else max(peak_threads, count)
            if args.signal_name is not None and signal_at is None:
                fire = False
                if by_frames:
                    for name in sorted(subdirectories(args.watch_dir) - watched_before):
                        frames = frame_count(args.watch_dir / name)
                        if frames >= args.after_frames:
                            fire, new_directory, frames_at_signal = True, name, frames
                            break
                else:
                    ready = state["ready"]
                    fire = ready is not None and now - ready >= (args.after_ready or 0.0)
                if fire:
                    proc.send_signal(SIGNALS[args.signal_name])
                    signal_at = time.monotonic()
            if now - t0 >= args.timeout:
                timed_out = True
                proc.kill()
                break
            time.sleep(POLL_SECONDS)
        returncode = proc.wait()
        t_exit = time.monotonic()
        reader.join(5.0)

    new_reports: list[str] = []
    if returncode < 0 and not timed_out and args.signal_name is None:
        deadline = time.monotonic() + REPORT_WAIT_SECONDS
        while time.monotonic() < deadline:
            new_reports = sorted(report_names() - reports_before)
            if new_reports:
                break
            time.sleep(0.5)
    else:
        new_reports = sorted(report_names() - reports_before)

    ready = state["ready"]
    lines = [
        ("host", "label", args.label),
        ("host", "command", " ".join(command)),
        ("host", "cwd", args.cwd if args.cwd is not None else Path.cwd()),
        ("host", "returncode", returncode, signal_name(returncode)),
        ("host", "seconds_total", f"{t_exit - t0:.2f}"),
        ("host", "ready_at", f"{ready - t0:.2f}" if ready is not None else "-"),
        ("host", "signal_sent", args.signal_name or "-", f"{signal_at - t0:.2f}" if signal_at is not None else "-"),
        ("host", "signal_to_exit_seconds", f"{t_exit - signal_at:.2f}" if signal_at is not None else "-"),
        ("host", "ready_to_exit_seconds", f"{t_exit - ready:.2f}" if ready is not None else "-"),
        ("host", "timed_out", timed_out, args.timeout),
        ("host", "peak_threads", peak_threads if peak_threads is not None else "-"),
        ("host", "new_crash_reports", len(new_reports), *new_reports),
    ]
    if by_frames:
        frames_after = frame_count(args.watch_dir / new_directory) if new_directory != "-" else "-"
        lines.append(("host", "watched_directory", new_directory, "frames_at_signal", frames_at_signal))
        lines.append(("host", "frames_after_exit", frames_after))
    lines += [("host", "stdout_tail", text) for text in tail]
    lines += [("host", "stderr_tail", text) for text in tail_of(err_path, TAIL_LINES)]
    lines.append(("host", "juliacall_in_host", "juliacall" in sys.modules))
    lines.append(("host_done", args.label))

    text = "".join("\t".join(str(part) for part in line) + "\n" for line in lines)
    with open(args.out, "a") as out:
        out.write(text)
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
