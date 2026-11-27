"""Compare two Julia manifests by package UUID.

Bridge roadmap B0.1, signal S2. juliacall runs with Jexpresso's project active
and takes PythonCall from a second, one-package environment on the load path.
PythonCall is loaded first, so a package that both manifests hold is loaded at
the second environment's version. This script lists every package the two
manifests share and says whether both name the same version and the same
source tree. It reads the two files and writes nothing.

Usage, from the root of drl-amr:

    python tools/jexpresso/bridge_b0/manifest_overlap.py REFERENCE SIDE

REFERENCE is the manifest that must not change (Jexpresso's). SIDE is the
manifest of the environment that holds PythonCall. The exit status is 0 when
no shared package differs, the two manifests were written by the same Julia
version, and every required package (PythonCall unless --require says
otherwise) is in SIDE; it is 1 otherwise. A SIDE that is still a bare copy of
REFERENCE therefore fails, although no shared package differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomllib

FORMATS = ("2.0", "2.1")
FIELDS = ("version", "tree", "source")


def load_manifest(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    """Return the Julia version and {uuid: entry} of a format-2 manifest.

    An entry is the package's name, version, git tree hash and source (a
    repository URL or a path; empty for a registered package). A field the
    manifest leaves out is the empty string.
    """
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    fmt = str(data.get("manifest_format", ""))
    if fmt not in FORMATS:
        raise SystemExit(f"{path}: manifest_format {fmt!r} is not one of {FORMATS}")
    entries: dict[str, dict[str, str]] = {}
    for name, records in data.get("deps", {}).items():
        for record in records:
            uuid = record.get("uuid")
            if uuid is None:
                raise SystemExit(f"{path}: the entry for {name} has no uuid")
            if uuid in entries:
                raise SystemExit(f"{path}: uuid {uuid} appears twice ({entries[uuid]['name']}, {name})")
            entries[uuid] = {
                "name": name,
                "version": str(record.get("version", "")),
                "tree": str(record.get("git-tree-sha1", "")),
                "source": str(record.get("repo-url", record.get("path", ""))),
            }
    return str(data.get("julia_version", "")), entries


def describe(entry: dict[str, str]) -> str:
    """One entry as text: the version, a short tree hash and the source."""
    parts = [entry["version"] or "(no version)"]
    if entry["tree"]:
        parts.append(entry["tree"][:12])
    if entry["source"]:
        parts.append(entry["source"])
    return " ".join(parts)


def compare(
    reference: dict[str, dict[str, str]], side: dict[str, dict[str, str]]
) -> tuple[list[str], list[str], list[str]]:
    """Return the names of shared packages that agree, that differ, and of packages only in SIDE."""
    same: list[str] = []
    differ: list[str] = []
    only_side: list[str] = []
    for uuid, entry in side.items():
        if uuid not in reference:
            only_side.append(entry["name"])
        elif reference[uuid]["name"] == entry["name"] and all(reference[uuid][f] == entry[f] for f in FIELDS):
            same.append(entry["name"])
        else:
            differ.append(entry["name"])
    return sorted(same), sorted(differ), sorted(only_side)


def by_name(entries: dict[str, dict[str, str]], name: str) -> dict[str, str] | None:
    """The entry with this package name, or None."""
    for entry in entries.values():
        if entry["name"] == name:
            return entry
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="the manifest that must not change")
    parser.add_argument("side", type=Path, help="the manifest of the environment that holds PythonCall")
    parser.add_argument(
        "--watch",
        action="append",
        default=None,
        help="a package name to report from both manifests whether shared or not (default: OpenSSL_jll, PythonCall)",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=None,
        help="a package name that SIDE must hold (default: PythonCall)",
    )
    args = parser.parse_args(argv)
    watch = args.watch if args.watch is not None else ["OpenSSL_jll", "PythonCall"]
    require = args.require if args.require is not None else ["PythonCall"]

    ref_julia, reference = load_manifest(args.reference)
    side_julia, side = load_manifest(args.side)
    same, differ, only_side = compare(reference, side)
    ref_names = {entry["name"]: entry for entry in reference.values()}
    side_names = {entry["name"]: entry for entry in side.values()}

    print(f"reference {args.reference}  julia {ref_julia}  entries {len(reference)}")
    print(f"side      {args.side}  julia {side_julia}  entries {len(side)}")
    julia_ok = ref_julia == side_julia
    print(f"julia_version {'SAME' if julia_ok else 'DIFFER'}")
    for name in differ:
        print(f"DIFFER    {name}: reference {describe(ref_names[name])} | side {describe(side_names[name])}")
    for name in same:
        print(f"SAME      {name} {describe(side_names[name])}")
    for name in only_side:
        print(f"ONLY-SIDE {name} {describe(side_names[name])}")
    for name in watch:
        ref_entry = by_name(reference, name)
        side_entry = by_name(side, name)
        ref_text = describe(ref_entry) if ref_entry else "absent"
        side_text = describe(side_entry) if side_entry else "absent"
        print(f"WATCH     {name}: reference {ref_text} | side {side_text}")
    missing = [name for name in require if name not in side_names]
    for name in missing:
        print(f"MISSING   {name}: required, and not in the side manifest")
    print(
        f"shared {len(same) + len(differ)} same {len(same)} differ {len(differ)} "
        f"only_side {len(only_side)} missing {len(missing)}"
    )
    ok = julia_ok and not differ and not missing
    print("RESULT OK" if ok else "RESULT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
