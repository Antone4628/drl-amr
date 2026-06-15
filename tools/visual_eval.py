"""CLI: visual evaluation of a trained multiround DRL-AMR model in the
restructured repo (RESTRUCTURE Phase 6, deployment-adapter viz).

Loads a trained model + its run config, runs a deterministic deployment rollout
(DeploymentRunner), and writes a multi-time composite snapshot (level bars +
DG-vs-exact; --no-bars drops the bars) and/or a single-panel animation
(DG-vs-exact with an info box + frame counter). DG-correct dense-polynomial
rendering throughout (analysis/deployment_viz.py).

Examples:
  python tools/visual_eval.py --model-path results/.../final_model.zip --icase 1
  python tools/visual_eval.py --model-path .../final_model.zip --icase 1,10,16
  python tools/visual_eval.py --model-path .../final_model.zip --icase 16 --no-bars --no-animate
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.deployment_viz import (  # noqa: E402
    composite_snapshot,
    save_animation,
    select_snapshots,
)
from deployment.build import (  # noqa: E402
    build_driver_from_config,
    load_config,
    load_model,
    make_exact_fn,
)
from deployment.runner import (  # noqa: E402
    DeploymentRunner,
    model_decide_fn,
    random_decide_fn,
)


def _suptitle(config: dict, icase: int) -> str:
    e = config["environment"]
    return (
        f"icase={icase}  |  indicator={e['error_indicator']}  |  "
        f"α={e['alpha']}  |  budget={e['element_budget']}  |  "
        f"max_level={config['solver']['max_level']}"
    )


def _run_one(config, model, args, icase, out_dir):
    """Run one IC's rollout and write the requested artifacts; return paths."""
    driver = build_driver_from_config(config)
    rng = np.random.default_rng(args.seed)
    decide = (random_decide_fn(rng) if args.random_policy
              else model_decide_fn(model, deterministic=True))

    runner = DeploymentRunner(
        driver,
        time_final=args.time_final,
        output_dt=args.time_final / args.n_frames,
        burnin=not args.no_burnin,
        n_burnin=args.n_burnin,
        capture_remesh=True,
    )
    snaps = runner.run(decide, icase=icase, rng=rng)
    print(
        f"icase {icase}: {len(snaps)} frames | "
        f"t in [{snaps[0].time:.3f}, {snaps[-1].time:.3f}] | "
        f"n_active {snaps[0].n_active} -> {snaps[-1].n_active}"
    )

    exact_fn = None if args.no_exact else make_exact_fn(icase)
    suptitle = _suptitle(config, icase)
    max_level = config["solver"]["max_level"]
    budget = config["environment"]["element_budget"]

    written = []
    if not args.no_snapshot:
        times = [float(t) for t in args.snapshot_times.split(",")]
        selected = select_snapshots(snaps, times)
        fig = composite_snapshot(
            selected, exact_fn=exact_fn, show_bars=not args.no_bars,
            max_level=max_level, suptitle=suptitle,
        )
        png = os.path.join(out_dir, f"snapshot_icase{icase}.png")
        pdf = os.path.join(out_dir, f"snapshot_icase{icase}.pdf")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        written += [png, pdf]
    if not args.no_animate:
        gif = os.path.join(out_dir, f"animation_icase{icase}.gif")
        save_animation(snaps, gif, fps=args.fps, exact_fn=exact_fn,
                       budget=budget, suptitle=suptitle)
        written.append(gif)
    return written


def parse_args():
    p = argparse.ArgumentParser(description="Visual eval of a trained multiround DRL-AMR model")
    p.add_argument("--model-path", required=True, help="Path to final_model.zip")
    p.add_argument("--icase", default="1", help="IC(s), comma-separated (default: 1)")
    p.add_argument("--time-final", type=float, default=1.0)
    p.add_argument("--snapshot-times", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--n-frames", type=int, default=120, help="animation frame count")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-bars", action="store_true", help="omit the level bars in the snapshot")
    p.add_argument("--no-exact", action="store_true", help="omit the exact-solution overlay")
    p.add_argument("--no-snapshot", action="store_true", help="skip the composite snapshot")
    p.add_argument("--no-animate", action="store_true", help="skip the animation")
    p.add_argument("--no-burnin", action="store_true", help="disable deployment burn-in")
    p.add_argument("--n-burnin", type=int, default=1, help="burn-in passes (default: 1)")
    p.add_argument("--random-policy", action="store_true", help="masked random actions (no model)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/visual_eval", help="output dir (gitignored)")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(os.path.join(os.path.dirname(args.model_path), "config.yaml"))
    model = None if args.random_policy else load_model(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    written = []
    for ic in (int(s) for s in str(args.icase).split(",")):
        written += _run_one(config, model, args, ic, args.output_dir)

    print("\nwrote:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()