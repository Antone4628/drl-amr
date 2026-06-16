"""Diagnostics callback for multi-round DRL-AMR training (RESTRUCTURE Phase 6).

Custom Stable-Baselines3 callback that monitors training for the multi-round
sequential architecture. Tracks metrics specific to the dual reward structure,
action masking, and multiround episode dynamics.

Tracked metrics:
    - Episode returns with local/global reward decomposition
    - Action distribution over training (refine/hold/coarsen fractions)
    - Coarsening frequency and mean coarsening reward (D-020 perverse-incentive check)
    - Action mask statistics (fraction where coarsen/refine masked)
    - Resource usage at end of adaptation phases
    - PPO convergence metrics (policy loss, value loss, entropy)

Outputs:
    - TensorBoard logging throughout training
    - PDF training report at training end (7 pages)
    - JSON structured-data export for programmatic analysis

Decisions:
    D-020: Positive coarsening reward — monitor for perverse incentives
    D-025: MaskablePPO action masking — track mask statistics

Per-step data comes from the MultiroundDriver.step info dict
(drivers/multiround_driver.py): element_id, action, pre_action_error,
n_active_pre/post, n_cascade, resource_usage, r_local, r_global, reward,
transition, queue_skipped, remesh_step, round_number, episode_steps; and on
interval/done steps solver_T, solver_n_steps, solver_max_error_peak.

See also: training/train_multiround.py (the script that wires this callback).
"""
from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3.common.callbacks import BaseCallback

plt.switch_backend("Agg")  # headless backend (HPC / no display)


class MultiroundDiagnosticsCallback(BaseCallback):
    """Training diagnostics callback for multi-round DRL-AMR.

    Extracts per-step information from the environment info dict (emitted by
    MultiroundDriver.step) and the MaskablePPO action masks, and accumulates
    episode-level and training-level statistics.

    Args:
        log_dir: Directory for saving outputs (PDF, JSON). Must exist.
        log_freq: TensorBoard logging frequency in timesteps.
        verbose: Verbosity level (0=silent, 1=info).
    """

    def __init__(self, log_dir: str, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq

        self.training_start_time: float | None = None
        self.training_end_time: float | None = None

        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Reset all tracking metrics to initial state."""
        # --- Episode-level accumulators (reset each episode) ---
        self._ep_local_rewards: list[float] = []
        self._ep_global_rewards: list[float] = []
        self._ep_actions: list[str] = []
        self._ep_coarsen_rewards: list[float] = []
        self._ep_masks_coarsen: list[bool] = []
        self._ep_masks_refine: list[bool] = []
        self._ep_resource_usage: list[float] = []
        self._ep_n_cascade: list[int] = []

        # --- Training-level accumulators (grow across episodes) ---
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_mean_local: list[float] = []
        self.episode_total_global: list[float] = []
        self.episodes_completed: int = 0

        self.action_history: list[str] = []

        self.episode_coarsen_freq: list[float] = []
        self.episode_mean_coarsen_reward: list[float] = []

        self.episode_final_resource: list[float] = []

        self.episode_coarsen_masked_frac: list[float] = []
        self.episode_refine_masked_frac: list[float] = []

        self.ppo_metrics: dict[str, list] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

    # --- SB3 callback hooks -------------------------------------------------

    def _on_training_start(self) -> None:
        """Record training start time."""
        self.training_start_time = time.time()
        if self.verbose > 0:
            print("MultiroundDiagnosticsCallback: training started")
            print(f"  Log directory: {self.log_dir}")
            print(f"  TensorBoard log freq: {self.log_freq} steps")

    def _on_step(self) -> bool:
        """Accumulate per-step metrics; handle episode boundaries; log periodically."""
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        action_label = info.get("action", "hold")  # 'coarsen', 'hold', 'refine'
        r_local = info.get("r_local", 0.0)
        r_global = info.get("r_global", 0.0)

        # Action mask extraction from MaskablePPO internals (shape (n_envs, 3)).
        # Index: [0]=coarsen, [1]=hold, [2]=refine.
        coarsen_masked = True  # default: masked (conservative)
        refine_masked = True
        try:
            masks = self.locals.get("action_masks")
            if masks is not None:
                mask = masks[0] if masks.ndim > 1 else masks
                coarsen_masked = not bool(mask[0])
                refine_masked = not bool(mask[2])
        except (IndexError, AttributeError, TypeError):
            pass  # keep defaults if extraction fails

        self._ep_local_rewards.append(r_local)
        self._ep_actions.append(action_label)
        self._ep_resource_usage.append(info.get("resource_usage", 0.0))
        self._ep_n_cascade.append(info.get("n_cascade", 0))
        self._ep_masks_coarsen.append(coarsen_masked)
        self._ep_masks_refine.append(refine_masked)

        if r_global != 0.0:  # nonzero only on interval-terminal / done steps
            self._ep_global_rewards.append(r_global)

        if action_label == "coarsen":
            self._ep_coarsen_rewards.append(r_local)

        self.action_history.append(action_label)

        if done:
            self._on_episode_end(info)

        if self.num_timesteps % self.log_freq == 0:
            self._log_to_tensorboard()

        return True

    def _on_episode_end(self, info: dict[str, Any]) -> None:
        """Compute episode-level metrics and reset per-episode buffers."""
        ep_len = len(self._ep_actions)
        if ep_len == 0:
            return  # guard against empty episodes

        # Episode return — recomputed from tracked components
        # (lambda_local * sum(local) + sum(global)). lambda_local now lives on
        # the agent core; reach it through the driver (env.unwrapped.driver.core).
        # The old env.unwrapped.lambda_local path no longer exists.
        total_local = sum(self._ep_local_rewards)
        total_global = sum(self._ep_global_rewards)
        lambda_local = 0.1  # fallback if the lookup fails
        try:
            lambda_local = self.model.env.envs[0].unwrapped.driver.core.lambda_local
        except (AttributeError, IndexError):
            pass
        ep_return = lambda_local * total_local + total_global

        self.episode_returns.append(ep_return)
        self.episode_lengths.append(ep_len)

        self.episode_mean_local.append(np.mean(self._ep_local_rewards))
        self.episode_total_global.append(total_global)

        n_coarsen = self._ep_actions.count("coarsen")

        coarsen_freq = n_coarsen / ep_len if ep_len > 0 else 0.0
        self.episode_coarsen_freq.append(coarsen_freq)
        self.episode_mean_coarsen_reward.append(
            np.mean(self._ep_coarsen_rewards) if self._ep_coarsen_rewards else 0.0
        )

        self.episode_final_resource.append(
            self._ep_resource_usage[-1] if self._ep_resource_usage else 0.0
        )

        self.episode_coarsen_masked_frac.append(
            np.mean(self._ep_masks_coarsen) if self._ep_masks_coarsen else 0.0
        )
        self.episode_refine_masked_frac.append(
            np.mean(self._ep_masks_refine) if self._ep_masks_refine else 0.0
        )

        self.episodes_completed += 1

        if self.verbose > 0 and self.episodes_completed % 50 == 0:
            print(f"  Episode {self.episodes_completed}: "
                  f"return={ep_return:.2f}, len={ep_len}, "
                  f"coarsen_freq={coarsen_freq:.2%}, "
                  f"resource={self._ep_resource_usage[-1]:.2f}")

        # Reset per-episode accumulators.
        self._ep_local_rewards = []
        self._ep_global_rewards = []
        self._ep_actions = []
        self._ep_coarsen_rewards = []
        self._ep_masks_coarsen = []
        self._ep_masks_refine = []
        self._ep_resource_usage = []
        self._ep_n_cascade = []

    # --- TensorBoard logging ------------------------------------------------

    def _log_to_tensorboard(self) -> None:
        """Log windowed metrics to TensorBoard at log_freq cadence."""
        if self.logger is None:
            return

        if self.action_history:
            window = min(self.log_freq, len(self.action_history))
            recent = self.action_history[-window:]
            n = len(recent)
            self.logger.record("actions/coarsen_frac", recent.count("coarsen") / n)
            self.logger.record("actions/hold_frac", recent.count("hold") / n)
            self.logger.record("actions/refine_frac", recent.count("refine") / n)

        if self.episode_mean_local:
            k = min(20, len(self.episode_mean_local))
            self.logger.record("reward/mean_local_per_step", np.mean(self.episode_mean_local[-k:]))
            self.logger.record("reward/mean_global_per_episode", np.mean(self.episode_total_global[-k:]))
            self.logger.record("reward/mean_episode_return", np.mean(self.episode_returns[-k:]))

        if self.episode_coarsen_freq:
            k = min(20, len(self.episode_coarsen_freq))
            self.logger.record("coarsening/frequency", np.mean(self.episode_coarsen_freq[-k:]))
            self.logger.record("coarsening/mean_reward", np.mean(self.episode_mean_coarsen_reward[-k:]))

        if self.episode_final_resource:
            k = min(20, len(self.episode_final_resource))
            self.logger.record("resources/final_usage", np.mean(self.episode_final_resource[-k:]))

        if self.episode_coarsen_masked_frac:
            k = min(20, len(self.episode_coarsen_masked_frac))
            self.logger.record("masks/coarsen_masked_frac", np.mean(self.episode_coarsen_masked_frac[-k:]))
            self.logger.record("masks/refine_masked_frac", np.mean(self.episode_refine_masked_frac[-k:]))

        self._capture_ppo_metrics()

        self.logger.record("episodes/total", self.episodes_completed)
        self.logger.dump(self.num_timesteps)

    def _capture_ppo_metrics(self) -> None:
        """Snapshot PPO training metrics from SB3's internal logger."""
        try:
            if not hasattr(self.model, "logger") or self.model.logger is None:
                return

            metrics_map = self.model.logger.name_to_value
            if not metrics_map:
                return

            key_mapping = {
                "train/policy_gradient_loss": "policy_loss",
                "train/value_loss": "value_loss",
                "train/entropy_loss": "entropy",
                "train/approx_kl": "approx_kl",
                "train/clip_fraction": "clip_fraction",
            }
            for sb3_key, our_key in key_mapping.items():
                if sb3_key in metrics_map:
                    self.ppo_metrics[our_key].append((self.num_timesteps, metrics_map[sb3_key]))
        except Exception:
            pass  # silently skip if extraction fails

    # --- Training end -------------------------------------------------------

    def on_training_end(self) -> None:
        """Record end time; save JSON + PDF report. Public SB3 hook (no underscore)."""
        self.training_end_time = time.time()
        duration = self.training_end_time - self.training_start_time

        if self.verbose > 0:
            print("\nMultiroundDiagnosticsCallback: training complete")
            print(f"  Duration: {duration / 60:.1f} minutes")
            print(f"  Episodes: {self.episodes_completed}")

        self._save_structured_data()
        self._create_pdf_report()

    def _save_structured_data(self) -> None:
        """Save training metrics as JSON for programmatic analysis."""
        duration = 0.0
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time

        metrics = {
            "training": {
                "total_timesteps": self.num_timesteps,
                "episodes_completed": self.episodes_completed,
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
            },
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "episode_mean_local": self.episode_mean_local,
            "episode_total_global": self.episode_total_global,
            "episode_coarsen_freq": self.episode_coarsen_freq,
            "episode_mean_coarsen_reward": self.episode_mean_coarsen_reward,
            "episode_final_resource": self.episode_final_resource,
            "episode_coarsen_masked_frac": self.episode_coarsen_masked_frac,
            "episode_refine_masked_frac": self.episode_refine_masked_frac,
            "summary": self._compute_summary_stats(),
            "ppo_metrics": {
                key: [(int(t), float(v)) for t, v in vals]
                for key, vals in self.ppo_metrics.items()
            },
            "action_distribution": {
                "coarsen": self.action_history.count("coarsen"),
                "hold": self.action_history.count("hold"),
                "refine": self.action_history.count("refine"),
                "total": len(self.action_history),
            },
        }

        json_path = os.path.join(self.log_dir, "training_diagnostics.json")
        try:
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2, default=_json_serialize)
            if self.verbose > 0:
                print(f"  Diagnostics JSON: {json_path}")
        except Exception as e:
            print(f"Error saving diagnostics JSON: {e}")

    def _compute_summary_stats(self) -> dict:
        """Summary statistics from the last 50 episodes."""
        k = min(50, self.episodes_completed) if self.episodes_completed > 0 else 0
        if k == 0:
            return {"note": "no episodes completed"}

        return {
            "mean_return": float(np.mean(self.episode_returns[-k:])),
            "std_return": float(np.std(self.episode_returns[-k:])),
            "mean_length": float(np.mean(self.episode_lengths[-k:])),
            "mean_local_reward": float(np.mean(self.episode_mean_local[-k:])),
            "mean_global_reward": float(np.mean(self.episode_total_global[-k:])),
            "mean_coarsen_freq": float(np.mean(self.episode_coarsen_freq[-k:])),
            "mean_coarsen_reward": float(np.mean(self.episode_mean_coarsen_reward[-k:])),
            "mean_final_resource": float(np.mean(self.episode_final_resource[-k:])),
            "mean_coarsen_masked": float(np.mean(self.episode_coarsen_masked_frac[-k:])),
            "mean_refine_masked": float(np.mean(self.episode_refine_masked_frac[-k:])),
        }

    # --- PDF report ---------------------------------------------------------

    def _create_pdf_report(self) -> None:
        """Generate the 7-page PDF training report."""
        report_path = os.path.join(self.log_dir, "training_report.pdf")
        try:
            with PdfPages(report_path) as pdf:
                self._page_parameters(pdf)
                self._page_convergence(pdf)
                self._page_reward_decomposition(pdf)
                self._page_action_distribution(pdf)
                self._page_coarsening_analysis(pdf)
                self._page_resource_usage(pdf)
                self._page_mask_statistics(pdf)
            if self.verbose > 0:
                print(f"  PDF report: {report_path}")
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()

    def _page_parameters(self, pdf: PdfPages) -> None:
        """Page 1: config dump and runtime summary."""
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.axis("off")

        lines = ["MULTI-ROUND DRL-AMR TRAINING REPORT", "=" * 50, ""]

        duration = 0.0
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time

        lines.extend([
            "RUNTIME",
            f"  Total timesteps:    {self.num_timesteps:,}",
            f"  Episodes completed: {self.episodes_completed}",
            f"  Duration:           {duration / 60:.1f} min ({duration / 3600:.2f} hr)",
            f"  Throughput:         {self.num_timesteps / max(duration, 1):.0f} steps/sec",
            "",
        ])

        config_path = os.path.join(self.log_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                for section_name in ["environment", "reward", "solver", "training", "checkpointing"]:
                    section = config.get(section_name, {})
                    if section:
                        lines.append(section_name.upper())
                        for k, v in section.items():
                            lines.append(f"  {k}: {v}")
                        lines.append("")
            except Exception as e:
                lines.append(f"Could not load config: {e}")
        else:
            lines.append("No config.yaml found in results directory.")

        stats = self._compute_summary_stats()
        if "note" not in stats:
            lines.extend([
                "SUMMARY (last 50 episodes)",
                f"  Mean return:          {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}",
                f"  Mean episode length:  {stats['mean_length']:.0f}",
                f"  Mean local reward:    {stats['mean_local_reward']:.4f}",
                f"  Mean global reward:   {stats['mean_global_reward']:.2f}",
                f"  Mean coarsen freq:    {stats['mean_coarsen_freq']:.2%}",
                f"  Mean final resource:  {stats['mean_final_resource']:.2f}",
                f"  Mean coarsen masked:  {stats['mean_coarsen_masked']:.2%}",
                f"  Mean refine masked:   {stats['mean_refine_masked']:.2%}",
            ])

        lines.extend(["", f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, verticalalignment="top", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_convergence(self, pdf: PdfPages) -> None:
        """Page 2: PPO losses, entropy, episode return."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Convergence Metrics", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        if self.ppo_metrics["policy_loss"]:
            ts, vals = zip(*self.ppo_metrics["policy_loss"], strict=False)
            ax.plot(ts, vals, "b-", alpha=0.7, linewidth=1)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if self.ppo_metrics["value_loss"]:
            ts, vals = zip(*self.ppo_metrics["value_loss"], strict=False)
            ax.plot(ts, vals, "r-", alpha=0.7, linewidth=1)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Value Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if self.ppo_metrics["entropy"]:
            ts, vals = zip(*self.ppo_metrics["entropy"], strict=False)
            ax.plot(ts, vals, "g-", alpha=0.7, linewidth=1)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if self.episode_returns:
            episodes = range(1, len(self.episode_returns) + 1)
            ax.plot(episodes, self.episode_returns, "b-", alpha=0.4, linewidth=1)
            window = min(50, max(1, len(self.episode_returns) // 10))
            if window > 1 and len(self.episode_returns) >= window:
                smoothed = np.convolve(self.episode_returns, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_returns) + 1), smoothed,
                        "r-", linewidth=2, label=f"{window}-ep moving avg")
                ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.set_title("Episode Return")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_reward_decomposition(self, pdf: PdfPages) -> None:
        """Page 3: local vs global reward components over training."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Reward Decomposition", fontsize=14, fontweight="bold")

        episodes = range(1, len(self.episode_returns) + 1) if self.episode_returns else []

        ax = axes[0, 0]
        if self.episode_mean_local:
            ax.plot(episodes, self.episode_mean_local, "b-", alpha=0.5, linewidth=1)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            window = min(50, max(1, len(self.episode_mean_local) // 10))
            if window > 1 and len(self.episode_mean_local) >= window:
                smoothed = np.convolve(self.episode_mean_local, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_mean_local) + 1), smoothed, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean r_local per step")
        ax.set_title("Local Shaping Reward (per step)")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if self.episode_total_global:
            ax.plot(episodes, self.episode_total_global, "g-", alpha=0.5, linewidth=1)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            window = min(50, max(1, len(self.episode_total_global) // 10))
            if window > 1 and len(self.episode_total_global) >= window:
                smoothed = np.convolve(self.episode_total_global, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_total_global) + 1), smoothed, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total r_global per episode")
        ax.set_title("Global Retrospective Reward (per episode)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if self.episode_mean_local and self.episode_total_global:
            local_contrib = [
                abs(ml * el * 0.1)  # approximate: mean * length * lambda
                for ml, el in zip(self.episode_mean_local, self.episode_lengths, strict=False)
            ]
            global_contrib = [abs(g) for g in self.episode_total_global]
            ax.plot(episodes, local_contrib, "b-", alpha=0.5, linewidth=1, label="|λ·Σ r_local|")
            ax.plot(episodes, global_contrib, "g-", alpha=0.5, linewidth=1, label="|Σ r_global|")
            ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Absolute magnitude")
        ax.set_title("Reward Component Magnitudes")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if self.episode_returns:
            ax.plot(episodes, self.episode_returns, "k-", alpha=0.5, linewidth=1)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            window = min(50, max(1, len(self.episode_returns) // 10))
            if window > 1 and len(self.episode_returns) >= window:
                smoothed = np.convolve(self.episode_returns, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_returns) + 1), smoothed, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.set_title("Combined Return (λ·Σlocal + Σglobal)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_action_distribution(self, pdf: PdfPages) -> None:
        """Page 4: action proportions over training."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Action Distribution", fontsize=14, fontweight="bold")

        ax = axes[0]
        if len(self.action_history) > 100:
            window = max(200, len(self.action_history) // 50)
            step = max(1, window // 5)
            timesteps, coarsen_pct, hold_pct, refine_pct = [], [], [], []
            for i in range(window, len(self.action_history), step):
                chunk = self.action_history[i - window:i]
                n = len(chunk)
                timesteps.append(i)
                coarsen_pct.append(chunk.count("coarsen") / n * 100)
                hold_pct.append(chunk.count("hold") / n * 100)
                refine_pct.append(chunk.count("refine") / n * 100)
            ax.plot(timesteps, refine_pct, "r-", label="Refine", linewidth=1.5)
            ax.plot(timesteps, hold_pct, "gray", label="Hold", linewidth=1.5)
            ax.plot(timesteps, coarsen_pct, "b-", label="Coarsen", linewidth=1.5)
            ax.legend()
            ax.set_ylim(0, 100)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Action %")
        ax.set_title("Action Proportions Over Training")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if self.action_history:
            counts = [
                self.action_history.count("refine"),
                self.action_history.count("hold"),
                self.action_history.count("coarsen"),
            ]
            labels = ["Refine", "Hold", "Coarsen"]
            colors = ["#e74c3c", "#95a5a6", "#3498db"]
            ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("Cumulative Action Distribution")
        else:
            ax.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_coarsening_analysis(self, pdf: PdfPages) -> None:
        """Page 5: coarsening frequency + reward (D-020 perverse-incentive check)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Coarsening Analysis (D-020 Perverse Incentive Check)",
                     fontsize=14, fontweight="bold")

        episodes = range(1, len(self.episode_coarsen_freq) + 1) if self.episode_coarsen_freq else []

        ax = axes[0]
        if self.episode_coarsen_freq:
            ax.plot(episodes, self.episode_coarsen_freq, "b-", alpha=0.5, linewidth=1)
            window = min(50, max(1, len(self.episode_coarsen_freq) // 10))
            if window > 1 and len(self.episode_coarsen_freq) >= window:
                smoothed = np.convolve(self.episode_coarsen_freq, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_coarsen_freq) + 1), smoothed, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Coarsen Fraction")
        ax.set_title("Coarsening Frequency")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if self.episode_mean_coarsen_reward:
            ax.plot(episodes, self.episode_mean_coarsen_reward, "g-", alpha=0.5, linewidth=1)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            window = min(50, max(1, len(self.episode_mean_coarsen_reward) // 10))
            if window > 1 and len(self.episode_mean_coarsen_reward) >= window:
                smoothed = np.convolve(self.episode_mean_coarsen_reward, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_mean_coarsen_reward) + 1), smoothed, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean r_local (coarsen only)")
        ax.set_title("Mean Coarsening Reward")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_resource_usage(self, pdf: PdfPages) -> None:
        """Page 6: end-of-adaptation resource usage."""
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Resource Usage", fontsize=14, fontweight="bold")

        episodes = range(1, len(self.episode_final_resource) + 1) if self.episode_final_resource else []

        if self.episode_final_resource:
            ax.plot(episodes, self.episode_final_resource, "b-", alpha=0.5, linewidth=1)
            ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Budget limit (1.0)")
            window = min(50, max(1, len(self.episode_final_resource) // 10))
            if window > 1 and len(self.episode_final_resource) >= window:
                smoothed = np.convolve(self.episode_final_resource, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_final_resource) + 1), smoothed,
                        "r-", linewidth=2, label=f"{window}-ep moving avg")
            ax.legend()
            ax.set_ylim(0, max(1.2, max(self.episode_final_resource) * 1.05))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Resource Usage (n_active / budget)")
        ax.set_title("End-of-Episode Resource Usage")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_mask_statistics(self, pdf: PdfPages) -> None:
        """Page 7: coarsen/refine masked fractions per episode."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Action Mask Statistics", fontsize=14, fontweight="bold")

        episodes = range(1, len(self.episode_coarsen_masked_frac) + 1) if self.episode_coarsen_masked_frac else []

        ax = axes[0]
        if self.episode_coarsen_masked_frac:
            ax.plot(episodes, self.episode_coarsen_masked_frac, "b-", alpha=0.5, linewidth=1)
            window = min(50, max(1, len(self.episode_coarsen_masked_frac) // 10))
            if window > 1 and len(self.episode_coarsen_masked_frac) >= window:
                smoothed = np.convolve(self.episode_coarsen_masked_frac, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_coarsen_masked_frac) + 1), smoothed, "r-", linewidth=2)
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Fraction Masked")
        ax.set_title("Coarsen Masked Fraction")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if self.episode_refine_masked_frac:
            ax.plot(episodes, self.episode_refine_masked_frac, "r-", alpha=0.5, linewidth=1)
            window = min(50, max(1, len(self.episode_refine_masked_frac) // 10))
            if window > 1 and len(self.episode_refine_masked_frac) >= window:
                smoothed = np.convolve(self.episode_refine_masked_frac, np.ones(window) / window, mode="valid")
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_refine_masked_frac) + 1), smoothed, "r-", linewidth=2)
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Fraction Masked")
        ax.set_title("Refine Masked Fraction")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _json_serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")