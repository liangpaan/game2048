"""Evaluation metrics and reproducible static charts."""

from __future__ import annotations

from collections import Counter
from itertools import accumulate
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(results: list[dict]) -> dict:
    sums = [item["sum"] for item in results]
    summary = {
        "episodes": len(results),
        "mean_sum": round(statistics.mean(sums), 2),
        "median_sum": statistics.median(sums),
        "best_sum": max(sums),
        "reach_rate": round(
            sum(item["reached_256"] for item in results) / len(results), 3
        ),
        "mean_max_tile": round(statistics.mean(item["max_tile"] for item in results), 2),
        "mean_steps": round(statistics.mean(item["steps"] for item in results), 2),
        "end_reasons": dict(Counter(item["end_reason"] for item in results)),
    }
    if "td_mean_abs_error" in results[0]:
        summary["mean_td_abs_error"] = round(
            statistics.mean(item["td_mean_abs_error"] for item in results), 4
        )
    if all("merge_score" in item for item in results):
        scores = [item["merge_score"] for item in results]
        summary.update(mean_merge_score=round(statistics.mean(scores), 2),
                       median_merge_score=statistics.median(scores),
                       best_merge_score=max(scores))
    return summary


def plot_metrics(results: list[dict], histories: list[list[dict]], path: Path,
                 *, strategy: str = "2048 agent") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(results) == 1:
        result = results[0]
        history = histories[0]
        steps = [item["step"] for item in history]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
        axes[0].plot(steps, [item["sum"] for item in history], color="#2979b9")
        axes[0].set_ylabel("Sum of tiles")
        axes[1].step(steps, [item["max_tile"] for item in history],
                     where="post", color="#d97929")
        axes[1].axhline(256, color="#a33a3a", linestyle="--", label="Target 256")
        axes[1].set_ylabel("Largest tile")
        axes[1].set_xlabel("Move")
        axes[1].legend()
        if result["target_step"] is not None:
            for axis in axes:
                axis.axvline(result["target_step"], color="#39905a", linestyle=":",
                             label="First 256")
            axes[0].legend()
        fig.suptitle(
            f"{strategy} · seed {result['seed']} · final sum {result['sum']} · "
            f"{result['end_reason']}"
        )
    else:
        episodes = list(range(1, len(results) + 1))
        sums = [item["sum"] for item in results]
        maximums = [item["max_tile"] for item in results]
        lengths = [item["steps"] for item in results]
        hit_counts = list(accumulate(int(item["reached_256"]) for item in results))
        running_mean = [total / episode for episode, total
                        in zip(episodes, accumulate(sums))]
        training = ("td_mean_abs_error" in results[0]
                    or "timesteps" in results[0])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        if training:
            window = min(50, len(results))
            rolling_mean = [
                statistics.mean(sums[max(0, index - window + 1):index + 1])
                for index in range(len(sums))
            ]
            axes[0, 0].scatter(episodes, sums, s=9, alpha=0.18,
                               color="#90b6d1", label="Episode")
            axes[0, 0].plot(episodes, rolling_mean, color="#145a8a", linewidth=2,
                            label=f"{window}-episode mean")
        else:
            axes[0, 0].plot(episodes, sums, marker="o", color="#90b6d1", label="Episode")
            axes[0, 0].plot(episodes, running_mean, color="#145a8a", linewidth=2,
                            label="Running mean")
        axes[0, 0].set_ylabel("Final sum of tiles")
        axes[0, 0].legend()

        if training:
            axes[0, 1].scatter(episodes, maximums, s=9, alpha=0.25, color="#d97929")
        else:
            axes[0, 1].step(episodes, maximums, where="mid", color="#d97929")
        axes[0, 1].axhline(256, color="#a33a3a", linestyle="--", label="Target 256")
        axes[0, 1].set_ylabel("Largest tile")
        axes[0, 1].legend()

        reach_rates = [100 * hits / episode for episode, hits
                       in zip(episodes, hit_counts)]
        axes[1, 0].plot(
            episodes, reach_rates,
            marker="o", color="#39905a",
        )
        axes[1, 0].set_ylim(
            0, min(100, max(0.02, 1.2 * max(reach_rates))) if training else 100
        )
        axes[1, 0].set_ylabel("Cumulative reach rate (%)")

        if "td_mean_abs_error" in results[0]:
            errors = [item["td_mean_abs_error"] for item in results]
            window = min(50, len(results))
            rolling_error = [
                statistics.mean(errors[max(0, index - window + 1):index + 1])
                for index in range(len(errors))
            ]
            axes[1, 1].scatter(episodes, errors, s=9, alpha=0.18, color="#a798c2")
            axes[1, 1].plot(episodes, rolling_error, color="#7154a0", linewidth=2)
            axes[1, 1].set_ylabel("Mean absolute TD error")
        else:
            if training:
                window = min(50, len(results))
                rolling_lengths = [
                    statistics.mean(lengths[max(0, index - window + 1):index + 1])
                    for index in range(len(lengths))
                ]
                axes[1, 1].scatter(episodes, lengths, s=9, alpha=0.18,
                                   color="#a798c2")
                axes[1, 1].plot(episodes, rolling_lengths,
                                color="#7154a0", linewidth=2)
            else:
                axes[1, 1].plot(episodes, lengths, marker="o", color="#7154a0")
            axes[1, 1].set_ylabel("Moves until end")
        for axis in axes.flat:
            axis.set_xlabel("Episode")
            axis.set_xlim(1, len(results))
        fig.suptitle(f"3×3 2048 · {len(results)} episodes · {strategy}")

    for axis in fig.axes:
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
