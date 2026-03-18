"""
compare_results.py
------------------
Vẽ biểu đồ so sánh:
  - Baseline (fixed-time) từ baseline_results/
  - RL agent         từ model/<run_name>/

Ba biểu đồ:
  1. Cumulative reward per episode
  2. Cumulative waiting time per episode
  3. Average queue length per episode

Cách dùng:
    python compare_results.py --baseline baseline_results --rl model/run-01
    python compare_results.py --baseline baseline_results --rl model/run-01 --out comparison/
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # chạy không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Màu sắc ────────────────────────────────────────────────────────────
COLOR_BASELINE = "#5B8DB8"   # xanh dương nhạt
COLOR_RL       = "#E07B4F"   # cam
COLOR_BG       = "#F8F8F8"
COLOR_GRID     = "#E0E0E0"


def _moving_avg(data, window=10):
    """Rolling average để làm mượt đường."""
    if len(data) < window:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_baseline(baseline_dir: str) -> dict:
    path = os.path.join(baseline_dir, "baseline_results.json")
    if not os.path.exists(path):
        # fallback: đọc từ txt files
        r = np.loadtxt(os.path.join(baseline_dir, "baseline_reward_data.txt")).tolist()
        w = np.loadtxt(os.path.join(baseline_dir, "baseline_wait_data.txt")).tolist()
        q = np.loadtxt(os.path.join(baseline_dir, "baseline_queue_data.txt")).tolist()
        return {"cumulative_rewards": r, "cumulative_waits": w, "avg_queues": q}
    with open(path) as f:
        return json.load(f)


def load_rl(rl_dir: str) -> dict:
    """
    Đọc kết quả RL từ thư mục model run.
    Hỗ trợ cả format txt (plot_*_data.txt) của repo gốc.
    """
    result = {}

    reward_txt = os.path.join(rl_dir, "plot_reward_data.txt")
    queue_txt  = os.path.join(rl_dir, "plot_queue_data.txt")
    delay_txt  = os.path.join(rl_dir, "plot_delay_data.txt")

    if os.path.exists(reward_txt):
        result["cumulative_rewards"] = np.loadtxt(reward_txt).tolist()
    if os.path.exists(queue_txt):
        result["avg_queues"] = np.loadtxt(queue_txt).tolist()
    if os.path.exists(delay_txt):
        result["cumulative_waits"] = np.loadtxt(delay_txt).tolist()

    # Nếu không có delay, dùng reward làm proxy
    if "cumulative_waits" not in result and "cumulative_rewards" in result:
        result["cumulative_waits"] = [-r for r in result["cumulative_rewards"]]

    return result


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(COLOR_BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparison(baseline: dict, rl: dict, out_dir: str, smooth_window: int = 10):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Baseline vs. DQN Agent", fontsize=15, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")

    datasets = [
        {
            "ax":     axes[0],
            "title":  "Cumulative Reward per Episode",
            "ylabel": "Cumulative reward",
            "b_key":  "cumulative_rewards",
            "rl_key": "cumulative_rewards",
            "higher_better": True,
        },
        {
            "ax":     axes[1],
            "title":  "Cumulative Waiting Time per Episode",
            "ylabel": "Total wait (seconds)",
            "b_key":  "cumulative_waits",
            "rl_key": "cumulative_waits",
            "higher_better": False,
        },
        {
            "ax":     axes[2],
            "title":  "Average Queue Length per Episode",
            "ylabel": "Avg queue (vehicles)",
            "b_key":  "avg_queues",
            "rl_key": "avg_queues",
            "higher_better": False,
        },
    ]

    for ds in datasets:
        ax   = ds["ax"]
        b_data  = np.array(baseline.get(ds["b_key"],  []))
        rl_data = np.array(rl.get(ds["rl_key"], []))

        n_b  = len(b_data)
        n_rl = len(rl_data)

        _setup_ax(ax, ds["title"], "Episode", ds["ylabel"])

        if n_b > 0:
            ax.plot(b_data, color=COLOR_BASELINE, alpha=0.25, linewidth=0.8)
            if n_b >= smooth_window:
                sm = _moving_avg(b_data, smooth_window)
                x  = np.arange(smooth_window - 1, n_b)
                ax.plot(x, sm, color=COLOR_BASELINE, linewidth=2.2,
                        label=f"Baseline (MA{smooth_window})")
            else:
                ax.plot(b_data, color=COLOR_BASELINE, linewidth=2.2, label="Baseline")

        if n_rl > 0:
            ax.plot(rl_data, color=COLOR_RL, alpha=0.25, linewidth=0.8)
            if n_rl >= smooth_window:
                sm = _moving_avg(rl_data, smooth_window)
                x  = np.arange(smooth_window - 1, n_rl)
                ax.plot(x, sm, color=COLOR_RL, linewidth=2.2,
                        label=f"DQN Agent (MA{smooth_window})")
            else:
                ax.plot(rl_data, color=COLOR_RL, linewidth=2.2, label="DQN Agent")

        ax.legend(fontsize=9)

        # Annotation: improvement ở episode cuối
        if n_b > 0 and n_rl > 0:
            last_b  = float(np.mean(b_data[-10:]))
            last_rl = float(np.mean(rl_data[-10:]))
            if last_b != 0:
                if ds["higher_better"]:
                    pct = (last_rl - last_b) / abs(last_b) * 100
                    sign = "+" if pct > 0 else ""
                else:
                    pct = (last_b - last_rl) / abs(last_b) * 100
                    sign = "+" if pct > 0 else ""
                ax.annotate(
                    f"RL {sign}{pct:.1f}%\nvs baseline\n(last 10 ep)",
                    xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8.5,
                    color=COLOR_RL,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLOR_RL, alpha=0.8),
                )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'DQN Agent':>12} {'Δ':>8}")
    print("-" * 60)

    def _last_mean(arr, n=10):
        if len(arr) == 0:
            return float("nan")
        return float(np.mean(arr[-n:]))

    for label, b_key, rl_key, higher in [
        ("Cumulative reward (last 10)", "cumulative_rewards", "cumulative_rewards", True),
        ("Cumulative wait  (last 10)", "cumulative_waits",   "cumulative_waits",   False),
        ("Avg queue        (last 10)", "avg_queues",         "avg_queues",         False),
    ]:
        bv  = _last_mean(baseline.get(b_key,  []))
        rlv = _last_mean(rl.get(rl_key, []))
        if not np.isnan(bv) and not np.isnan(rlv) and bv != 0:
            delta_pct = (rlv - bv) / abs(bv) * 100
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "N/A"
        print(f"  {label:<28} {bv:>12.1f} {rlv:>12.1f} {delta_str:>8}")

    print("=" * 60)
    print("(+) = RL higher than baseline  |  (-) = RL lower than baseline")
    print("For reward: higher is better.  For wait/queue: lower is better.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs RL results")
    parser.add_argument("--baseline", default="baseline_results",
                        help="Dir chứa baseline_results.json (từ run_baseline.py)")
    parser.add_argument("--rl",       default="model/run-01",
                        help="Dir chứa plot_*_data.txt của RL run")
    parser.add_argument("--out",      default="comparison",
                        help="Output dir cho biểu đồ")
    parser.add_argument("--smooth",   type=int, default=10,
                        help="Window size cho moving average (default: 10)")
    args = parser.parse_args()

    print(f"Loading baseline from : {args.baseline}")
    baseline = load_baseline(args.baseline)
    print(f"Loading RL results from: {args.rl}")
    rl = load_rl(args.rl)

    plot_comparison(baseline, rl, args.out, smooth_window=args.smooth)
