"""
compare_results.py
------------------
Vẽ biểu đồ so sánh:
  - Baseline (fixed-time) từ baseline_results/
  - RL agent         từ model/<run_name>/

Hai biểu đồ — chỉ vẽ những đại lượng được đo BẰNG CÙNG MỘT HÀM ở cả hai phía
(src/metrics.py):
  1. Total delay per episode  (vehicle-seconds)
  2. Average queue length per episode  (vehicles)

Reward KHÔNG được vẽ ở đây. Reward là tín hiệu huấn luyện của agent; bộ điều
khiển fixed-time không có đại lượng tương ứng, nên đặt hai đường lên cùng một
trục là so sánh hai thứ khác nhau. Xem PROBLEMS.md P0-1.

Cách dùng (chạy từ project root):
    python comparison/compare_results.py --baseline baseline_results --rl model/run-01
    python comparison/compare_results.py --baseline baseline_results --rl model/run-01 --out comparison/
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # chạy không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import AVG_QUEUE_LABEL, METRICS_VERSION, TOTAL_DELAY_LABEL


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
    """Đọc kết quả baseline. Thiếu file hoặc sai metrics_version => dừng hẳn.

    Không có fallback: một input thiếu phải làm dừng chương trình, chứ không
    được âm thầm đổi ý nghĩa của biểu đồ (PROBLEMS.md P0-3).
    """
    path = os.path.join(baseline_dir, "baseline_results.json")
    if not os.path.exists(path):
        sys.exit(
            f"Baseline results not found: {path}\n"
            f"Generate with: python comparison/run_baseline.py --out {baseline_dir}"
        )

    with open(path) as f:
        data = json.load(f)

    version = data.get("metrics_version")
    if version != METRICS_VERSION:
        sys.exit(
            f"{path} was written under metrics_version={version!r}, "
            f"but this build expects {METRICS_VERSION}.\n"
            f"Its numbers use a different definition of waiting time "
            f"(see PROBLEMS.md P0-2) and are not comparable.\n"
            f"Re-run: python comparison/run_baseline.py --out {baseline_dir}"
        )

    return {"total_delay": data["total_delays"], "avg_queue": data["avg_queues"]}


def _require_series(path: str) -> list:
    """Đọc một series bắt buộc; thiếu thì dừng và in ra đường dẫn mong đợi."""
    if not os.path.exists(path):
        sys.exit(
            f"Required RL series not found: {path}\n"
            f"It is written by src/train.py at the end of a run. "
            f"Point --rl at the directory passed to `train.py --out`."
        )
    return np.atleast_1d(np.loadtxt(path)).tolist()


def load_rl(rl_dir: str) -> dict:
    """Đọc kết quả RL từ thư mục model run (plot_*_data.txt của src/train.py).

    Chỉ đọc những series được đo giống hệt phía baseline. plot_reward_data.txt
    cố ý KHÔNG được đọc: nó là chẩn đoán huấn luyện, không phải thước đo để so
    sánh với baseline (PROBLEMS.md P0-1).
    """
    return {
        "total_delay": _require_series(os.path.join(rl_dir, "plot_delay_data.txt")),
        "avg_queue":   _require_series(os.path.join(rl_dir, "plot_queue_data.txt")),
    }


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

    n_b_eps, n_rl_eps = len(baseline["total_delay"]), len(rl["total_delay"])
    if n_b_eps != n_rl_eps:
        print(
            f"  WARNING: baseline has {n_b_eps} episodes, RL has {n_rl_eps}. "
            f"Episode n on one curve is not episode n on the other."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline vs. DQN Agent", fontsize=15, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")

    # Chỉ những đại lượng do cùng một hàm trong src/metrics.py tính ra.
    datasets = [
        {
            "ax":     axes[0],
            "title":  "Total Delay per Episode",
            "ylabel": TOTAL_DELAY_LABEL,
            "b_key":  "total_delay",
            "rl_key": "total_delay",
            "higher_better": False,
        },
        {
            "ax":     axes[1],
            "title":  "Average Queue Length per Episode",
            "ylabel": AVG_QUEUE_LABEL,
            "b_key":  "avg_queue",
            "rl_key": "avg_queue",
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

    fig.text(
        0.5, -0.04,
        "Both series computed by src/metrics.py. Total delay = sum over simulation "
        "steps of halting vehicles on incoming edges (1 s steps).\n"
        "Avg queue = total delay / max_steps — the same measurement in another unit, "
        "not independent corroboration.",
        ha="center", fontsize=8, color="#666666",
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
        ("Total delay (last 10)", "total_delay", "total_delay", False),
        ("Avg queue   (last 10)", "avg_queue",   "avg_queue",   False),
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
    print("Lower is better for both metrics.")
    print("Avg queue = total delay / max_steps: the same measurement rescaled,")
    print("so the two rows agree by construction, not by corroboration.")
    print(f"Total delay = {TOTAL_DELAY_LABEL} (sum over simulation steps of")
    print("halting vehicles on the incoming edges; 1-second steps).")


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
