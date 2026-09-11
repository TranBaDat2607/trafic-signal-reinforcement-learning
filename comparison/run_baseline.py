"""
run_baseline.py
---------------
Chạy SUMO thuần túy — không có agent, không có Python can thiệp đèn.
SUMO tự điều khiển đèn theo tlLogic định nghĩa trong environment.net.xml.

Python chỉ làm 2 việc:
  1. Sinh file xe (generate_routefile)
  2. Chạy từng bước và thu thập metrics

Đây mới là baseline thật sự: không có bất kỳ controller nào từ Python.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import traci
import yaml
from sumolib import checkBinary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from environment.generator import generate_routefile
from environment.reward import get_queue_length
from metrics import METRICS_VERSION, avg_queue, total_delay


def load_settings(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one_episode(sumocfg_file: Path, max_steps: int, gui: bool) -> dict:
    """Chạy 1 episode — SUMO tự điều khiển đèn, Python chỉ đo.

    Metrics dùng chung định nghĩa với agent (src/metrics.py). Baseline không
    có reward: reward là tín hiệu huấn luyện, không phải thước đo chất lượng
    của bộ điều khiển. Xem PROBLEMS.md P0-1.
    """
    binary = checkBinary("sumo-gui" if gui else "sumo")

    if traci.isLoaded():
        traci.close()

    traci.start([
        binary,
        "-c", str(sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps),
    ])

    queue_per_step = []
    step           = 0

    while step < max_steps:
        traci.simulationStep()   # SUMO tự chạy đèn theo tlLogic
        step += 1
        queue_per_step.append(get_queue_length())

    traci.close()

    return {
        "total_delay": total_delay(queue_per_step),
        "avg_queue":   avg_queue(queue_per_step, max_steps),
    }


def run_baseline(config_path: str, out_dir: str, n_episodes=None, gui: bool = False):
    cfg = load_settings(config_path)

    total_episodes = n_episodes or cfg["total_episodes"]
    max_steps      = cfg["max_steps"]
    n_cars         = cfg["n_cars_generated"]
    turn_chance    = cfg.get("turn_chance", 0.25)
    sumocfg_file   = Path(cfg.get("sumocfg_file", "intersection/sumo_config.sumocfg"))

    os.makedirs(out_dir, exist_ok=True)

    all_delays = []
    all_queues = []

    print(f"Running {total_episodes} baseline episodes")
    print("Mode: SUMO native tlLogic — no Python controller, no agent\n")

    for ep in range(total_episodes):
        # Sinh xe cùng seed với RL → so sánh công bằng
        generate_routefile(
            seed=ep,
            n_cars_generated=n_cars,
            max_steps=max_steps,
            turn_chance=turn_chance,
        )

        result = run_one_episode(sumocfg_file, max_steps, gui)

        all_delays.append(result["total_delay"])
        all_queues.append(result["avg_queue"])

        print(
            f"  Ep {ep+1:3d}/{total_episodes}"
            f"  delay={result['total_delay']:9d} veh-s"
            f"  queue={result['avg_queue']:.2f}"
        )

    # Lưu kết quả
    results = {
        "metrics_version": METRICS_VERSION,
        "mode":            "baseline_sumo_native",
        "total_episodes":  total_episodes,
        "total_delays":    all_delays,
        "avg_queues":      all_queues,
    }
    with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savetxt(os.path.join(out_dir, "baseline_delay_data.txt"), all_delays)
    np.savetxt(os.path.join(out_dir, "baseline_queue_data.txt"), all_queues)

    print(f"\nSaved to: {out_dir}/")
    print(f"  Mean total delay : {np.mean(all_delays):.1f} vehicle-seconds")
    print(f"  Mean queue       : {np.mean(all_queues):.2f} vehicles")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="settings/training_settings.yaml")
    parser.add_argument("--out",      default="baseline_results")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gui",      action="store_true")
    args = parser.parse_args()
    run_baseline(args.config, args.out, args.episodes, args.gui)
