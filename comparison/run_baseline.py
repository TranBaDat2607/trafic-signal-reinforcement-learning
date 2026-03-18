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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from constants import INCOMING_EDGES
from environment.generator import generate_routefile


def load_settings(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cumulated_waiting_time() -> float:
    """Tổng giây chờ tích lũy của tất cả xe trên incoming lanes."""
    total = 0.0
    for car_id in traci.vehicle.getIDList():
        if traci.vehicle.getRoadID(car_id) in INCOMING_EDGES:
            total += traci.vehicle.getAccumulatedWaitingTime(car_id)
    return total


def get_queue_length() -> int:
    """Số xe đang đứng yên trên incoming lanes."""
    return sum(
        traci.edge.getLastStepHaltingNumber(edge)
        for edge in INCOMING_EDGES
    )


def run_one_episode(sumocfg_file: Path, max_steps: int, gui: bool) -> dict:
    """Chạy 1 episode — SUMO tự điều khiển đèn, Python chỉ đo."""
    binary = checkBinary("sumo-gui" if gui else "sumo")

    if traci.isLoaded():
        traci.close()

    traci.start([
        binary,
        "-c", str(sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps),
    ])

    cumulative_wait   = 0.0
    cumulative_reward = 0.0
    queue_per_step    = []
    old_wait          = 0.0
    step              = 0

    while step < max_steps:
        traci.simulationStep()   # SUMO tự chạy đèn theo tlLogic
        step += 1

        current_wait      = get_cumulated_waiting_time()
        reward            = old_wait - current_wait
        cumulative_reward += reward
        cumulative_wait   += current_wait
        old_wait           = current_wait
        queue_per_step.append(get_queue_length())

    traci.close()

    return {
        "cumulative_reward": cumulative_reward,
        "cumulative_wait":   cumulative_wait,
        "avg_queue":         float(np.mean(queue_per_step)),
    }


def run_baseline(config_path: str, out_dir: str, n_episodes=None, gui: bool = False):
    cfg = load_settings(config_path)

    total_episodes = n_episodes or cfg["total_episodes"]
    max_steps      = cfg["max_steps"]
    n_cars         = cfg["n_cars_generated"]
    turn_chance    = cfg.get("turn_chance", 0.25)
    sumocfg_file   = Path(cfg.get("sumocfg_file", "intersection/sumo_config.sumocfg"))

    os.makedirs(out_dir, exist_ok=True)

    all_rewards = []
    all_waits   = []
    all_queues  = []

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

        all_rewards.append(result["cumulative_reward"])
        all_waits.append(result["cumulative_wait"])
        all_queues.append(result["avg_queue"])

        print(
            f"  Ep {ep+1:3d}/{total_episodes}"
            f"  reward={result['cumulative_reward']:9.1f}"
            f"  wait={result['cumulative_wait']:9.1f}"
            f"  queue={result['avg_queue']:.2f}"
        )

    # Lưu kết quả
    results = {
        "mode":               "baseline_sumo_native",
        "total_episodes":     total_episodes,
        "cumulative_rewards": all_rewards,
        "cumulative_waits":   all_waits,
        "avg_queues":         all_queues,
    }
    with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savetxt(os.path.join(out_dir, "baseline_reward_data.txt"), all_rewards)
    np.savetxt(os.path.join(out_dir, "baseline_wait_data.txt"),   all_waits)
    np.savetxt(os.path.join(out_dir, "baseline_queue_data.txt"),  all_queues)

    print(f"\nSaved to: {out_dir}/")
    print(f"  Mean reward : {np.mean(all_rewards):.1f}")
    print(f"  Mean wait   : {np.mean(all_waits):.1f} s")
    print(f"  Mean queue  : {np.mean(all_queues):.2f} vehicles")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="settings/training_settings.yaml")
    parser.add_argument("--out",      default="baseline_results")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gui",      action="store_true")
    args = parser.parse_args()
    run_baseline(args.config, args.out, args.episodes, args.gui)
