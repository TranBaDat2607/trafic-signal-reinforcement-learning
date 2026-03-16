"""Reward and queue-length computation from TraCI vehicle data."""

from __future__ import annotations

import traci

from constants import INCOMING_EDGES


def get_cumulated_waiting_time() -> float:
    """Compute the sum of accumulated waiting times for vehicles on incoming edges.

    Returns:
        Total accumulated waiting time across all vehicles on incoming edges.
    """
    waiting_times = 0.0

    for car_id in traci.vehicle.getIDList():
        road_id = traci.vehicle.getRoadID(car_id)
        if road_id not in INCOMING_EDGES:
            continue
        wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
        waiting_times += wait_time

    return waiting_times


def get_queue_length() -> int:
    """Return the number of stopped vehicles on all incoming edges.

    Returns:
        Total number of vehicles with speed 0 on all four incoming edges.
    """
    halt_n = traci.edge.getLastStepHaltingNumber("N2TL")
    halt_s = traci.edge.getLastStepHaltingNumber("S2TL")
    halt_e = traci.edge.getLastStepHaltingNumber("E2TL")
    halt_w = traci.edge.getLastStepHaltingNumber("W2TL")
    return int(halt_n + halt_s + halt_e + halt_w)
