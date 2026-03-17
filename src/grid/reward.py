"""Reward and queue-length computation for individual intersections.

Parameterised rewrite of ``environment/reward.py`` using per-junction
incoming edges from :class:`~grid.config.IntersectionConfig`.
"""

from __future__ import annotations

import traci

from grid.config import IntersectionConfig


def get_intersection_cumulated_waiting_time(cfg: IntersectionConfig) -> float:
    """Sum accumulated waiting times for vehicles on the junction's incoming edges.

    Args:
        cfg: Configuration of the target junction.

    Returns:
        Total accumulated waiting time (seconds) across all vehicles on
        the four incoming edges.
    """
    incoming = set(cfg.incoming_edges)
    total = 0.0
    for car_id in traci.vehicle.getIDList():
        if traci.vehicle.getRoadID(car_id) in incoming:
            total += float(traci.vehicle.getAccumulatedWaitingTime(car_id))
    return total


def get_intersection_queue_length(cfg: IntersectionConfig) -> int:
    """Return the number of stopped vehicles on the junction's incoming edges.

    Args:
        cfg: Configuration of the target junction.

    Returns:
        Total number of vehicles with speed ≈ 0 on all four incoming edges.
    """
    return sum(traci.edge.getLastStepHaltingNumber(e) for e in cfg.incoming_edges)
