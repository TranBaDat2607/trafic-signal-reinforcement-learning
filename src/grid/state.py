"""State extraction for individual intersections and neighbour-aware aggregation.

Parameterised rewrite of ``environment/state.py`` using per-junction
:class:`~grid.config.IntersectionConfig` instead of global constants.
"""

from __future__ import annotations

import numpy as np
import traci
from numpy.typing import NDArray

from constants import (
    CELLS_PER_LANE_GROUP,
    LANE_DISTANCE_TO_CELL,
    MAX_SPEED,
    MAX_WAIT_TIME,
    NUM_CELLS,
    ROAD_MAX_LENGTH,
    STATE_SIZE,
)
from grid.config import GridConfig, IntersectionConfig


def _get_lane_cell(lane_pos: float) -> int:
    """Map a lane position to a discrete cell (0 = closest to junction)."""
    lane_pos = ROAD_MAX_LENGTH - lane_pos
    lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))
    for distance, cell in LANE_DISTANCE_TO_CELL.items():
        if lane_pos <= distance:
            return cell
    msg = "Error while getting lane cell."
    raise RuntimeError(msg)


def get_intersection_state(cfg: IntersectionConfig) -> NDArray:
    """Compute the 240-element state vector for a single intersection.

    Mirrors :func:`environment.state.get_state` but uses the per-junction
    ``lane_id_to_group`` mapping from *cfg*.

    Args:
        cfg: Configuration of the target junction.

    Returns:
        NumPy array of shape ``(STATE_SIZE,)`` = ``(240,)`` with values in
        ``[0, 1]``.  Layout: ``[presence | speed_norm | wait_norm]``.
    """
    presence  = np.zeros(NUM_CELLS, dtype=float)
    speed_sum = np.zeros(NUM_CELLS, dtype=float)
    wait_sum  = np.zeros(NUM_CELLS, dtype=float)
    count     = np.zeros(NUM_CELLS, dtype=float)

    for car_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(car_id)
        lane_group = cfg.lane_id_to_group.get(lane_id)
        if lane_group is None:
            continue

        lane_pos: float = traci.vehicle.getLanePosition(car_id)
        lane_cell = _get_lane_cell(lane_pos)
        car_pos = lane_group * CELLS_PER_LANE_GROUP + lane_cell

        presence[car_pos] = 1.0
        speed_sum[car_pos] += traci.vehicle.getSpeed(car_id)
        wait_sum[car_pos]  += traci.vehicle.getWaitingTime(car_id)
        count[car_pos]     += 1.0

    denom = np.maximum(count, 1.0)
    speed_norm = np.clip(speed_sum / denom / MAX_SPEED,     0.0, 1.0)
    wait_norm  = np.clip(wait_sum  / denom / MAX_WAIT_TIME, 0.0, 1.0)

    return np.concatenate([presence, speed_norm, wait_norm])


def get_neighbor_aware_state(tl_id: str, grid_cfg: GridConfig) -> NDArray:
    """Compute a 1200-element neighbour-aware state vector for *tl_id*.

    Concatenates the 240-element state of the target junction with those of
    its four neighbours (W, N, E, S order), padding absent neighbours with
    zeros.

    Args:
        tl_id: Target junction ID.
        grid_cfg: Full grid configuration.

    Returns:
        NumPy array of shape ``(STATE_SIZE * 5,)`` = ``(1200,)``.
    """
    cfg = grid_cfg.intersections[tl_id]
    parts: list[NDArray] = [get_intersection_state(cfg)]

    for direction in ("W", "N", "E", "S"):
        nbr_id = cfg.neighbor_tl_ids.get(direction)
        if nbr_id is not None:
            nbr_cfg = grid_cfg.intersections[nbr_id]
            parts.append(get_intersection_state(nbr_cfg))
        else:
            parts.append(np.zeros(STATE_SIZE, dtype=float))

    return np.concatenate(parts)
