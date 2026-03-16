"""State extraction: converts TraCI vehicle data into a binary state vector."""

from __future__ import annotations

import numpy as np
import traci
from numpy.typing import NDArray

from constants import (
    CELLS_PER_LANE_GROUP,
    LANE_DISTANCE_TO_CELL,
    LANE_ID_TO_GROUP,
    ROAD_MAX_LENGTH,
    STATE_SIZE,
)


def get_lane_cell(lane_pos: float) -> int:
    """Map a continuous lane position to a discrete cell index.

    The lane is inverted so that cell 0 is at the traffic light and clamped
    to [0, ROAD_MAX_LENGTH].

    Args:
        lane_pos: Distance from the start of the edge in meters.

    Returns:
        Index of the discretized cell (0-based).

    Raises:
        RuntimeError: If lane_pos does not map to any known cell bucket.
    """
    lane_pos = ROAD_MAX_LENGTH - lane_pos
    lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))

    for distance, cell in LANE_DISTANCE_TO_CELL.items():
        if lane_pos <= distance:
            return cell

    msg = "Error while getting lane cell."
    raise RuntimeError(msg)


def get_state() -> NDArray:
    """Compute the binary occupancy state vector from all active vehicles.

    Iterates over all active vehicles in the simulation, maps each one to its
    lane group and cell, and sets the corresponding position in the state vector
    to 1.

    Returns:
        A NumPy array of shape (STATE_SIZE,) with 0/1 occupancy values.

    Raises:
        ValueError: If a computed car_position is out of bounds.
    """
    state = np.zeros(STATE_SIZE, dtype=float)

    for car_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(car_id)
        lane_group = LANE_ID_TO_GROUP.get(lane_id)
        if lane_group is None:
            # Ignore cars that are not on incoming lanes.
            continue

        lane_pos: float = traci.vehicle.getLanePosition(car_id)
        lane_cell = get_lane_cell(lane_pos)

        car_position = lane_group * CELLS_PER_LANE_GROUP + lane_cell

        if car_position < 0 or car_position >= STATE_SIZE:
            msg = "Out of bounds car position."
            raise ValueError(msg)

        state[car_position] = 1.0

    return state
