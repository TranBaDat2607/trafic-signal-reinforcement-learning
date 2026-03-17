"""Grid configuration dataclasses and builder for NxN intersection grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IntersectionConfig:
    """Configuration for a single traffic-light junction within the grid.

    Attributes:
        tl_id: SUMO traffic-light ID, e.g. ``TL_0_1``.
        incoming_edges: 4-tuple of incoming edge IDs in W, N, E, S order.
        lane_id_to_group: Maps each incoming lane ID to a lane-group index (0-7).
        neighbor_tl_ids: Maps direction strings to neighbour TL IDs or None.
    """

    tl_id: str
    incoming_edges: tuple[str, ...]
    lane_id_to_group: dict[str, int]
    neighbor_tl_ids: dict[str, str | None] = field(default_factory=dict)


@dataclass
class GridConfig:
    """Top-level configuration for an NxN intersection grid.

    Attributes:
        n: Grid dimension (number of intersections per side).
        tl_ids: All TL junction IDs in row-major order.
        intersections: Mapping from TL ID to its IntersectionConfig.
        net_file: Path to the generated SUMO net XML file.
        sumocfg_file: Path to the SUMO configuration file.
        routes_file: Path to the episode routes file.
    """

    n: int
    tl_ids: list[str]
    intersections: dict[str, IntersectionConfig]
    net_file: Path
    sumocfg_file: Path
    routes_file: Path


def _build_lane_id_to_group(r: int, c: int) -> dict[str, int]:
    """Build the lane-ID-to-group map for junction TL_{r}_{c}.

    Groups (each spanning 10 cells):
      0: W straight lanes (0-2), 1: W left-turn lane (3)
      2: N straight lanes (0-2), 3: N left-turn lane (3)
      4: E straight lanes (0-2), 5: E left-turn lane (3)
      6: S straight lanes (0-2), 7: S left-turn lane (3)
    """
    suffix = f"{r}_{c}"
    result: dict[str, int] = {}
    dirs_base = [("W", 0), ("N", 2), ("E", 4), ("S", 6)]
    for d, base in dirs_base:
        edge = f"{d}2TL_{suffix}"
        for lane in range(3):
            result[f"{edge}_{lane}"] = base
        result[f"{edge}_3"] = base + 1
    return result


def build_grid_config(
    n: int,
    spacing: int,
    net_file: Path,
    sumocfg_file: Path,
    routes_file: Path,
) -> GridConfig:
    """Build a :class:`GridConfig` for an NxN intersection grid.

    Junction TL_{r}_{c} is placed at SUMO XY = ``(c*spacing, (N-1-r)*spacing)``.
    All edge IDs and lane-group mappings are derived by formula.

    Args:
        n: Grid dimension.
        spacing: Distance (m) between adjacent junction centres.
        net_file: Path to the SUMO net XML file.
        sumocfg_file: Path to the SUMO configuration file.
        routes_file: Path to the route file.

    Returns:
        A fully populated :class:`GridConfig`.
    """
    tl_ids: list[str] = []
    intersections: dict[str, IntersectionConfig] = {}

    for r in range(n):
        for c in range(n):
            tl_id = f"TL_{r}_{c}"
            tl_ids.append(tl_id)

            incoming_edges = (
                f"W2TL_{r}_{c}",
                f"N2TL_{r}_{c}",
                f"E2TL_{r}_{c}",
                f"S2TL_{r}_{c}",
            )
            lane_id_to_group = _build_lane_id_to_group(r, c)

            neighbor_tl_ids: dict[str, str | None] = {
                "W": f"TL_{r}_{c - 1}" if c > 0 else None,
                "N": f"TL_{r - 1}_{c}" if r > 0 else None,
                "E": f"TL_{r}_{c + 1}" if c < n - 1 else None,
                "S": f"TL_{r + 1}_{c}" if r < n - 1 else None,
            }

            intersections[tl_id] = IntersectionConfig(
                tl_id=tl_id,
                incoming_edges=incoming_edges,
                lane_id_to_group=lane_id_to_group,
                neighbor_tl_ids=neighbor_tl_ids,
            )

    return GridConfig(
        n=n,
        tl_ids=tl_ids,
        intersections=intersections,
        net_file=net_file,
        sumocfg_file=sumocfg_file,
        routes_file=routes_file,
    )
