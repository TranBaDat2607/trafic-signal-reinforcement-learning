"""Route file generation for NxN intersection grids.

Vehicles enter from a random border dead-end and travel straight through
the grid, exiting at the opposite border. A fraction (``turn_chance``) of
vehicles instead make one 90-degree turn at a randomly chosen junction and
exit at a perpendicular border. Departure times follow the same Weibull
distribution used by the single-intersection generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _map_to_interval(values: NDArray, new_min: int, new_max: int) -> NDArray:
    """Linearly map *values* to [new_min, new_max] (same as generator.py)."""
    old_min = float(values.min())
    old_max = float(values.max())
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def _straight_routes(n: int) -> list[tuple[str, list[str]]]:
    """Return (route_id, [edge_ids]) for all straight-through routes.

    Each route crosses all *n* junctions in one cardinal direction.
    """
    routes: list[tuple[str, list[str]]] = []

    for r in range(n):
        # West → East
        edges = [f"W2TL_{r}_0"] + [f"W2TL_{r}_{c}" for c in range(1, n)] + [f"TL_{r}_{n - 1}_2E"]
        routes.append((f"WE_r{r}", edges))

        # East → West
        edges = [f"E2TL_{r}_{n - 1}"] + [f"E2TL_{r}_{c}" for c in range(n - 2, -1, -1)] + [f"TL_{r}_0_2W"]
        routes.append((f"EW_r{r}", edges))

    for c in range(n):
        # North → South
        edges = [f"N2TL_0_{c}"] + [f"N2TL_{r}_{c}" for r in range(1, n)] + [f"TL_{n - 1}_{c}_2S"]
        routes.append((f"NS_c{c}", edges))

        # South → North
        edges = [f"S2TL_{n - 1}_{c}"] + [f"S2TL_{r}_{c}" for r in range(n - 2, -1, -1)] + [f"TL_0_{c}_2N"]
        routes.append((f"SN_c{c}", edges))

    return routes


def _turn_routes(n: int) -> list[tuple[str, list[str]]]:
    """Return (route_id, [edge_ids]) for routes that make one 90-degree turn.

    All possible (entry, turn-junction, exit-direction) combinations.
    """
    routes: list[tuple[str, list[str]]] = []

    # Enter from West at row r, travel k+1 junctions east, then turn N or S
    for r in range(n):
        for k in range(n):  # turn at column k
            # Approach segment: W2TL_{r}_0, ..., W2TL_{r}_{k}  (k+1 edges)
            if k == 0:
                approach = [f"W2TL_{r}_0"]
            else:
                approach = [f"W2TL_{r}_0"] + [f"W2TL_{r}_{cc}" for cc in range(1, k + 1)]
            # Turn north (if possible) or south
            if r > 0:
                # Turn north: exit via S2TL_{r-1}_{k} (approaching TL_{r-1}_{k} from south)
                # then exit at TL_{0}_{k}_2N
                exit_edges_n: list[str] = []
                for rr in range(r - 1, -1, -1):
                    exit_edges_n.append(f"S2TL_{rr}_{k}")
                exit_edges_n.append(f"TL_0_{k}_2N")
                routes.append((f"W_turn_N_r{r}_c{k}", approach + exit_edges_n))
            if r < n - 1:
                # Turn south
                exit_edges_s: list[str] = []
                for rr in range(r + 1, n):
                    exit_edges_s.append(f"N2TL_{rr}_{k}")
                exit_edges_s.append(f"TL_{n - 1}_{k}_2S")
                routes.append((f"W_turn_S_r{r}_c{k}", approach + exit_edges_s))

    # Enter from East at row r, travel west, turn N or S
    for r in range(n):
        for k in range(n - 1, -1, -1):  # turn at column k
            if k == n - 1:
                approach = [f"E2TL_{r}_{n - 1}"]
            else:
                approach = [f"E2TL_{r}_{n - 1}"] + [f"E2TL_{r}_{cc}" for cc in range(n - 2, k - 1, -1)]
            if r > 0:
                exit_edges_n = [f"S2TL_{rr}_{k}" for rr in range(r - 1, -1, -1)] + [f"TL_0_{k}_2N"]
                routes.append((f"E_turn_N_r{r}_c{k}", approach + exit_edges_n))
            if r < n - 1:
                exit_edges_s = [f"N2TL_{rr}_{k}" for rr in range(r + 1, n)] + [f"TL_{n - 1}_{k}_2S"]
                routes.append((f"E_turn_S_r{r}_c{k}", approach + exit_edges_s))

    # Enter from North at col c, travel south, turn W or E
    for c in range(n):
        for k in range(n):  # turn at row k
            if k == 0:
                approach = [f"N2TL_0_{c}"]
            else:
                approach = [f"N2TL_0_{c}"] + [f"N2TL_{rr}_{c}" for rr in range(1, k + 1)]
            if c > 0:
                exit_edges_w = [f"E2TL_{k}_{cc}" for cc in range(c - 1, -1, -1)] + [f"TL_{k}_0_2W"]
                routes.append((f"N_turn_W_r{k}_c{c}", approach + exit_edges_w))
            if c < n - 1:
                exit_edges_e = [f"W2TL_{k}_{cc}" for cc in range(c + 1, n)] + [f"TL_{k}_{n - 1}_2E"]
                routes.append((f"N_turn_E_r{k}_c{c}", approach + exit_edges_e))

    # Enter from South at col c, travel north, turn W or E
    for c in range(n):
        for k in range(n - 1, -1, -1):  # turn at row k
            if k == n - 1:
                approach = [f"S2TL_{n - 1}_{c}"]
            else:
                approach = [f"S2TL_{n - 1}_{c}"] + [f"S2TL_{rr}_{c}" for rr in range(n - 2, k - 1, -1)]
            if c > 0:
                exit_edges_w = [f"E2TL_{k}_{cc}" for cc in range(c - 1, -1, -1)] + [f"TL_{k}_0_2W"]
                routes.append((f"S_turn_W_r{k}_c{c}", approach + exit_edges_w))
            if c < n - 1:
                exit_edges_e = [f"W2TL_{k}_{cc}" for cc in range(c + 1, n)] + [f"TL_{k}_{n - 1}_2E"]
                routes.append((f"S_turn_E_r{k}_c{c}", approach + exit_edges_e))

    return routes


def generate_grid_routefile(
    n: int,
    out_path: Path,
    seed: int,
    n_cars: int,
    max_steps: int,
    turn_chance: float,
) -> None:
    """Generate a SUMO route file for one episode of an NxN grid simulation.

    Args:
        n: Grid dimension.
        out_path: Destination path for the ``.rou.xml`` file.
        seed: Random seed for reproducibility.
        n_cars: Number of vehicles to generate.
        max_steps: Maximum simulation step (last possible departure time).
        turn_chance: Probability that a vehicle takes a turning route.
    """
    rng = np.random.default_rng(seed)

    timings = np.sort(rng.weibull(2.0, size=n_cars))
    depart_steps = np.rint(_map_to_interval(timings, 0, max_steps)).astype(int)

    straight = _straight_routes(n)
    turning = _turn_routes(n)

    # Fall back to straight-only when there are no turning routes (n==1)
    has_turns = len(turning) > 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write('<routes>\n')
        f.write(
            '    <vType accel="1.0" decel="4.5" id="standard_car"'
            ' length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />\n\n'
        )

        # Emit route definitions
        all_routes = straight + (turning if has_turns else [])
        for route_id, edges in all_routes:
            f.write(f'    <route id="{route_id}" edges="{" ".join(edges)}" />\n')

        f.write("\n")

        # Emit vehicles
        for car_i, step in enumerate(depart_steps):
            use_turn = has_turns and rng.random() < turn_chance
            pool = turning if use_turn else straight
            idx = int(rng.integers(0, len(pool)))
            route_id = pool[idx][0]
            f.write(
                f'    <vehicle id="{route_id}_{car_i}" type="standard_car"'
                f' route="{route_id}" depart="{step}"'
                f' departLane="random" departSpeed="10" />\n'
            )

        f.write('</routes>\n')
