"""SUMO environment for NxN intersection grids.

A single SUMO process controls all N² traffic lights simultaneously.
All junctions share the same simulation clock; yellow and green phases
advance together each time ``execute()`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from constants import ACTION_TO_TL_PHASE, TL_GREEN_TO_YELLOW
from grid.config import GridConfig
from grid.reward import get_intersection_cumulated_waiting_time, get_intersection_queue_length
from grid.route_gen import generate_grid_routefile
from grid.state import get_intersection_state, get_neighbor_aware_state


@dataclass
class GridEnvStats:
    """Per-step environment statistics aggregated across all junctions.

    Attributes:
        queue_lengths: Number of stopped vehicles per junction ID.
    """

    queue_lengths: dict[str, int]


class GridEnvironment:
    """SUMO environment wrapper for an NxN intersection grid.

    Args:
        grid_cfg: Full grid configuration (junctions, edges, paths).
        n_cars_generated: Number of vehicles per episode.
        max_steps: Maximum simulation steps per episode.
        yellow_duration: Steps to hold yellow phases.
        green_duration: Steps to hold green phases.
        turn_chance: Fraction of vehicles that take turning routes.
        gui: Whether to launch ``sumo-gui`` instead of headless ``sumo``.
    """

    def __init__(  # noqa: PLR0913
        self,
        grid_cfg: GridConfig,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        gui: bool,
    ) -> None:
        self.grid_cfg = grid_cfg
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.gui = gui
        self.step = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_sumo_cmd(self) -> list[str]:
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cfg = self.grid_cfg.sumocfg_file
        if not cfg.exists():
            msg = f"SUMO config not found at '{cfg}'"
            raise FileNotFoundError(msg)
        return [
            sumo_binary,
            "-c", str(cfg),
            "--no-step-log", "true",
            "--waiting-time-memory", str(self.max_steps),
        ]

    def activate(self) -> None:
        """Start the SUMO simulation via TraCI."""
        self.step = 0
        traci.start(self._build_sumo_cmd())

    def deactivate(self) -> None:
        """Close the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Return True once the step counter has reached max_steps."""
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        """Generate the episode route file.

        Args:
            seed: Random seed for route generation.
        """
        generate_grid_routefile(
            n=self.grid_cfg.n,
            out_path=self.grid_cfg.routes_file,
            seed=seed,
            n_cars=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_states(self) -> dict[str, NDArray]:
        """Return the 240-element state vector for every junction.

        Returns:
            Mapping from TL ID to its state array of shape ``(240,)``.
        """
        return {
            tl: get_intersection_state(self.grid_cfg.intersections[tl])
            for tl in self.grid_cfg.tl_ids
        }

    def get_neighbor_aware_state(self, tl_id: str) -> NDArray:
        """Return the 1200-element neighbour-aware state vector for *tl_id*.

        Args:
            tl_id: Target junction ID.

        Returns:
            Array of shape ``(1200,)``.
        """
        return get_neighbor_aware_state(tl_id, self.grid_cfg)

    def get_cumulated_waiting_time(self, tl_id: str) -> float:
        """Total accumulated waiting time on the junction's incoming edges.

        Args:
            tl_id: Target junction ID.
        """
        return get_intersection_cumulated_waiting_time(self.grid_cfg.intersections[tl_id])

    def get_queue_length(self, tl_id: str) -> int:
        """Number of stopped vehicles on the junction's incoming edges.

        Args:
            tl_id: Target junction ID.
        """
        return get_intersection_queue_length(self.grid_cfg.intersections[tl_id])

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _simulate(self, duration: int) -> list[GridEnvStats]:
        """Advance the simulation for *duration* steps (capped at max_steps)."""
        stats: list[GridEnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)
        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            stats.append(GridEnvStats(
                queue_lengths={tl: self.get_queue_length(tl) for tl in self.grid_cfg.tl_ids}
            ))
        return stats

    def execute(self, actions: dict[str, int]) -> list[GridEnvStats]:
        """Apply actions for all junctions and advance the simulation.

        For each junction whose requested green phase differs from the current
        one, a yellow transition is inserted first (shared across all
        junctions).  Then all junctions switch to their green phases for
        ``green_duration`` steps.

        Args:
            actions: Mapping from TL ID to action index (0–3).

        Returns:
            List of :class:`GridEnvStats`, one per simulation step executed.
        """
        # Determine which junctions need a yellow phase
        phase_changes: dict[str, int] = {}
        for tl in self.grid_cfg.tl_ids:
            next_green = ACTION_TO_TL_PHASE[actions[tl]]
            current = traci.trafficlight.getPhase(tl)
            if next_green != current:
                phase_changes[tl] = next_green

        stats: list[GridEnvStats] = []

        # Insert yellow for all changing junctions simultaneously
        if phase_changes:
            for tl, next_green in phase_changes.items():
                yellow = TL_GREEN_TO_YELLOW[next_green]
                traci.trafficlight.setPhase(tl, yellow)
            stats.extend(self._simulate(self.yellow_duration))

        if self.is_over():
            return stats

        # Set green for all junctions
        for tl in self.grid_cfg.tl_ids:
            traci.trafficlight.setPhase(tl, ACTION_TO_TL_PHASE[actions[tl]])
        stats.extend(self._simulate(self.green_duration))

        return stats
