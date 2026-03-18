"""SUMO environment lifecycle management and action execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from constants import (
    ACTION_TO_TL_PHASE,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)
from environment.generator import generate_routefile
from environment.reward import get_cumulated_waiting_time, get_queue_length
from environment.state import get_state


@dataclass
class EnvStats:
    """Snapshot of environment statistics for a single simulation step."""

    queue_length: int


class Environment:
    """Reinforcement-learning environment wrapper around a SUMO traffic simulation."""

    def __init__(  # noqa: PLR0913
        self,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        sumocfg_file: Path,
        gui: bool,
    ) -> None:
        """Initialize the environment.

        Args:
            n_cars_generated: Number of cars to generate for the episode.
            max_steps: Maximum number of simulation steps in an episode.
            yellow_duration: Number of steps to hold a yellow phase.
            green_duration: Number of steps to hold a green phase.
            turn_chance: Probability for each car to turn instead of going straight.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to use the SUMO GUI binary.
        """
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.sumocfg_file = sumocfg_file
        self.gui = gui
        self.step = 0

    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line based on configuration settings.

        Returns:
            List of command-line arguments to start SUMO.

        Raises:
            FileNotFoundError: If the SUMO config file does not exist.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        return [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]

    def activate(self) -> None:
        """Start the SUMO simulation."""
        if traci.isLoaded():
            traci.close()
        traci.start(self.build_sumo_cmd())

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Check whether the maximum number of steps has been reached.

        Returns:
            True if the episode is finished, False otherwise.
        """
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        """Generate a route file for the current episode.

        Args:
            seed: Random seed used for route generation.
        """
        generate_routefile(
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
        )

    def get_state(self) -> NDArray:
        """Compute the discrete state representation of all vehicles.

        Returns:
            A NumPy array of shape (STATE_SIZE,) with 0/1 occupancy values.
        """
        return get_state()

    def get_cumulated_waiting_time(self) -> float:
        """Compute the sum of waiting times for vehicles on incoming edges.

        Returns:
            Total accumulated waiting time of all vehicles on incoming edges.
        """
        return get_cumulated_waiting_time()

    def get_queue_length(self) -> int:
        """Return the number of stopped vehicles on all incoming edges.

        Returns:
            Total number of vehicles with speed 0 on incoming edges.
        """
        return get_queue_length()

    def _set_yellow_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the yellow phase corresponding to a green phase.

        Args:
            green_phase_code: Code of the current green phase.
        """
        yellow_phase_code = TL_GREEN_TO_YELLOW[green_phase_code]
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_code)

    def _set_green_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the given green phase.

        Args:
            green_phase_code: Code of the green phase to activate.
        """
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, green_phase_code)

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the simulation for a given number of steps.

        The actual number of steps is capped so as not to exceed `max_steps`.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            A list of EnvStats, one entry per simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            stats.append(EnvStats(queue_length=get_queue_length()))

        return stats

    def execute(self, action: int) -> list[EnvStats]:
        """Execute an action by changing the traffic light phase.

        If the requested phase differs from the current one, a yellow phase is
        inserted before switching to the new green phase.

        Args:
            action: Discrete action index mapped to a traffic light phase.

        Returns:
            A list of EnvStats collected during the applied phases.
        """
        next_green_phase = ACTION_TO_TL_PHASE[action]
        current_green_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats.extend(self._simulate(self.yellow_duration))

        if self.is_over():
            return stats

        self._set_green_phase(next_green_phase)
        stats.extend(self._simulate(self.green_duration))

        return stats
