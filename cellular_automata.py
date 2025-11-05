"""
Cellular Automata Engine for Emergence-Based Puzzle Generation.

This module implements various cellular automata rules and engines
for generating puzzles through emergent complexity. As described in
the README, CA-based generation provides:
- Simple local rules leading to complex patterns
- Bounded computational complexity
- Natural semantic filtering
"""

from typing import Callable, Dict, List, Tuple, Optional
from grid import Grid
import numpy as np
from dataclasses import dataclass
from enum import Enum


class CARule(Enum):
    """Predefined cellular automaton rules."""
    GAME_OF_LIFE = "game_of_life"
    SEEDS = "seeds"
    BRIAN_BRAIN = "brian_brain"
    WIREWORLD = "wireworld"
    CUSTOM = "custom"


@dataclass
class CAConfig:
    """Configuration for cellular automaton."""
    rule: CARule
    steps: int
    alive_color: int = 1
    dead_color: int = 0
    neighborhood_type: str = "moore"  # "moore" or "von_neumann"

    # Custom rule parameters
    survive_conditions: Optional[List[int]] = None  # For custom rules
    birth_conditions: Optional[List[int]] = None


class CellularAutomaton:
    """Base cellular automaton engine."""

    def __init__(self, config: CAConfig):
        self.config = config

    def step(self, grid: Grid) -> Grid:
        """Execute one step of the CA."""
        if self.config.rule == CARule.GAME_OF_LIFE:
            return self._step_game_of_life(grid)
        elif self.config.rule == CARule.SEEDS:
            return self._step_seeds(grid)
        elif self.config.rule == CARule.BRIAN_BRAIN:
            return self._step_brian_brain(grid)
        elif self.config.rule == CARule.WIREWORLD:
            return self._step_wireworld(grid)
        elif self.config.rule == CARule.CUSTOM:
            return self._step_custom(grid)
        else:
            raise ValueError(f"Unknown rule: {self.config.rule}")

    def run(self, initial_grid: Grid, steps: Optional[int] = None) -> Grid:
        """Run the CA for a number of steps."""
        if steps is None:
            steps = self.config.steps

        current = initial_grid
        for _ in range(steps):
            current = self.step(current)

        return current

    def run_with_history(self, initial_grid: Grid,
                        steps: Optional[int] = None) -> List[Grid]:
        """Run the CA and return all intermediate states."""
        if steps is None:
            steps = self.config.steps

        history = [initial_grid]
        current = initial_grid

        for _ in range(steps):
            current = self.step(current)
            history.append(current)

        return history

    def _count_neighbors(self, grid: Grid, row: int, col: int,
                        target_color: int) -> int:
        """Count neighbors with a specific color."""
        count = 0

        if self.config.neighborhood_type == "moore":
            # 8-connected neighborhood
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        else:  # von_neumann
            # 4-connected neighborhood
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                if grid.get(nr, nc) == target_color:
                    count += 1

        return count

    def _step_game_of_life(self, grid: Grid) -> Grid:
        """Conway's Game of Life."""
        new_grid = Grid.empty(grid.height, grid.width, self.config.dead_color)

        for r in range(grid.height):
            for c in range(grid.width):
                alive_neighbors = self._count_neighbors(
                    grid, r, c, self.config.alive_color
                )
                current = grid.get(r, c)

                if current == self.config.alive_color:
                    # Cell is alive
                    if alive_neighbors in [2, 3]:
                        new_grid.set(r, c, self.config.alive_color)
                    else:
                        new_grid.set(r, c, self.config.dead_color)
                else:
                    # Cell is dead
                    if alive_neighbors == 3:
                        new_grid.set(r, c, self.config.alive_color)
                    else:
                        new_grid.set(r, c, self.config.dead_color)

        return new_grid

    def _step_seeds(self, grid: Grid) -> Grid:
        """Seeds rule: B2/S (birth on 2, survive on none)."""
        new_grid = Grid.empty(grid.height, grid.width, self.config.dead_color)

        for r in range(grid.height):
            for c in range(grid.width):
                alive_neighbors = self._count_neighbors(
                    grid, r, c, self.config.alive_color
                )
                current = grid.get(r, c)

                if current == self.config.alive_color:
                    # All living cells die
                    new_grid.set(r, c, self.config.dead_color)
                else:
                    # Birth on exactly 2 neighbors
                    if alive_neighbors == 2:
                        new_grid.set(r, c, self.config.alive_color)
                    else:
                        new_grid.set(r, c, self.config.dead_color)

        return new_grid

    def _step_brian_brain(self, grid: Grid) -> Grid:
        """Brian's Brain: 3-state automaton (dead=0, alive=1, dying=2)."""
        new_grid = Grid.empty(grid.height, grid.width, 0)

        for r in range(grid.height):
            for c in range(grid.width):
                current = grid.get(r, c)
                alive_neighbors = self._count_neighbors(grid, r, c, 1)

                if current == 0:  # Dead
                    if alive_neighbors == 2:
                        new_grid.set(r, c, 1)  # Birth
                    else:
                        new_grid.set(r, c, 0)
                elif current == 1:  # Alive
                    new_grid.set(r, c, 2)  # Dying
                else:  # Dying (2)
                    new_grid.set(r, c, 0)  # Dead

        return new_grid

    def _step_wireworld(self, grid: Grid) -> Grid:
        """
        Wireworld: 4-state automaton for simulating electronics.
        States: 0=empty, 1=electron_head, 2=electron_tail, 3=conductor
        """
        new_grid = Grid.empty(grid.height, grid.width, 0)

        for r in range(grid.height):
            for c in range(grid.width):
                current = grid.get(r, c)

                if current == 0:  # Empty
                    new_grid.set(r, c, 0)
                elif current == 1:  # Electron head
                    new_grid.set(r, c, 2)  # Becomes tail
                elif current == 2:  # Electron tail
                    new_grid.set(r, c, 3)  # Becomes conductor
                elif current == 3:  # Conductor
                    # Count electron heads in neighborhood
                    head_neighbors = self._count_neighbors(grid, r, c, 1)
                    if head_neighbors in [1, 2]:
                        new_grid.set(r, c, 1)  # Becomes head
                    else:
                        new_grid.set(r, c, 3)  # Stays conductor

        return new_grid

    def _step_custom(self, grid: Grid) -> Grid:
        """Custom rule using survive/birth conditions."""
        if (self.config.survive_conditions is None or
            self.config.birth_conditions is None):
            raise ValueError("Custom rule requires survive and birth conditions")

        new_grid = Grid.empty(grid.height, grid.width, self.config.dead_color)

        for r in range(grid.height):
            for c in range(grid.width):
                alive_neighbors = self._count_neighbors(
                    grid, r, c, self.config.alive_color
                )
                current = grid.get(r, c)

                if current == self.config.alive_color:
                    # Cell is alive - check survive conditions
                    if alive_neighbors in self.config.survive_conditions:
                        new_grid.set(r, c, self.config.alive_color)
                    else:
                        new_grid.set(r, c, self.config.dead_color)
                else:
                    # Cell is dead - check birth conditions
                    if alive_neighbors in self.config.birth_conditions:
                        new_grid.set(r, c, self.config.alive_color)
                    else:
                        new_grid.set(r, c, self.config.dead_color)

        return new_grid


class CAFilter:
    """Semantic filtering for CA-generated patterns."""

    @staticmethod
    def has_object_cohesion(grid: Grid, min_object_size: int = 2) -> bool:
        """Check if pattern has cohesive objects."""
        objects = grid.extract_objects(background=0)
        return len(objects) > 0 and all(obj.size >= min_object_size for obj in objects)

    @staticmethod
    def has_bounded_complexity(grid: Grid) -> bool:
        """Check if pattern complexity is bounded."""
        # Measure using color entropy and object count
        color_counts = grid.count_colors()
        num_colors = len(color_counts)
        objects = grid.extract_objects(background=0)

        # Not too simple, not too complex
        return 2 <= num_colors <= 5 and 1 <= len(objects) <= 10

    @staticmethod
    def preserves_symmetry(initial: Grid, final: Grid, axis: str = 'horizontal') -> bool:
        """Check if CA preserves symmetry."""
        initial_symmetric = initial.has_symmetry(axis)
        final_symmetric = final.has_symmetry(axis)
        return initial_symmetric == final_symmetric

    @staticmethod
    def has_interesting_dynamics(history: List[Grid]) -> bool:
        """
        Check if the CA has interesting dynamics (not static, not chaotic).
        Interesting = some change but not too much.
        """
        if len(history) < 2:
            return False

        changes = []
        for i in range(1, len(history)):
            # Count cells that changed
            diff = np.sum(history[i].data != history[i-1].data)
            total = history[i].height * history[i].width
            change_ratio = diff / total
            changes.append(change_ratio)

        avg_change = np.mean(changes)

        # Interesting if between 5% and 40% change per step on average
        return 0.05 <= avg_change <= 0.40

    @staticmethod
    def reaches_stable_state(history: List[Grid], window: int = 3) -> bool:
        """Check if CA reaches a stable or periodic state."""
        if len(history) < window + 1:
            return False

        # Check last few states for stability
        for i in range(len(history) - window, len(history)):
            if not np.array_equal(history[i].data, history[-1].data):
                # Not static - check for period-2 oscillation
                if i < len(history) - 2:
                    if not np.array_equal(history[i].data, history[i+2].data):
                        return False

        return True


class CAPatternGenerator:
    """Generate interesting patterns using CA with semantic filtering."""

    def __init__(self, config: CAConfig, filters: Optional[List[Callable]] = None):
        self.ca = CellularAutomaton(config)
        self.filters = filters or []

    def generate_pattern(self, initial_grid: Grid,
                        max_attempts: int = 10) -> Optional[Tuple[Grid, Grid]]:
        """
        Generate an interesting pattern by running CA.

        Returns:
            Tuple of (initial_grid, final_grid) if successful, None otherwise.
        """
        for _ in range(max_attempts):
            history = self.ca.run_with_history(initial_grid)
            final_grid = history[-1]

            # Apply filters
            passes_all = True
            for filter_fn in self.filters:
                if isinstance(filter_fn, str):
                    # Use predefined filters
                    if filter_fn == "cohesion":
                        if not CAFilter.has_object_cohesion(final_grid):
                            passes_all = False
                            break
                    elif filter_fn == "complexity":
                        if not CAFilter.has_bounded_complexity(final_grid):
                            passes_all = False
                            break
                    elif filter_fn == "dynamics":
                        if not CAFilter.has_interesting_dynamics(history):
                            passes_all = False
                            break
                    elif filter_fn == "stable":
                        if not CAFilter.reaches_stable_state(history):
                            passes_all = False
                            break
                else:
                    # Custom filter function
                    if not filter_fn(initial_grid, final_grid, history):
                        passes_all = False
                        break

            if passes_all:
                return (initial_grid, final_grid)

            # Try a different initial state (add some randomness)
            initial_grid = self._perturb_grid(initial_grid)

        return None

    def _perturb_grid(self, grid: Grid, perturbation_rate: float = 0.1) -> Grid:
        """Add small random perturbations to grid."""
        new_data = grid.data.copy()
        mask = np.random.random(grid.shape) < perturbation_rate
        new_data[mask] = np.random.randint(0, 2, size=np.sum(mask))
        return Grid(new_data)


# ============================================================================
# PREDEFINED INTERESTING CA RULES
# ============================================================================

class InterestingRules:
    """Collection of CA rules known to produce interesting patterns."""

    @staticmethod
    def get_rule(name: str) -> CAConfig:
        """Get a predefined interesting rule."""
        rules = {
            "life": CAConfig(CARule.GAME_OF_LIFE, steps=10),
            "seeds": CAConfig(CARule.SEEDS, steps=5),
            "brian_brain": CAConfig(CARule.BRIAN_BRAIN, steps=10),
            "maze": CAConfig(
                CARule.CUSTOM,
                steps=5,
                survive_conditions=[1, 2, 3, 4, 5],
                birth_conditions=[3]
            ),
            "coral": CAConfig(
                CARule.CUSTOM,
                steps=8,
                survive_conditions=[4, 5, 6, 7, 8],
                birth_conditions=[3]
            ),
            "amoeba": CAConfig(
                CARule.CUSTOM,
                steps=6,
                survive_conditions=[1, 3, 5, 8],
                birth_conditions=[3, 5, 7]
            ),
        }
        return rules.get(name, rules["life"])

    @staticmethod
    def all_rules() -> List[str]:
        """Get names of all predefined rules."""
        return ["life", "seeds", "brian_brain", "maze", "coral", "amoeba"]
