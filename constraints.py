"""
Constraint Synthesis and Satisfaction System.

This module converts semantic priors into formal constraints and uses
constraint satisfaction to find transformations that respect these priors.

As per the README architecture:
1. Sample semantic goals
2. Encode as constraint problem
3. Use solver to find transformations satisfying constraints
"""

from typing import List, Dict, Optional, Callable, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from grid import Grid
from semantic_priors import SemanticPrior, CompositePrior
from dsl import Transform, DSL


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preference, can be violated with penalty


@dataclass
class Constraint:
    """A single constraint."""
    name: str
    constraint_type: ConstraintType
    check: Callable[[Grid, Grid], bool]
    weight: float = 1.0
    description: str = ""


class ConstraintSystem:
    """System for managing and checking constraints."""

    def __init__(self):
        self.hard_constraints: List[Constraint] = []
        self.soft_constraints: List[Constraint] = []

    def add_hard_constraint(self, name: str, check: Callable[[Grid, Grid], bool],
                           description: str = ""):
        """Add a hard constraint that must be satisfied."""
        constraint = Constraint(name, ConstraintType.HARD, check,
                              description=description)
        self.hard_constraints.append(constraint)

    def add_soft_constraint(self, name: str, check: Callable[[Grid, Grid], bool],
                           weight: float = 1.0, description: str = ""):
        """Add a soft constraint (preference)."""
        constraint = Constraint(name, ConstraintType.SOFT, check, weight,
                              description=description)
        self.soft_constraints.append(constraint)

    def check_hard_constraints(self, input_grid: Grid,
                              output_grid: Grid) -> Tuple[bool, List[str]]:
        """
        Check if all hard constraints are satisfied.

        Returns:
            (satisfied, violated_constraints)
        """
        violated = []
        for constraint in self.hard_constraints:
            if not constraint.check(input_grid, output_grid):
                violated.append(constraint.name)

        return len(violated) == 0, violated

    def score_soft_constraints(self, input_grid: Grid, output_grid: Grid) -> float:
        """
        Score soft constraints (higher is better).

        Returns:
            Weighted score of satisfied soft constraints.
        """
        score = 0.0
        total_weight = sum(c.weight for c in self.soft_constraints)

        if total_weight == 0:
            return 1.0

        for constraint in self.soft_constraints:
            if constraint.check(input_grid, output_grid):
                score += constraint.weight

        return score / total_weight

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> Tuple[bool, float]:
        """
        Evaluate both hard and soft constraints.

        Returns:
            (hard_satisfied, soft_score)
        """
        hard_satisfied, _ = self.check_hard_constraints(input_grid, output_grid)
        soft_score = self.score_soft_constraints(input_grid, output_grid)
        return hard_satisfied, soft_score


class ConstraintSynthesizer:
    """Convert semantic priors into constraints."""

    @staticmethod
    def from_priors(priors: List[SemanticPrior],
                   threshold: float = 0.7) -> ConstraintSystem:
        """
        Convert semantic priors into a constraint system.

        Args:
            priors: List of semantic priors
            threshold: Minimum score to satisfy constraint
        """
        system = ConstraintSystem()

        for prior in priors:
            # Create a constraint from the prior
            def make_check(p: SemanticPrior, t: float):
                def check(inp: Grid, out: Grid) -> bool:
                    return p.evaluate(inp, out) >= t
                return check

            # Determine if this should be hard or soft constraint
            # Symmetry and structural priors are typically hard
            # Aesthetic priors are typically soft
            prior_name = prior.name.lower()

            if any(keyword in prior_name for keyword in
                   ['preserve', 'maintain', 'count', 'cohesion']):
                # Make it a hard constraint
                system.add_hard_constraint(
                    prior.name,
                    make_check(prior, threshold),
                    description=f"Ensure {prior.name}"
                )
            else:
                # Make it a soft constraint
                system.add_soft_constraint(
                    prior.name,
                    make_check(prior, threshold),
                    weight=1.0,
                    description=f"Prefer {prior.name}"
                )

        return system

    @staticmethod
    def from_composite_prior(composite: CompositePrior,
                           threshold: float = 0.7) -> ConstraintSystem:
        """Convert a composite prior into a constraint system."""
        priors = [p for p, _ in composite.priors]
        return ConstraintSynthesizer.from_priors(priors, threshold)


class TransformationFinder:
    """Find transformations that satisfy constraints."""

    def __init__(self, constraint_system: ConstraintSystem,
                 available_transforms: Optional[List[Transform]] = None):
        self.constraint_system = constraint_system
        self.available_transforms = available_transforms or DSL.primitives()

    def find_single_transform(self, input_grid: Grid,
                            max_attempts: int = 50) -> Optional[Transform]:
        """
        Find a single transformation that satisfies constraints.

        This uses random search over available transformations.
        """
        best_transform = None
        best_score = -1.0

        for transform in self.available_transforms:
            try:
                output_grid = transform.apply(input_grid)
                hard_satisfied, soft_score = self.constraint_system.evaluate(
                    input_grid, output_grid
                )

                if hard_satisfied and soft_score > best_score:
                    best_score = soft_score
                    best_transform = transform

            except Exception:
                continue

        return best_transform

    def find_composite_transform(self, input_grid: Grid,
                                max_depth: int = 3,
                                beam_width: int = 5) -> Optional[Transform]:
        """
        Find a composite transformation using beam search.

        Args:
            input_grid: Input grid
            max_depth: Maximum composition depth
            beam_width: Number of candidates to keep at each level

        Returns:
            Best composite transformation found
        """
        from dsl import Compose

        # Beam search
        beam = [([], input_grid, 0.0)]  # (transforms, current_grid, score)

        for depth in range(max_depth):
            candidates = []

            for transforms, current_grid, current_score in beam:
                # Try adding each available transform
                for transform in self.available_transforms:
                    try:
                        new_grid = transform.apply(current_grid)
                        new_transforms = transforms + [transform]

                        # Evaluate
                        hard_satisfied, soft_score = self.constraint_system.evaluate(
                            input_grid, new_grid
                        )

                        if hard_satisfied:
                            # Penalize longer compositions slightly
                            score = soft_score - 0.01 * len(new_transforms)
                            candidates.append((new_transforms, new_grid, score))

                    except Exception:
                        continue

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:beam_width]

        if beam:
            best_transforms, _, _ = beam[0]
            if len(best_transforms) == 1:
                return best_transforms[0]
            elif len(best_transforms) > 1:
                return Compose(best_transforms)

        return None

    def generate_candidate_transforms(self, input_grid: Grid,
                                     num_candidates: int = 10,
                                     max_depth: int = 2) -> List[Tuple[Transform, float]]:
        """
        Generate multiple candidate transformations with scores.

        Returns:
            List of (transform, score) tuples sorted by score.
        """
        candidates = []

        # Try single transforms
        for transform in self.available_transforms:
            try:
                output_grid = transform.apply(input_grid)
                hard_satisfied, soft_score = self.constraint_system.evaluate(
                    input_grid, output_grid
                )

                if hard_satisfied:
                    candidates.append((transform, soft_score))

            except Exception:
                continue

        # Try some composite transforms if max_depth > 1
        if max_depth > 1:
            for _ in range(num_candidates):
                # Random composition
                depth = np.random.randint(2, max_depth + 1)
                transforms = np.random.choice(self.available_transforms,
                                            size=depth, replace=True)

                from dsl import Compose
                composite = Compose(list(transforms))

                try:
                    output_grid = composite.apply(input_grid)
                    hard_satisfied, soft_score = self.constraint_system.evaluate(
                        input_grid, output_grid
                    )

                    if hard_satisfied:
                        # Penalize complexity
                        score = soft_score - 0.02 * composite.description_length()
                        candidates.append((composite, score))

                except Exception:
                    continue

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:num_candidates]


class ComplexityAnalyzer:
    """Analyze the complexity of transformations."""

    @staticmethod
    def minimum_description_length(transform: Transform) -> int:
        """Calculate MDL of a transformation."""
        return transform.description_length()

    @staticmethod
    def estimate_search_tree_size(transform: Transform,
                                 num_primitives: int = 9) -> int:
        """
        Estimate the search tree size needed to find this transformation.

        This is exponential in the MDL.
        """
        mdl = transform.description_length()
        return num_primitives ** mdl

    @staticmethod
    def is_interesting(transform: Transform,
                      mdl_threshold: int = 4,
                      search_size_threshold: int = 1000) -> bool:
        """
        Check if a transformation is "interesting" according to README criteria:
        - Small MDL (elegant solution)
        - Large search space (hard to find)

        This creates the "Goldilocks zone" for puzzle difficulty.
        """
        mdl = ComplexityAnalyzer.minimum_description_length(transform)
        search_size = ComplexityAnalyzer.estimate_search_tree_size(transform)

        return (mdl <= mdl_threshold and search_size >= search_size_threshold)

    @staticmethod
    def calculate_compression_ratio(input_grid: Grid, output_grid: Grid,
                                   transform: Transform) -> float:
        """
        Calculate conceptual compression ratio.

        High compression = can describe transformation simply but
        execution requires many steps.
        """
        # Description complexity (how simply can we describe it)
        description_complexity = transform.description_length()

        # Execution complexity (how much change occurs)
        execution_complexity = np.sum(input_grid.data != output_grid.data)

        if description_complexity == 0:
            return 0.0

        return execution_complexity / description_complexity


class HardnessEmbedding:
    """
    Embed computational hardness into puzzles (as per README Section 1).

    This creates puzzles where solving requires solving an NP-hard subproblem.
    """

    @staticmethod
    def graph_coloring_constraint(num_colors: int = 3) -> Constraint:
        """
        Create a constraint that requires graph coloring.

        The puzzle implicitly requires solving a graph coloring problem.
        """
        def check(inp: Grid, out: Grid) -> bool:
            # Check if output respects graph coloring constraints
            # (adjacent cells should have different colors)
            for r in range(out.height):
                for c in range(out.width):
                    color = out.get(r, c)
                    # Check neighbors
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < out.height and 0 <= nc < out.width:
                            neighbor_color = out.get(nr, nc)
                            if color == neighbor_color and color != 0:
                                return False
            return True

        return Constraint(
            "graph_coloring",
            ConstraintType.HARD,
            check,
            description="Adjacent cells must have different colors"
        )

    @staticmethod
    def satisfiability_constraint(clauses: List[Tuple]) -> Constraint:
        """
        Create a constraint encoding a SAT problem.

        This is more abstract and requires custom implementation per puzzle.
        """
        def check(inp: Grid, out: Grid) -> bool:
            # Simplified: check if output satisfies some boolean formula
            # encoded in the grid structure
            # This is a placeholder for actual SAT encoding
            return True

        return Constraint(
            "satisfiability",
            ConstraintType.HARD,
            check,
            description="Output must satisfy boolean formula"
        )

    @staticmethod
    def hamiltonian_path_constraint() -> Constraint:
        """
        Create a constraint requiring a Hamiltonian path.

        Useful for puzzles involving visiting all objects in sequence.
        """
        def check(inp: Grid, out: Grid) -> bool:
            # Check if colored path visits all objects exactly once
            # Placeholder implementation
            objects = inp.extract_objects()
            return len(objects) > 0

        return Constraint(
            "hamiltonian_path",
            ConstraintType.HARD,
            check,
            description="Must visit all objects exactly once"
        )
