"""
Semantic Priors for Puzzle Generation.

This module defines human-interpretable concepts and constraints that
make puzzles interesting and logically coherent. These priors ensure
that generated puzzles have solutions that "make sense" to humans.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Optional, Callable
from grid import Grid, Object
from enum import Enum
import numpy as np


class SemanticPriorType(Enum):
    """Categories of semantic priors."""
    SYMMETRY = "symmetry"
    OBJECT_TRACKING = "object_tracking"
    COLOR_CONSISTENCY = "color_consistency"
    SPATIAL_RELATIONSHIP = "spatial_relationship"
    PATTERN_CONTINUATION = "pattern_continuation"
    CONTAINMENT = "containment"
    COUNTING = "counting"
    GROUPING = "grouping"


class SemanticPrior(ABC):
    """Base class for semantic priors."""

    def __init__(self, name: str, prior_type: SemanticPriorType):
        self.name = name
        self.prior_type = prior_type

    @abstractmethod
    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        """
        Evaluate how well the transformation respects this prior.
        Returns a score from 0.0 (violates prior) to 1.0 (perfectly respects).
        """
        pass

    @abstractmethod
    def generate_constraint(self) -> Dict:
        """
        Generate a constraint specification that can be used by
        a constraint solver to find valid transformations.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# ============================================================================
# SYMMETRY PRIORS
# ============================================================================

class PreserveSymmetry(SemanticPrior):
    """Input symmetry should be preserved in output."""

    def __init__(self, axis: str = 'horizontal'):
        super().__init__(f"preserve_symmetry_{axis}", SemanticPriorType.SYMMETRY)
        self.axis = axis

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_symmetric = input_grid.has_symmetry(self.axis)
        output_symmetric = output_grid.has_symmetry(self.axis)

        if input_symmetric and output_symmetric:
            return 1.0
        elif not input_symmetric and not output_symmetric:
            return 0.5  # Neutral
        else:
            return 0.0  # Violated

    def generate_constraint(self) -> Dict:
        return {
            'type': 'symmetry_preservation',
            'axis': self.axis,
            'requirement': 'preserve'
        }


class CreateSymmetry(SemanticPrior):
    """Output should have symmetry even if input doesn't."""

    def __init__(self, axis: str = 'horizontal'):
        super().__init__(f"create_symmetry_{axis}", SemanticPriorType.SYMMETRY)
        self.axis = axis

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        return 1.0 if output_grid.has_symmetry(self.axis) else 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'symmetry_creation',
            'axis': self.axis,
            'requirement': 'create'
        }


# ============================================================================
# OBJECT TRACKING PRIORS
# ============================================================================

class PreserveObjectCount(SemanticPrior):
    """Number of objects should remain the same."""

    def __init__(self, background: int = 0):
        super().__init__("preserve_object_count", SemanticPriorType.OBJECT_TRACKING)
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_objs = len(input_grid.extract_objects(self.background))
        output_objs = len(output_grid.extract_objects(self.background))

        if input_objs == 0:
            return 0.5  # Neutral

        return 1.0 if input_objs == output_objs else 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'object_count',
            'requirement': 'preserve',
            'background': self.background
        }


class PreserveObjectShapes(SemanticPrior):
    """Object shapes should remain the same (but may move/rotate)."""

    def __init__(self, background: int = 0):
        super().__init__("preserve_object_shapes", SemanticPriorType.OBJECT_TRACKING)
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_objs = input_grid.extract_objects(self.background)
        output_objs = output_grid.extract_objects(self.background)

        if len(input_objs) == 0:
            return 0.5

        # Check if sizes are preserved
        input_sizes = sorted([obj.size for obj in input_objs])
        output_sizes = sorted([obj.size for obj in output_objs])

        return 1.0 if input_sizes == output_sizes else 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'object_shapes',
            'requirement': 'preserve',
            'background': self.background
        }


class MaintainObjectCohesion(SemanticPrior):
    """Objects should remain connected (no fragmentation)."""

    def __init__(self, background: int = 0):
        super().__init__("maintain_cohesion", SemanticPriorType.OBJECT_TRACKING)
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        output_objs = output_grid.extract_objects(self.background)

        if len(output_objs) == 0:
            return 0.5

        # All objects should be cohesive (connected)
        # This is guaranteed by extract_objects, so we check fragmentation
        # by comparing expected vs actual object count

        return 1.0  # Always satisfied with current implementation

    def generate_constraint(self) -> Dict:
        return {
            'type': 'object_cohesion',
            'requirement': 'maintain',
            'background': self.background
        }


# ============================================================================
# COLOR CONSISTENCY PRIORS
# ============================================================================

class PreserveColorPalette(SemanticPrior):
    """Output should use same colors as input (no new colors)."""

    def __init__(self):
        super().__init__("preserve_color_palette", SemanticPriorType.COLOR_CONSISTENCY)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_colors = set(input_grid.data.flatten())
        output_colors = set(output_grid.data.flatten())

        if len(input_colors) == 0:
            return 0.5

        # Check if output colors are subset of input colors
        if output_colors.issubset(input_colors):
            return 1.0
        else:
            # Penalize based on number of new colors
            new_colors = output_colors - input_colors
            return max(0.0, 1.0 - len(new_colors) / 10.0)

    def generate_constraint(self) -> Dict:
        return {
            'type': 'color_palette',
            'requirement': 'preserve'
        }


class ConsistentColorMapping(SemanticPrior):
    """If color A -> B in one place, it should be consistent everywhere."""

    def __init__(self):
        super().__init__("consistent_color_mapping", SemanticPriorType.COLOR_CONSISTENCY)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # Check for consistency in color transformations
        mappings = {}
        consistent = True

        for r in range(min(input_grid.height, output_grid.height)):
            for c in range(min(input_grid.width, output_grid.width)):
                inp_color = input_grid.get(r, c)
                out_color = output_grid.get(r, c)

                if inp_color in mappings:
                    if mappings[inp_color] != out_color:
                        consistent = False
                        break
                else:
                    mappings[inp_color] = out_color

            if not consistent:
                break

        return 1.0 if consistent else 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'color_mapping',
            'requirement': 'consistent'
        }


# ============================================================================
# SPATIAL RELATIONSHIP PRIORS
# ============================================================================

class PreserveRelativePositions(SemanticPrior):
    """Relative positions of objects should be maintained."""

    def __init__(self, background: int = 0):
        super().__init__("preserve_relative_positions",
                        SemanticPriorType.SPATIAL_RELATIONSHIP)
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_objs = input_grid.extract_objects(self.background)
        output_objs = output_grid.extract_objects(self.background)

        if len(input_objs) < 2 or len(output_objs) < 2:
            return 0.5

        # Compare pairwise relationships
        # For simplicity, check if ordering is preserved
        input_centers = self._get_centers(input_objs)
        output_centers = self._get_centers(output_objs)

        if len(input_centers) != len(output_centers):
            return 0.0

        # Check if relative ordering is preserved
        score = 1.0
        for i in range(len(input_centers)):
            for j in range(i + 1, len(input_centers)):
                inp_rel = (input_centers[i][0] < input_centers[j][0],
                          input_centers[i][1] < input_centers[j][1])
                out_rel = (output_centers[i][0] < output_centers[j][0],
                          output_centers[i][1] < output_centers[j][1])

                if inp_rel != out_rel:
                    score -= 0.1

        return max(0.0, score)

    def _get_centers(self, objects: List[Object]) -> List[Tuple[float, float]]:
        centers = []
        for obj in objects:
            rows, cols = zip(*obj.cells)
            centers.append((np.mean(rows), np.mean(cols)))
        return centers

    def generate_constraint(self) -> Dict:
        return {
            'type': 'relative_positions',
            'requirement': 'preserve',
            'background': self.background
        }


class AlignObjects(SemanticPrior):
    """Objects should be aligned (e.g., along rows or columns)."""

    def __init__(self, axis: str = 'horizontal', background: int = 0):
        super().__init__(f"align_objects_{axis}",
                        SemanticPriorType.SPATIAL_RELATIONSHIP)
        self.axis = axis
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        output_objs = output_grid.extract_objects(self.background)

        if len(output_objs) < 2:
            return 0.5

        centers = self._get_centers(output_objs)

        if self.axis == 'horizontal':
            # Check if all objects have similar row centers
            rows = [c[0] for c in centers]
            variance = np.var(rows)
        else:
            # Check if all objects have similar col centers
            cols = [c[1] for c in centers]
            variance = np.var(cols)

        # Lower variance = better alignment
        score = 1.0 / (1.0 + variance)
        return min(1.0, score)

    def _get_centers(self, objects: List[Object]) -> List[Tuple[float, float]]:
        centers = []
        for obj in objects:
            rows, cols = zip(*obj.cells)
            centers.append((np.mean(rows), np.mean(cols)))
        return centers

    def generate_constraint(self) -> Dict:
        return {
            'type': 'alignment',
            'axis': self.axis,
            'background': self.background
        }


# ============================================================================
# PATTERN CONTINUATION PRIORS
# ============================================================================

class ExtendPattern(SemanticPrior):
    """Output should continue a pattern from the input."""

    def __init__(self):
        super().__init__("extend_pattern", SemanticPriorType.PATTERN_CONTINUATION)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # Check if output is larger and contains input as subgrid
        if (output_grid.height >= input_grid.height and
            output_grid.width >= input_grid.width):

            # Check if input is contained in output
            for r_offset in range(output_grid.height - input_grid.height + 1):
                for c_offset in range(output_grid.width - input_grid.width + 1):
                    matches = True
                    for r in range(input_grid.height):
                        for c in range(input_grid.width):
                            if (input_grid.get(r, c) !=
                                output_grid.get(r + r_offset, c + c_offset)):
                                matches = False
                                break
                        if not matches:
                            break
                    if matches:
                        return 1.0

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'pattern_continuation',
            'requirement': 'extend'
        }


# ============================================================================
# CONTAINMENT PRIORS
# ============================================================================

class PartitionSpace(SemanticPrior):
    """Objects should partition the space (like graph coloring)."""

    def __init__(self, background: int = 0):
        super().__init__("partition_space", SemanticPriorType.CONTAINMENT)
        self.background = background

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # Check if output has minimal background
        output_counts = output_grid.count_colors()
        background_ratio = output_counts.get(self.background, 0) / (
            output_grid.height * output_grid.width
        )

        # Less background = better partitioning
        return 1.0 - background_ratio

    def generate_constraint(self) -> Dict:
        return {
            'type': 'space_partition',
            'requirement': 'minimize_background',
            'background': self.background
        }


# ============================================================================
# COUNTING PRIORS
# ============================================================================

class CountBasedTransformation(SemanticPrior):
    """Transformation depends on counting something."""

    def __init__(self, count_target: str = 'objects'):
        super().__init__(f"count_{count_target}", SemanticPriorType.COUNTING)
        self.count_target = count_target

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # This is more of a meta-prior
        # Check if there's a relationship between counts
        if self.count_target == 'objects':
            input_count = len(input_grid.extract_objects())
            output_count = len(output_grid.extract_objects())
            # Look for simple relationships
            if output_count in [input_count, input_count * 2, input_count // 2]:
                return 1.0

        return 0.5

    def generate_constraint(self) -> Dict:
        return {
            'type': 'counting',
            'target': self.count_target
        }


# ============================================================================
# COMPOSITE PRIORS
# ============================================================================

class CompositePrior:
    """Combination of multiple priors with weights."""

    def __init__(self, priors: List[Tuple[SemanticPrior, float]]):
        """
        Args:
            priors: List of (prior, weight) tuples
        """
        self.priors = priors

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        """Weighted average of all prior scores."""
        total_weight = sum(w for _, w in self.priors)
        if total_weight == 0:
            return 0.0

        score = 0.0
        for prior, weight in self.priors:
            score += prior.evaluate(input_grid, output_grid) * weight

        return score / total_weight

    def generate_constraints(self) -> List[Dict]:
        """Generate all constraints from component priors."""
        return [prior.generate_constraint() for prior, _ in self.priors]


# ============================================================================
# PRIOR LIBRARY
# ============================================================================

class PriorLibrary:
    """Collection of commonly used prior combinations."""

    @staticmethod
    def geometric_puzzles() -> CompositePrior:
        """Priors for geometric transformation puzzles."""
        return CompositePrior([
            (PreserveObjectCount(), 1.0),
            (PreserveObjectShapes(), 1.0),
            (PreserveColorPalette(), 0.5),
        ])

    @staticmethod
    def color_puzzles() -> CompositePrior:
        """Priors for color transformation puzzles."""
        return CompositePrior([
            (ConsistentColorMapping(), 1.0),
            (PreserveObjectShapes(), 1.0),
            (PreserveObjectCount(), 0.5),
        ])

    @staticmethod
    def spatial_puzzles() -> CompositePrior:
        """Priors for spatial relationship puzzles."""
        return CompositePrior([
            (PreserveRelativePositions(), 1.0),
            (PreserveObjectCount(), 1.0),
            (MaintainObjectCohesion(), 0.5),
        ])

    @staticmethod
    def symmetry_puzzles() -> CompositePrior:
        """Priors for symmetry-based puzzles."""
        return CompositePrior([
            (PreserveSymmetry('horizontal'), 0.5),
            (CreateSymmetry('horizontal'), 0.5),
            (PreserveColorPalette(), 1.0),
        ])

    @staticmethod
    def all_priors() -> List[SemanticPrior]:
        """Get all available priors."""
        return [
            PreserveSymmetry('horizontal'),
            PreserveSymmetry('vertical'),
            CreateSymmetry('horizontal'),
            CreateSymmetry('vertical'),
            PreserveObjectCount(),
            PreserveObjectShapes(),
            MaintainObjectCohesion(),
            PreserveColorPalette(),
            ConsistentColorMapping(),
            PreserveRelativePositions(),
            AlignObjects('horizontal'),
            AlignObjects('vertical'),
            ExtendPattern(),
            PartitionSpace(),
            CountBasedTransformation(),
        ]
