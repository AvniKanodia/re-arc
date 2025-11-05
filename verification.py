"""
Puzzle Verification and Filtering System.

This module implements the "Goldilocks zone" filtering as described in the README:
- Puzzles that are too easy (brute-forceable) are rejected
- Puzzles that are too hard (unsolvable) are rejected
- Puzzles in the sweet spot are kept

The system also implements program synthesis evaluation to estimate solvability.
"""

from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from grid import Grid, Puzzle
from dsl import Transform, DSL, Compose
from constraints import ComplexityAnalyzer


class DifficultyLevel(Enum):
    """Puzzle difficulty levels."""
    TOO_EASY = "too_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    TOO_HARD = "too_hard"


@dataclass
class VerificationResult:
    """Result of puzzle verification."""
    is_valid: bool
    difficulty: DifficultyLevel
    solvability_score: float
    search_time_ms: float
    solution: Optional[Transform]
    metadata: Dict


class ProgramSynthesisSolver:
    """
    Simplified program synthesis solver for estimating puzzle difficulty.

    This implements exhaustive search over the DSL to find solutions,
    which is the baseline approach that we want puzzles to be hard for.
    """

    def __init__(self, max_depth: int = 3, timeout_ms: float = 5000):
        self.max_depth = max_depth
        self.timeout_ms = timeout_ms
        self.primitives = DSL.primitives()

    def solve(self, input_grid: Grid, target_output: Grid) -> Optional[Transform]:
        """
        Try to find a transformation that maps input to output.

        Returns:
            Transform if found within timeout, None otherwise.
        """
        start_time = time.time()

        # Try single transforms first
        for transform in self.primitives:
            if self._check_timeout(start_time):
                return None

            try:
                output = transform.apply(input_grid)
                if output == target_output:
                    return transform
            except Exception:
                continue

        # Try compositions up to max_depth
        for depth in range(2, self.max_depth + 1):
            if self._check_timeout(start_time):
                return None

            solution = self._search_depth(input_grid, target_output, depth, start_time)
            if solution is not None:
                return solution

        return None

    def _search_depth(self, input_grid: Grid, target: Grid,
                     depth: int, start_time: float) -> Optional[Transform]:
        """Search for solution at specific depth using DFS."""
        if depth == 1:
            # Base case: try each primitive
            for transform in self.primitives:
                if self._check_timeout(start_time):
                    return None

                try:
                    output = transform.apply(input_grid)
                    if output == target:
                        return transform
                except Exception:
                    continue
        else:
            # Recursive case: try each primitive followed by depth-1 search
            for transform in self.primitives:
                if self._check_timeout(start_time):
                    return None

                try:
                    intermediate = transform.apply(input_grid)
                    sub_solution = self._search_depth(intermediate, target,
                                                     depth - 1, start_time)
                    if sub_solution is not None:
                        # Compose this transform with the sub-solution
                        if isinstance(sub_solution, Compose):
                            return Compose([transform] + sub_solution.transforms)
                        else:
                            return Compose([transform, sub_solution])
                except Exception:
                    continue

        return None

    def _check_timeout(self, start_time: float) -> bool:
        """Check if timeout has been reached."""
        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms >= self.timeout_ms

    def estimate_search_space_size(self, depth: int) -> int:
        """Estimate the size of the search space at given depth."""
        n = len(self.primitives)
        # Sum of n^1 + n^2 + ... + n^depth
        return sum(n ** d for d in range(1, depth + 1))


class PuzzleVerifier:
    """
    Verify and filter puzzles based on difficulty criteria.

    Implements the Goldilocks zone filtering from README Section 3.
    """

    def __init__(self,
                 min_search_time_ms: float = 100,
                 max_search_time_ms: float = 5000,
                 min_mdl: int = 2,
                 max_mdl: int = 6):
        """
        Args:
            min_search_time_ms: Minimum time to solve (if faster, too easy)
            max_search_time_ms: Maximum time to solve (if slower, too hard)
            min_mdl: Minimum description length (if shorter, too simple)
            max_mdl: Maximum description length (if longer, too complex)
        """
        self.min_search_time_ms = min_search_time_ms
        self.max_search_time_ms = max_search_time_ms
        self.min_mdl = min_mdl
        self.max_mdl = max_mdl
        self.solver = ProgramSynthesisSolver(timeout_ms=max_search_time_ms)

    def verify(self, puzzle: Puzzle) -> VerificationResult:
        """
        Verify a puzzle and determine if it's in the Goldilocks zone.

        Args:
            puzzle: Puzzle to verify

        Returns:
            VerificationResult with difficulty assessment
        """
        if puzzle.num_train == 0:
            return VerificationResult(
                is_valid=False,
                difficulty=DifficultyLevel.TOO_EASY,
                solvability_score=0.0,
                search_time_ms=0.0,
                solution=None,
                metadata={'error': 'No training examples'}
            )

        # Use first training example to estimate difficulty
        input_grid, output_grid = puzzle.train_pairs[0]

        # Try to solve with program synthesis
        start_time = time.time()
        solution = self.solver.solve(input_grid, output_grid)
        search_time_ms = (time.time() - start_time) * 1000

        # Determine difficulty
        difficulty = self._assess_difficulty(solution, search_time_ms)

        # Calculate solvability score (0.0 to 1.0)
        solvability_score = self._calculate_solvability_score(
            solution, search_time_ms
        )

        # Check if in Goldilocks zone
        is_valid = difficulty in [DifficultyLevel.EASY,
                                  DifficultyLevel.MEDIUM,
                                  DifficultyLevel.HARD]

        metadata = {
            'search_time_ms': search_time_ms,
            'mdl': solution.description_length() if solution else None,
            'found_solution': solution is not None,
        }

        return VerificationResult(
            is_valid=is_valid,
            difficulty=difficulty,
            solvability_score=solvability_score,
            search_time_ms=search_time_ms,
            solution=solution,
            metadata=metadata
        )

    def _assess_difficulty(self, solution: Optional[Transform],
                          search_time_ms: float) -> DifficultyLevel:
        """
        Assess puzzle difficulty based on solution and search time.

        Goldilocks criteria from README:
        - MDL(T) is small BUT SearchTreeSize(T) is large
        """
        if solution is None:
            # No solution found within timeout
            return DifficultyLevel.TOO_HARD

        mdl = solution.description_length()

        # Check MDL bounds
        if mdl < self.min_mdl:
            return DifficultyLevel.TOO_EASY
        if mdl > self.max_mdl:
            return DifficultyLevel.TOO_HARD

        # Check search time (proxy for search tree size)
        if search_time_ms < self.min_search_time_ms:
            return DifficultyLevel.TOO_EASY
        elif search_time_ms < self.max_search_time_ms * 0.3:
            return DifficultyLevel.EASY
        elif search_time_ms < self.max_search_time_ms * 0.7:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD

    def _calculate_solvability_score(self, solution: Optional[Transform],
                                    search_time_ms: float) -> float:
        """
        Calculate a solvability score (0.0 = unsolvable, 1.0 = trivial).

        The Goldilocks zone is around 0.4-0.7.
        """
        if solution is None:
            return 0.0

        # Normalize search time to [0, 1]
        time_score = min(1.0, search_time_ms / self.max_search_time_ms)

        # Normalize MDL
        mdl = solution.description_length()
        mdl_score = max(0.0, min(1.0, (mdl - self.min_mdl) / (self.max_mdl - self.min_mdl)))

        # Combined score: balance between time and complexity
        return 0.5 * (1.0 - time_score) + 0.5 * (1.0 - mdl_score)

    def batch_verify(self, puzzles: List[Puzzle]) -> List[VerificationResult]:
        """Verify multiple puzzles."""
        results = []
        for puzzle in puzzles:
            result = self.verify(puzzle)
            results.append(result)
        return results

    def filter_valid_puzzles(self, puzzles: List[Puzzle]) -> List[Tuple[Puzzle, VerificationResult]]:
        """
        Filter puzzles to keep only those in the Goldilocks zone.

        Returns:
            List of (puzzle, verification_result) tuples for valid puzzles.
        """
        valid = []
        for puzzle in puzzles:
            result = self.verify(puzzle)
            if result.is_valid:
                valid.append((puzzle, result))
        return valid


class AdversarialFilter:
    """
    Adversarial filtering to reject puzzles solvable by pattern matching.

    As mentioned in README Section 5, this fights against transformers
    that reduce problems to linearized subgraph matching.
    """

    def __init__(self):
        self.known_patterns = []

    def add_pattern(self, pattern: Transform):
        """Add a known pattern that should be avoided."""
        self.known_patterns.append(pattern)

    def is_novel(self, puzzle: Puzzle, threshold: float = 0.8) -> bool:
        """
        Check if puzzle is sufficiently different from known patterns.

        Args:
            puzzle: Puzzle to check
            threshold: Similarity threshold (lower = more different required)

        Returns:
            True if puzzle is novel enough
        """
        if not self.known_patterns:
            return True

        # Check similarity to each known pattern
        for pattern in self.known_patterns:
            similarity = self._calculate_similarity(puzzle, pattern)
            if similarity > threshold:
                return False

        return True

    def _calculate_similarity(self, puzzle: Puzzle, pattern: Transform) -> float:
        """
        Calculate similarity between puzzle and a pattern.

        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical)
        """
        if puzzle.num_train == 0:
            return 0.0

        matches = 0
        total = puzzle.num_train

        for input_grid, output_grid in puzzle.train_pairs:
            try:
                predicted_output = pattern.apply(input_grid)
                if predicted_output == output_grid:
                    matches += 1
            except Exception:
                continue

        return matches / total


class DiversityMetrics:
    """Measure diversity of generated puzzles."""

    @staticmethod
    def transformation_diversity(puzzles: List[Tuple[Puzzle, Transform]]) -> float:
        """
        Measure diversity of transformations used.

        Returns:
            Diversity score from 0.0 (all same) to 1.0 (all different)
        """
        if not puzzles:
            return 0.0

        transform_types = set()
        for _, transform in puzzles:
            transform_types.add(type(transform).__name__)

        # Diversity = ratio of unique types to total
        return len(transform_types) / len(puzzles)

    @staticmethod
    def semantic_diversity(puzzles: List[Puzzle]) -> float:
        """
        Measure semantic diversity (grid sizes, color palettes, etc.).

        Returns:
            Diversity score from 0.0 to 1.0
        """
        if not puzzles:
            return 0.0

        # Collect features
        sizes = set()
        color_sets = []

        for puzzle in puzzles:
            if puzzle.num_train > 0:
                inp, out = puzzle.train_pairs[0]
                sizes.add((inp.height, inp.width, out.height, out.width))

                inp_colors = set(inp.data.flatten())
                out_colors = set(out.data.flatten())
                color_sets.append(inp_colors.union(out_colors))

        # Size diversity
        size_diversity = len(sizes) / len(puzzles)

        # Color palette diversity (average Jaccard distance)
        color_diversity = 0.0
        if len(color_sets) > 1:
            n = len(color_sets)
            for i in range(n):
                for j in range(i + 1, n):
                    # Jaccard distance
                    intersection = len(color_sets[i] & color_sets[j])
                    union = len(color_sets[i] | color_sets[j])
                    if union > 0:
                        color_diversity += 1.0 - (intersection / union)
            color_diversity /= (n * (n - 1) / 2)

        return 0.5 * size_diversity + 0.5 * color_diversity


class QualityMetrics:
    """Measure quality of individual puzzles."""

    @staticmethod
    def consistency_score(puzzle: Puzzle) -> float:
        """
        Check if all training examples follow the same transformation.

        Returns:
            1.0 if perfectly consistent, lower if inconsistent
        """
        if puzzle.num_train < 2:
            return 1.0

        # This is a simplified check - ideally we'd verify the same
        # transformation works for all examples
        # For now, check if outputs have similar properties

        outputs = [out for _, out in puzzle.train_pairs]

        # Check size consistency
        sizes = [(g.height, g.width) for g in outputs]
        if len(set(sizes)) > 1:
            return 0.5  # Sizes vary

        # Check color palette consistency
        color_sets = [set(g.data.flatten()) for g in outputs]
        avg_jaccard = 0.0
        n = len(color_sets)
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(color_sets[i] & color_sets[j])
                union = len(color_sets[i] | color_sets[j])
                if union > 0:
                    avg_jaccard += intersection / union
        if n > 1:
            avg_jaccard /= (n * (n - 1) / 2)
            return avg_jaccard

        return 1.0

    @staticmethod
    def elegance_score(puzzle: Puzzle, solution: Optional[Transform]) -> float:
        """
        Measure puzzle elegance (simple solution with complex appearance).

        This implements the "conceptual compression" principle from README Section 6.
        """
        if solution is None:
            return 0.0

        if puzzle.num_train == 0:
            return 0.0

        inp, out = puzzle.train_pairs[0]

        # Elegant = small MDL but large actual change
        mdl = solution.description_length()
        change_amount = np.sum(inp.data != out.data)

        if mdl == 0:
            return 0.0

        # Compression ratio
        compression_ratio = change_amount / mdl

        # Normalize to [0, 1]
        return min(1.0, compression_ratio / 20.0)
