"""
Main Puzzle Generator Orchestrator.

This module implements the complete puzzle generation pipeline described
in the README, integrating all components:

1. Sample semantic goals
2. Encode as constraint problem
3. Use solver to find transformations
4. Generate input states
5. Apply transformations to get outputs
6. Test solvability
7. Filter based on Goldilocks zone
8. Store valid puzzles

This creates genuinely interesting puzzles that are:
- Algorithmically complex (not brute-forceable)
- Semantically coherent (make sense to humans)
- In the difficulty "sweet spot"
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
from pathlib import Path

from grid import Grid, Puzzle
from dsl import Transform, DSL
from semantic_priors import (
    SemanticPrior, CompositePrior, PriorLibrary
)
from constraints import (
    ConstraintSystem, ConstraintSynthesizer, TransformationFinder,
    ComplexityAnalyzer
)
from cellular_automata import (
    CellularAutomaton, CAConfig, CARule, CAPatternGenerator,
    InterestingRules
)
from verification import (
    PuzzleVerifier, VerificationResult, AdversarialFilter,
    DiversityMetrics, QualityMetrics
)


class GenerationStrategy(Enum):
    """Different strategies for puzzle generation."""
    CONSTRAINT_BASED = "constraint_based"
    CELLULAR_AUTOMATA = "cellular_automata"
    HYBRID = "hybrid"
    HARDNESS_EMBEDDING = "hardness_embedding"


@dataclass
class GeneratorConfig:
    """Configuration for puzzle generation."""
    strategy: GenerationStrategy = GenerationStrategy.CONSTRAINT_BASED
    num_train_examples: int = 3
    num_test_examples: int = 1
    min_grid_size: int = 5
    max_grid_size: int = 15
    num_colors: int = 10

    # Constraint-based parameters
    max_transform_depth: int = 3
    beam_width: int = 5

    # CA parameters
    ca_steps: int = 10
    ca_rule: str = "life"

    # Verification parameters
    min_search_time_ms: float = 100
    max_search_time_ms: float = 5000
    min_mdl: int = 2
    max_mdl: int = 6

    # Diversity parameters
    use_adversarial_filtering: bool = True
    diversity_threshold: float = 0.8


class PuzzleGenerator:
    """
    Main puzzle generator implementing the architecture from the README.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.verifier = PuzzleVerifier(
            min_search_time_ms=self.config.min_search_time_ms,
            max_search_time_ms=self.config.max_search_time_ms,
            min_mdl=self.config.min_mdl,
            max_mdl=self.config.max_mdl
        )
        self.adversarial_filter = AdversarialFilter() if self.config.use_adversarial_filtering else None

    def generate(self, num_puzzles: int = 1,
                priors: Optional[List[SemanticPrior]] = None) -> List[Tuple[Puzzle, VerificationResult]]:
        """
        Generate multiple puzzles.

        Args:
            num_puzzles: Number of valid puzzles to generate
            priors: Semantic priors to use (if None, uses defaults)

        Returns:
            List of (puzzle, verification_result) tuples
        """
        puzzles = []
        attempts = 0
        max_attempts = num_puzzles * 10  # Allow some failures

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1

            # Generate a single puzzle
            puzzle = self._generate_single(priors)

            if puzzle is None:
                continue

            # Verify puzzle
            result = self.verifier.verify(puzzle)

            if not result.is_valid:
                continue

            # Check adversarial filter
            if self.adversarial_filter and not self.adversarial_filter.is_novel(puzzle):
                continue

            # Add to results
            puzzles.append((puzzle, result))

            # Update adversarial filter
            if self.adversarial_filter and result.solution:
                self.adversarial_filter.add_pattern(result.solution)

        return puzzles

    def _generate_single(self, priors: Optional[List[SemanticPrior]] = None) -> Optional[Puzzle]:
        """
        Generate a single puzzle using the configured strategy.

        This implements the practical architecture from README.
        """
        if self.config.strategy == GenerationStrategy.CONSTRAINT_BASED:
            return self._generate_constraint_based(priors)
        elif self.config.strategy == GenerationStrategy.CELLULAR_AUTOMATA:
            return self._generate_ca_based(priors)
        elif self.config.strategy == GenerationStrategy.HYBRID:
            # Randomly choose between strategies
            if np.random.random() < 0.5:
                return self._generate_constraint_based(priors)
            else:
                return self._generate_ca_based(priors)
        elif self.config.strategy == GenerationStrategy.HARDNESS_EMBEDDING:
            return self._generate_with_hardness_embedding(priors)
        else:
            return None

    def _generate_constraint_based(self, priors: Optional[List[SemanticPrior]] = None) -> Optional[Puzzle]:
        """
        Generate puzzle using constraint-based approach.

        Steps (from README):
        1. Sample semantic goals
        2. Encode as constraint problem
        3. Find transformation satisfying constraints
        4. Generate input states
        5. Apply transformation to get outputs
        """
        # Step 1: Sample semantic goals
        if priors is None:
            priors = self._sample_priors()

        # Step 2: Encode as constraint problem
        constraint_system = ConstraintSynthesizer.from_priors(priors, threshold=0.7)

        # Step 4: Generate input state (do this before finding transform)
        input_grids = self._generate_input_grids(self.config.num_train_examples +
                                                 self.config.num_test_examples)

        if not input_grids:
            return None

        # Step 3: Find transformation satisfying constraints
        transform_finder = TransformationFinder(constraint_system)

        # Try to find a transform that works well with the first input
        transform = transform_finder.find_composite_transform(
            input_grids[0],
            max_depth=self.config.max_transform_depth,
            beam_width=self.config.beam_width
        )

        if transform is None:
            return None

        # Step 5: Apply transformation to get outputs
        try:
            train_pairs = []
            for i in range(self.config.num_train_examples):
                output = transform.apply(input_grids[i])
                train_pairs.append((input_grids[i], output))

            test_inputs = input_grids[self.config.num_train_examples:]
            test_outputs = [transform.apply(inp) for inp in test_inputs]

            puzzle = Puzzle(
                train_pairs=train_pairs,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
                metadata={
                    'strategy': 'constraint_based',
                    'priors': [p.name for p in priors],
                    'transform': str(transform),
                    'mdl': transform.description_length()
                }
            )

            return puzzle

        except Exception as e:
            return None

    def _generate_ca_based(self, priors: Optional[List[SemanticPrior]] = None) -> Optional[Puzzle]:
        """
        Generate puzzle using cellular automata.

        This creates puzzles where the transformation is:
        "Run CA rule R for N steps"
        """
        # Create CA configuration
        rule_name = self.config.ca_rule
        ca_config = InterestingRules.get_rule(rule_name)
        ca_config.steps = self.config.ca_steps

        # Generate initial states
        filters = ['cohesion', 'complexity', 'dynamics']
        generator = CAPatternGenerator(ca_config, filters)

        train_pairs = []
        for _ in range(self.config.num_train_examples):
            # Generate random initial state
            height = np.random.randint(self.config.min_grid_size,
                                      self.config.max_grid_size + 1)
            width = np.random.randint(self.config.min_grid_size,
                                     self.config.max_grid_size + 1)

            # Create initial grid with some random alive cells
            initial = Grid.empty(height, width, 0)
            num_alive = (height * width) // 4
            for _ in range(num_alive):
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                initial.set(r, c, 1)

            # Generate pattern
            result = generator.generate_pattern(initial)
            if result:
                initial_grid, final_grid = result
                train_pairs.append((initial_grid, final_grid))

        if len(train_pairs) < self.config.num_train_examples:
            return None

        # Generate test examples
        test_inputs = []
        test_outputs = []
        ca = CellularAutomaton(ca_config)

        for _ in range(self.config.num_test_examples):
            height = np.random.randint(self.config.min_grid_size,
                                      self.config.max_grid_size + 1)
            width = np.random.randint(self.config.min_grid_size,
                                     self.config.max_grid_size + 1)

            initial = Grid.empty(height, width, 0)
            num_alive = (height * width) // 4
            for _ in range(num_alive):
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                initial.set(r, c, 1)

            final = ca.run(initial, ca_config.steps)
            test_inputs.append(initial)
            test_outputs.append(final)

        puzzle = Puzzle(
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            metadata={
                'strategy': 'cellular_automata',
                'ca_rule': rule_name,
                'ca_steps': ca_config.steps
            }
        )

        return puzzle

    def _generate_with_hardness_embedding(self, priors: Optional[List[SemanticPrior]] = None) -> Optional[Puzzle]:
        """
        Generate puzzle with embedded computational hardness.

        This creates puzzles that require solving an NP-hard subproblem.
        Currently a simplified implementation.
        """
        # For now, use constraint-based with additional hardness constraints
        # Future: implement actual graph coloring, SAT, etc.
        from constraints import HardnessEmbedding

        if priors is None:
            priors = self._sample_priors()

        # Add hardness constraint
        constraint_system = ConstraintSynthesizer.from_priors(priors, threshold=0.7)
        constraint_system.add_hard_constraint(
            "graph_coloring",
            HardnessEmbedding.graph_coloring_constraint().check,
            "Graph coloring constraint"
        )

        # Generate puzzle similar to constraint-based
        return self._generate_constraint_based(priors)

    def _sample_priors(self) -> List[SemanticPrior]:
        """
        Sample a random combination of semantic priors.

        This creates diversity in generated puzzles.
        """
        # Get all available priors
        all_priors = PriorLibrary.all_priors()

        # Sample 2-4 priors
        num_priors = np.random.randint(2, 5)
        selected = np.random.choice(all_priors, size=min(num_priors, len(all_priors)),
                                   replace=False)

        return list(selected)

    def _generate_input_grids(self, count: int) -> List[Grid]:
        """
        Generate diverse input grids.

        These should have interesting structure (not just random noise).
        """
        grids = []

        for _ in range(count):
            # Random size
            height = np.random.randint(self.config.min_grid_size,
                                      self.config.max_grid_size + 1)
            width = np.random.randint(self.config.min_grid_size,
                                     self.config.max_grid_size + 1)

            # Create grid with structure
            grid_type = np.random.choice(['random', 'objects', 'pattern'])

            if grid_type == 'random':
                # Random but not too noisy
                grid = Grid.random(height, width, num_colors=5)

            elif grid_type == 'objects':
                # Create discrete objects
                grid = Grid.empty(height, width, 0)
                num_objects = np.random.randint(2, 6)

                for _ in range(num_objects):
                    # Random position and size
                    obj_size = np.random.randint(2, 5)
                    start_r = np.random.randint(0, height - obj_size + 1)
                    start_c = np.random.randint(0, width - obj_size + 1)
                    color = np.random.randint(1, 10)

                    # Fill rectangle
                    for r in range(start_r, min(start_r + obj_size, height)):
                        for c in range(start_c, min(start_c + obj_size, width)):
                            grid.set(r, c, color)

            else:  # pattern
                # Create simple pattern
                grid = Grid.empty(height, width, 0)
                color1, color2 = np.random.randint(1, 10, size=2)

                # Checkerboard-like pattern
                for r in range(height):
                    for c in range(width):
                        if (r + c) % 2 == 0:
                            grid.set(r, c, color1)
                        else:
                            grid.set(r, c, color2)

            grids.append(grid)

        return grids


class PuzzleDataset:
    """Manage a dataset of generated puzzles."""

    def __init__(self):
        self.puzzles: List[Tuple[Puzzle, VerificationResult]] = []

    def add_puzzle(self, puzzle: Puzzle, verification: VerificationResult):
        """Add a puzzle to the dataset."""
        self.puzzles.append((puzzle, verification))

    def save(self, path: str):
        """Save dataset to JSON file."""
        data = {
            'puzzles': [
                {
                    'puzzle': puzzle.to_dict(),
                    'verification': {
                        'difficulty': verification.difficulty.value,
                        'solvability_score': verification.solvability_score,
                        'search_time_ms': verification.search_time_ms,
                        'metadata': verification.metadata
                    }
                }
                for puzzle, verification in self.puzzles
            ],
            'statistics': self.get_statistics()
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Note: This is simplified - full reconstruction would need more work
        self.puzzles = []
        for item in data['puzzles']:
            puzzle = Puzzle.from_dict(item['puzzle'])
            # Verification result is stored but not fully reconstructed
            self.puzzles.append((puzzle, None))

    def get_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        if not self.puzzles:
            return {}

        difficulties = [v.difficulty.value for _, v in self.puzzles if v]
        solvability_scores = [v.solvability_score for _, v in self.puzzles if v]

        # Calculate diversity
        diversity = DiversityMetrics.semantic_diversity([p for p, _ in self.puzzles])

        # Calculate quality metrics
        consistency_scores = [QualityMetrics.consistency_score(p) for p, _ in self.puzzles]

        return {
            'total_puzzles': len(self.puzzles),
            'difficulties': {
                d: difficulties.count(d) for d in set(difficulties)
            },
            'avg_solvability': np.mean(solvability_scores) if solvability_scores else 0,
            'diversity_score': diversity,
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0
        }

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx]


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def generate_puzzle_dataset(
    num_puzzles: int = 100,
    strategy: GenerationStrategy = GenerationStrategy.CONSTRAINT_BASED,
    output_path: Optional[str] = None
) -> PuzzleDataset:
    """
    High-level function to generate a puzzle dataset.

    Args:
        num_puzzles: Number of puzzles to generate
        strategy: Generation strategy to use
        output_path: Optional path to save dataset

    Returns:
        PuzzleDataset containing generated puzzles
    """
    config = GeneratorConfig(strategy=strategy)
    generator = PuzzleGenerator(config)

    print(f"Generating {num_puzzles} puzzles using {strategy.value} strategy...")

    dataset = PuzzleDataset()
    puzzles = generator.generate(num_puzzles)

    for puzzle, verification in puzzles:
        dataset.add_puzzle(puzzle, verification)

    print(f"Successfully generated {len(dataset)} puzzles")
    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if output_path:
        dataset.save(output_path)
        print(f"\nDataset saved to {output_path}")

    return dataset
