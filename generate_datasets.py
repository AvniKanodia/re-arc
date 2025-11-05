"""
Comprehensive Dataset Generation Script

Generates diverse puzzle datasets using:
- Custom transformations
- Custom semantic priors
- Custom CA rules
- Multiple generation strategies
- Color and shape-based puzzles

Creates visualizations for all generated puzzles.
"""

import os
import json
from typing import List, Dict
import numpy as np

from puzzle_generator import (
    PuzzleGenerator, GeneratorConfig, GenerationStrategy,
    PuzzleDataset
)
from grid import Grid, Puzzle
from dsl import Compose
from constraints import TransformationFinder, ConstraintSynthesizer
from verification import PuzzleVerifier
from visualization import (
    create_puzzle_grid_visualization,
    create_dataset_summary_visualization,
    PuzzleVisualizer, ShapeVisualizer
)

# Import custom extensions
from custom_transforms import get_all_custom_transforms
from custom_priors import get_all_custom_priors, get_aha_moment_priors
from custom_ca_rules import get_custom_ca_rules, get_aha_moment_rules


class ExtendedPuzzleGenerator:
    """
    Extended puzzle generator using custom transforms, priors, and CA rules.
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.generator = PuzzleGenerator(config)

        # Add custom transforms to available pool
        self.custom_transforms = get_all_custom_transforms()
        self.custom_priors = get_all_custom_priors()
        self.aha_priors = get_aha_moment_priors()

    def generate_with_custom_transforms(self, num_puzzles: int = 10) -> List[tuple]:
        """Generate puzzles using custom transformations."""
        print(f"Generating {num_puzzles} puzzles with custom transforms...")

        puzzles = []
        attempts = 0
        max_attempts = num_puzzles * 15

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1

            # Select random custom transform
            transform = np.random.choice(self.custom_transforms)

            # Generate input grids
            input_grids = []
            for _ in range(self.config.num_train_examples + 1):
                height = np.random.randint(
                    self.config.min_grid_size,
                    self.config.max_grid_size + 1
                )
                width = np.random.randint(
                    self.config.min_grid_size,
                    self.config.max_grid_size + 1
                )

                # Create structured input (not random noise)
                grid = self._create_structured_grid(height, width)
                input_grids.append(grid)

            # Apply transform
            try:
                train_pairs = []
                for i in range(self.config.num_train_examples):
                    output = transform.apply(input_grids[i])
                    train_pairs.append((input_grids[i], output))

                test_input = input_grids[-1]
                test_output = transform.apply(test_input)

                puzzle = Puzzle(
                    train_pairs=train_pairs,
                    test_inputs=[test_input],
                    test_outputs=[test_output],
                    metadata={
                        'strategy': 'custom_transform',
                        'transform': transform.name,
                        'mdl': transform.description_length()
                    }
                )

                # Verify difficulty
                verifier = PuzzleVerifier()
                result = verifier.verify(puzzle)

                if result.is_valid:
                    puzzle.metadata['difficulty'] = result.difficulty.value
                    puzzles.append((puzzle, result))
                    print(f"  Generated puzzle {len(puzzles)}/{num_puzzles} "
                          f"(transform: {transform.name}, difficulty: {result.difficulty.value})")

            except Exception as e:
                continue

        return puzzles

    def generate_with_aha_priors(self, num_puzzles: int = 10) -> List[tuple]:
        """Generate puzzles optimized for "aha moment" insights."""
        print(f"Generating {num_puzzles} puzzles with aha-moment priors...")

        puzzles = []
        attempts = 0
        max_attempts = num_puzzles * 15

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1

            # Sample aha-moment priors
            num_priors = np.random.randint(2, 4)
            selected_priors = np.random.choice(
                self.aha_priors,
                size=min(num_priors, len(self.aha_priors)),
                replace=False
            )

            # Use constraint-based generation with these priors
            try:
                puzzle_result = self.generator.generate(1, priors=list(selected_priors))

                if puzzle_result:
                    puzzle, result = puzzle_result[0]
                    puzzle.metadata['prior_types'] = [p.name for p in selected_priors]
                    puzzle.metadata['aha_optimized'] = True

                    puzzles.append((puzzle, result))
                    print(f"  Generated puzzle {len(puzzles)}/{num_puzzles} "
                          f"(priors: {[p.name for p in selected_priors][:2]}...)")

            except Exception as e:
                continue

        return puzzles

    def generate_with_custom_ca(self, num_puzzles: int = 5) -> List[tuple]:
        """Generate puzzles using custom CA rules."""
        print(f"Generating {num_puzzles} puzzles with custom CA rules...")

        from cellular_automata import CellularAutomaton
        from custom_ca_rules import get_aha_moment_rules

        puzzles = []
        ca_rules = get_custom_ca_rules()
        aha_rules = get_aha_moment_rules()

        for rule_name in aha_rules[:num_puzzles]:
            try:
                config = ca_rules[rule_name]
                ca = CellularAutomaton(config)

                # Generate examples
                train_pairs = []
                for _ in range(self.config.num_train_examples):
                    h = np.random.randint(8, 12)
                    w = np.random.randint(8, 12)

                    # Create initial state
                    initial = self._create_ca_initial_state(h, w)
                    final = ca.run(initial, config.steps)

                    train_pairs.append((initial, final))

                # Test example
                h = np.random.randint(8, 12)
                w = np.random.randint(8, 12)
                test_input = self._create_ca_initial_state(h, w)
                test_output = ca.run(test_input, config.steps)

                puzzle = Puzzle(
                    train_pairs=train_pairs,
                    test_inputs=[test_input],
                    test_outputs=[test_output],
                    metadata={
                        'strategy': 'custom_ca',
                        'ca_rule': rule_name,
                        'ca_steps': config.steps
                    }
                )

                # Verify
                verifier = PuzzleVerifier()
                result = verifier.verify(puzzle)

                puzzle.metadata['difficulty'] = result.difficulty.value
                puzzles.append((puzzle, result))

                print(f"  Generated CA puzzle {len(puzzles)}/{num_puzzles} "
                      f"(rule: {rule_name})")

            except Exception as e:
                print(f"  Failed to generate CA puzzle with {rule_name}: {e}")
                continue

        return puzzles

    def _create_structured_grid(self, height: int, width: int) -> Grid:
        """Create a grid with interesting structure (not random noise)."""
        grid_type = np.random.choice(['objects', 'pattern', 'sparse', 'dense'])

        if grid_type == 'objects':
            # Discrete objects
            grid = Grid.empty(height, width, 0)
            num_objects = np.random.randint(2, 5)

            for _ in range(num_objects):
                obj_h = np.random.randint(1, min(4, height))
                obj_w = np.random.randint(1, min(4, width))
                start_r = np.random.randint(0, height - obj_h + 1)
                start_c = np.random.randint(0, width - obj_w + 1)
                color = np.random.randint(1, 6)

                for r in range(start_r, min(start_r + obj_h, height)):
                    for c in range(start_c, min(start_c + obj_w, width)):
                        grid.set(r, c, color)

        elif grid_type == 'pattern':
            # Repeating pattern
            grid = Grid.empty(height, width, 0)
            pattern_type = np.random.choice(['checkerboard', 'stripes', 'grid'])

            if pattern_type == 'checkerboard':
                for r in range(height):
                    for c in range(width):
                        if (r + c) % 2 == 0:
                            grid.set(r, c, 1)
                        else:
                            grid.set(r, c, 2)

            elif pattern_type == 'stripes':
                for r in range(height):
                    color = (r % 3) + 1
                    for c in range(width):
                        grid.set(r, c, color)

            else:  # grid
                for r in range(height):
                    for c in range(width):
                        if r % 2 == 0 or c % 2 == 0:
                            grid.set(r, c, 1)

        elif grid_type == 'sparse':
            # Sparse random cells
            grid = Grid.empty(height, width, 0)
            num_cells = (height * width) // 6
            for _ in range(num_cells):
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                color = np.random.randint(1, 5)
                grid.set(r, c, color)

        else:  # dense
            # Mostly filled with few empty spots
            grid = Grid.random(height, width, num_colors=5)
            # Clear some spots
            num_clear = (height * width) // 8
            for _ in range(num_clear):
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                grid.set(r, c, 0)

        return grid

    def _create_ca_initial_state(self, height: int, width: int) -> Grid:
        """Create interesting initial state for CA."""
        state_type = np.random.choice(['random', 'centered', 'edges', 'cross'])

        grid = Grid.empty(height, width, 0)

        if state_type == 'random':
            num_alive = (height * width) // 4
            for _ in range(num_alive):
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                grid.set(r, c, 1)

        elif state_type == 'centered':
            center_r, center_c = height // 2, width // 2
            size = min(height, width) // 3
            for r in range(max(0, center_r - size), min(height, center_r + size)):
                for c in range(max(0, center_c - size), min(width, center_c + size)):
                    if np.random.random() < 0.6:
                        grid.set(r, c, 1)

        elif state_type == 'edges':
            # Alive cells on edges
            for r in range(height):
                if np.random.random() < 0.5:
                    grid.set(r, 0, 1)
                    grid.set(r, width - 1, 1)
            for c in range(width):
                if np.random.random() < 0.5:
                    grid.set(0, c, 1)
                    grid.set(height - 1, c, 1)

        else:  # cross
            center_r, center_c = height // 2, width // 2
            for r in range(height):
                grid.set(r, center_c, 1)
            for c in range(width):
                grid.set(center_r, c, 1)

        return grid


def generate_comprehensive_dataset(output_dir: str = "puzzle_datasets"):
    """
    Generate a comprehensive dataset with diverse puzzle types.
    """
    print("=" * 70)
    print("COMPREHENSIVE PUZZLE DATASET GENERATION")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Configuration for high-quality puzzles
    config = GeneratorConfig(
        strategy=GenerationStrategy.HYBRID,
        num_train_examples=3,
        num_test_examples=1,
        min_grid_size=5,
        max_grid_size=12,
        max_transform_depth=3,
        min_search_time_ms=100,
        max_search_time_ms=8000,
        min_mdl=2,
        max_mdl=6,
        use_adversarial_filtering=True
    )

    ext_generator = ExtendedPuzzleGenerator(config)

    # Generate different puzzle types
    all_puzzles = []

    # 1. Custom transform puzzles (most diverse)
    custom_puzzles = ext_generator.generate_with_custom_transforms(num_puzzles=15)
    all_puzzles.extend(custom_puzzles)

    # 2. Aha-moment optimized puzzles
    aha_puzzles = ext_generator.generate_with_aha_priors(num_puzzles=10)
    all_puzzles.extend(aha_puzzles)

    # 3. Custom CA puzzles
    ca_puzzles = ext_generator.generate_with_custom_ca(num_puzzles=5)
    all_puzzles.extend(ca_puzzles)

    # 4. Standard constraint-based puzzles
    print(f"Generating 10 puzzles with standard constraint-based approach...")
    standard_generator = PuzzleGenerator(config)
    standard_puzzles = standard_generator.generate(num_puzzles=10)
    all_puzzles.extend(standard_puzzles)

    print(f"\n{'=' * 70}")
    print(f"TOTAL PUZZLES GENERATED: {len(all_puzzles)}")
    print(f"{'=' * 70}\n")

    # Create dataset
    dataset = PuzzleDataset()
    for puzzle, verification in all_puzzles:
        dataset.add_puzzle(puzzle, verification)

    # Save dataset
    dataset_path = os.path.join(output_dir, "comprehensive_dataset.json")
    dataset.save(dataset_path)
    print(f"Dataset saved to: {dataset_path}")

    # Print statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return dataset, all_puzzles


def create_visualizations(puzzles: List[tuple], output_dir: str = "puzzle_visualizations"):
    """
    Create PNG visualizations for all puzzles.
    """
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 70}\n")

    os.makedirs(output_dir, exist_ok=True)

    for i, (puzzle, verification) in enumerate(puzzles):
        try:
            filename = f"puzzle_{i+1:03d}_{verification.difficulty.value}"

            # Use appropriate visualizer
            viz_type = 'color'  # Default

            if puzzle.metadata.get('use_shapes'):
                viz_type = 'shape'

            output_path = create_puzzle_grid_visualization(
                puzzle,
                output_dir,
                filename,
                visualizer_type=viz_type
            )

            print(f"  Created visualization {i+1}/{len(puzzles)}: {filename}.png")

        except Exception as e:
            print(f"  Failed to visualize puzzle {i+1}: {e}")

    # Create summary visualization
    puzzle_list = [p for p, _ in puzzles]
    summary_path = os.path.join(output_dir, "dataset_summary.png")
    create_dataset_summary_visualization(puzzle_list, summary_path, samples=min(6, len(puzzle_list)))
    print(f"\n  Created dataset summary: dataset_summary.png")

    print(f"\n{'=' * 70}")
    print(f"VISUALIZATIONS COMPLETE")
    print(f"All images saved to: {output_dir}")
    print(f"{'=' * 70}")


def main():
    """Main execution function."""
    import time

    start_time = time.time()

    print("\n" + "=" * 70)
    print("RE-ARC COMPREHENSIVE PUZZLE GENERATION")
    print("Generating diverse, interesting puzzles with 'aha moments'")
    print("=" * 70 + "\n")

    # Generate dataset
    dataset, puzzles = generate_comprehensive_dataset(
        output_dir="puzzle_datasets"
    )

    # Create visualizations
    create_visualizations(
        puzzles,
        output_dir="puzzle_visualizations"
    )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Total puzzles: {len(puzzles)}")
    print(f"Average time per puzzle: {elapsed/len(puzzles):.1f}s")
    print(f"\nOutputs:")
    print(f"  - Dataset JSON: puzzle_datasets/comprehensive_dataset.json")
    print(f"  - Visualizations: puzzle_visualizations/*.png")
    print(f"  - Summary: puzzle_visualizations/dataset_summary.png")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
