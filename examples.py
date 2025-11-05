"""
Example Usage and Demonstrations.

This file demonstrates how to use the puzzle generator system.
"""

import numpy as np
from grid import Grid, Puzzle
from dsl import *
from semantic_priors import *
from puzzle_generator import (
    PuzzleGenerator, GeneratorConfig, GenerationStrategy,
    generate_puzzle_dataset
)


def example_1_basic_grid_operations():
    """Example 1: Basic grid operations."""
    print("=" * 60)
    print("Example 1: Basic Grid Operations")
    print("=" * 60)

    # Create a grid
    data = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 0]
    ])
    grid = Grid(data)

    print(f"\nOriginal grid ({grid.height}x{grid.width}):")
    print(grid)

    # Extract objects
    objects = grid.extract_objects(background=0)
    print(f"\nFound {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: color={obj.color}, size={obj.size}")

    # Check symmetry
    print(f"\nHorizontal symmetry: {grid.has_symmetry('horizontal')}")
    print(f"Vertical symmetry: {grid.has_symmetry('vertical')}")

    # Transformations
    rotated = grid.rotate(1)
    print(f"\nRotated 90° clockwise:")
    print(rotated)

    flipped = grid.flip('horizontal')
    print(f"\nFlipped horizontally:")
    print(flipped)


def example_2_dsl_transformations():
    """Example 2: Using DSL transformations."""
    print("\n" + "=" * 60)
    print("Example 2: DSL Transformations")
    print("=" * 60)

    # Create a test grid
    grid = Grid(np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ]))

    print("\nOriginal grid:")
    print(grid)

    # Apply various transformations
    transforms = [
        Rotate(1),
        Flip('horizontal'),
        Transpose(),
        SwapColors(1, 3),
        InvertColors()
    ]

    for transform in transforms:
        result = transform.apply(grid)
        mdl = transform.description_length()
        print(f"\n{transform.name} (MDL={mdl}):")
        print(result)


def example_3_semantic_priors():
    """Example 3: Evaluating semantic priors."""
    print("\n" + "=" * 60)
    print("Example 3: Semantic Priors")
    print("=" * 60)

    # Create input and output grids
    input_grid = Grid(np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2]
    ]))

    output_grid = Grid(np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0]
    ]))

    print("\nInput grid:")
    print(input_grid)
    print("\nOutput grid:")
    print(output_grid)

    # Evaluate various priors
    priors = [
        PreserveObjectCount(),
        PreserveObjectShapes(),
        PreserveColorPalette(),
        PreserveRelativePositions()
    ]

    print("\nPrior evaluations:")
    for prior in priors:
        score = prior.evaluate(input_grid, output_grid)
        print(f"  {prior.name}: {score:.2f}")


def example_4_cellular_automata():
    """Example 4: Cellular automata patterns."""
    print("\n" + "=" * 60)
    print("Example 4: Cellular Automata")
    print("=" * 60)

    from cellular_automata import (
        CellularAutomaton, CAConfig, CARule, InterestingRules
    )

    # Create initial state (glider pattern for Game of Life)
    initial = Grid(np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]))

    print("\nInitial state (Glider):")
    print(initial)

    # Run Game of Life
    config = InterestingRules.get_rule("life")
    config.steps = 4
    ca = CellularAutomaton(config)

    history = ca.run_with_history(initial, steps=4)

    print("\nEvolution:")
    for i, grid in enumerate(history[1:], 1):
        print(f"\nStep {i}:")
        print(grid)


def example_5_generate_simple_puzzle():
    """Example 5: Generate a simple puzzle."""
    print("\n" + "=" * 60)
    print("Example 5: Generate Simple Puzzle")
    print("=" * 60)

    # Configure generator
    config = GeneratorConfig(
        strategy=GenerationStrategy.CONSTRAINT_BASED,
        num_train_examples=2,
        num_test_examples=1,
        min_grid_size=4,
        max_grid_size=6,
        max_transform_depth=2
    )

    generator = PuzzleGenerator(config)

    # Generate a single puzzle
    print("\nGenerating puzzle...")
    puzzles = generator.generate(num_puzzles=1)

    if puzzles:
        puzzle, verification = puzzles[0]

        print(f"\nGenerated puzzle with {puzzle.num_train} training examples")
        print(f"Difficulty: {verification.difficulty.value}")
        print(f"Solvability score: {verification.solvability_score:.2f}")
        print(f"Search time: {verification.search_time_ms:.1f}ms")

        # Show first training example
        print("\nTraining Example 1:")
        print("Input:")
        print(puzzle.train_pairs[0][0])
        print("\nOutput:")
        print(puzzle.train_pairs[0][1])

        if verification.solution:
            print(f"\nSolution: {verification.solution}")
            print(f"MDL: {verification.solution.description_length()}")
    else:
        print("Failed to generate valid puzzle")


def example_6_generate_ca_puzzle():
    """Example 6: Generate CA-based puzzle."""
    print("\n" + "=" * 60)
    print("Example 6: Generate CA-Based Puzzle")
    print("=" * 60)

    config = GeneratorConfig(
        strategy=GenerationStrategy.CELLULAR_AUTOMATA,
        num_train_examples=2,
        num_test_examples=1,
        min_grid_size=6,
        max_grid_size=8,
        ca_rule="life",
        ca_steps=5
    )

    generator = PuzzleGenerator(config)

    print("\nGenerating CA-based puzzle...")
    puzzles = generator.generate(num_puzzles=1)

    if puzzles:
        puzzle, verification = puzzles[0]

        print(f"\nGenerated puzzle using rule: {puzzle.metadata.get('ca_rule')}")
        print(f"Steps: {puzzle.metadata.get('ca_steps')}")
        print(f"Difficulty: {verification.difficulty.value}")

        # Show first training example
        print("\nTraining Example 1:")
        print("Initial state:")
        print(puzzle.train_pairs[0][0])
        print("\nFinal state:")
        print(puzzle.train_pairs[0][1])
    else:
        print("Failed to generate valid CA puzzle")


def example_7_generate_dataset():
    """Example 7: Generate a puzzle dataset."""
    print("\n" + "=" * 60)
    print("Example 7: Generate Puzzle Dataset")
    print("=" * 60)

    # Generate small dataset
    dataset = generate_puzzle_dataset(
        num_puzzles=5,
        strategy=GenerationStrategy.HYBRID,
        output_path="puzzles_dataset.json"
    )

    print(f"\nGenerated {len(dataset)} puzzles")
    print("\nFirst puzzle metadata:")
    puzzle, verification = dataset[0]
    print(f"  Strategy: {puzzle.metadata.get('strategy')}")
    print(f"  Difficulty: {verification.difficulty.value}")
    print(f"  Solvability: {verification.solvability_score:.2f}")


def example_8_custom_priors():
    """Example 8: Using custom semantic priors."""
    print("\n" + "=" * 60)
    print("Example 8: Custom Semantic Priors")
    print("=" * 60)

    # Define custom priors for a specific puzzle type
    custom_priors = [
        PreserveObjectCount(),
        PreserveSymmetry('horizontal'),
        ConsistentColorMapping()
    ]

    config = GeneratorConfig(
        strategy=GenerationStrategy.CONSTRAINT_BASED,
        num_train_examples=2,
        max_transform_depth=2
    )

    generator = PuzzleGenerator(config)

    print("\nGenerating puzzle with custom priors:")
    print("  - Preserve object count")
    print("  - Preserve horizontal symmetry")
    print("  - Consistent color mapping")

    puzzles = generator.generate(num_puzzles=1, priors=custom_priors)

    if puzzles:
        puzzle, verification = puzzles[0]
        print(f"\nSuccess! Difficulty: {verification.difficulty.value}")
        print("\nTraining Example:")
        print("Input:")
        print(puzzle.train_pairs[0][0])
        print("\nOutput:")
        print(puzzle.train_pairs[0][1])
    else:
        print("\nFailed to generate puzzle with these priors")


def example_9_composition_search():
    """Example 9: Finding composite transformations."""
    print("\n" + "=" * 60)
    print("Example 9: Composite Transformation Search")
    print("=" * 60)

    from constraints import ConstraintSystem, TransformationFinder

    # Create a specific input/output pair
    input_grid = Grid(np.array([
        [1, 2],
        [3, 4]
    ]))

    target_output = Grid(np.array([
        [3, 4],
        [1, 2]
    ]))

    print("Find transformation from:")
    print(input_grid)
    print("\nTo:")
    print(target_output)

    # Create constraint system
    constraint_system = ConstraintSystem()

    def target_check(inp, out):
        return out == target_output

    constraint_system.add_hard_constraint("matches_target", target_check)

    # Find transformation
    finder = TransformationFinder(constraint_system)

    print("\nSearching for transformation...")
    transform = finder.find_composite_transform(input_grid, max_depth=3, beam_width=10)

    if transform:
        print(f"\nFound: {transform}")
        print(f"MDL: {transform.description_length()}")

        result = transform.apply(input_grid)
        print("\nVerification:")
        print(result)
        print(f"Matches target: {result == target_output}")
    else:
        print("\nNo transformation found")


def main():
    """Run all examples."""
    examples = [
        example_1_basic_grid_operations,
        example_2_dsl_transformations,
        example_3_semantic_priors,
        example_4_cellular_automata,
        example_5_generate_simple_puzzle,
        example_6_generate_ca_puzzle,
        example_7_generate_dataset,
        example_8_custom_priors,
        example_9_composition_search
    ]

    print("\n" + "=" * 60)
    print("PUZZLE GENERATOR EXAMPLES")
    print("=" * 60)

    for i, example_fn in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"\nExample {i} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Run a quick demo with just the first few examples
    print("Running basic examples...")
    example_1_basic_grid_operations()
    example_2_dsl_transformations()
    example_3_semantic_priors()

    print("\n\nTo run all examples including puzzle generation:")
    print("  from examples import main; main()")
