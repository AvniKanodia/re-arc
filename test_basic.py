"""
Basic tests to verify the system works.

Run with: python test_basic.py
"""

import numpy as np
from grid import Grid, Puzzle
from dsl import Rotate, Flip, Compose
from semantic_priors import PreserveObjectCount, PreserveColorPalette
from constraints import ConstraintSynthesizer
from cellular_automata import InterestingRules, CellularAutomaton


def test_grid_creation():
    """Test basic grid creation and operations."""
    print("Testing grid creation...")

    # Create grid
    data = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    grid = Grid(data)

    assert grid.height == 3
    assert grid.width == 3
    assert grid.get(0, 1) == 1

    # Test operations
    rotated = grid.rotate(1)
    assert rotated.height == 3
    assert rotated.width == 3

    flipped = grid.flip('horizontal')
    assert np.array_equal(flipped.data[0], grid.data[-1])

    print("✓ Grid tests passed")


def test_transformations():
    """Test DSL transformations."""
    print("Testing transformations...")

    grid = Grid(np.array([[1, 2], [3, 4]]))

    # Single transform
    rotated = Rotate(1).apply(grid)
    assert rotated.shape == (2, 2)

    # Composition
    transform = Compose([Rotate(1), Flip('horizontal')])
    result = transform.apply(grid)
    assert result.shape == (2, 2)

    # MDL calculation
    mdl = transform.description_length()
    assert mdl > 0

    print("✓ Transformation tests passed")


def test_semantic_priors():
    """Test semantic priors."""
    print("Testing semantic priors...")

    input_grid = Grid(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]]))
    output_grid = Grid(np.array([[0, 1, 1], [0, 1, 1], [2, 0, 0]]))

    # Test priors
    prior1 = PreserveObjectCount()
    score1 = prior1.evaluate(input_grid, output_grid)
    assert 0.0 <= score1 <= 1.0

    prior2 = PreserveColorPalette()
    score2 = prior2.evaluate(input_grid, output_grid)
    assert 0.0 <= score2 <= 1.0

    print("✓ Semantic prior tests passed")


def test_constraints():
    """Test constraint system."""
    print("Testing constraint system...")

    priors = [PreserveObjectCount(), PreserveColorPalette()]
    constraint_system = ConstraintSynthesizer.from_priors(priors)

    assert len(constraint_system.hard_constraints) > 0 or len(constraint_system.soft_constraints) > 0

    print("✓ Constraint tests passed")


def test_cellular_automata():
    """Test cellular automata."""
    print("Testing cellular automata...")

    # Create initial state
    initial = Grid(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]))

    # Run CA
    config = InterestingRules.get_rule("life")
    config.steps = 1
    ca = CellularAutomaton(config)

    result = ca.step(initial)
    assert result.shape == initial.shape

    print("✓ Cellular automata tests passed")


def test_puzzle_structure():
    """Test puzzle data structure."""
    print("Testing puzzle structure...")

    # Create simple puzzle
    input1 = Grid(np.array([[1, 2], [3, 4]]))
    output1 = Grid(np.array([[3, 1], [4, 2]]))

    input2 = Grid(np.array([[5, 6], [7, 8]]))
    output2 = Grid(np.array([[7, 5], [8, 6]]))

    puzzle = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[1, 1], [2, 2]]))],
        metadata={'test': True}
    )

    assert puzzle.num_train == 2
    assert puzzle.num_test == 1

    # Test serialization
    puzzle_dict = puzzle.to_dict()
    assert 'train' in puzzle_dict
    assert 'test' in puzzle_dict
    assert len(puzzle_dict['train']) == 2

    print("✓ Puzzle structure tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)

    tests = [
        test_grid_creation,
        test_transformations,
        test_semantic_priors,
        test_constraints,
        test_cellular_automata,
        test_puzzle_structure
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
