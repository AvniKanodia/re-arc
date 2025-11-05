"""
Test transformations to ensure they work correctly and make sense.
"""

import numpy as np
from grid import Grid
from dsl import *
from custom_transforms import *

def test_transform(name, transform, input_grid):
    """Test a transform and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    print("\nInput:")
    print_grid(input_grid)

    try:
        output = transform.apply(input_grid)
        print("\nOutput:")
        print_grid(output)
        print(f"✓ Transform successful")
        return True
    except Exception as e:
        print(f"\n✗ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_grid(grid):
    """Print grid in readable format."""
    for row in grid.data:
        print(' '.join(str(x) for x in row))

def create_simple_test_grid():
    """Create a simple test grid that's easy to understand."""
    return Grid(np.array([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 0]
    ]))

def create_pattern_grid():
    """Create a grid with a clear pattern."""
    return Grid(np.array([
        [1, 2],
        [3, 4]
    ]))

def create_symmetric_grid():
    """Create a symmetric grid."""
    return Grid(np.array([
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ]))

def main():
    print("="*60)
    print("TRANSFORMATION TESTING")
    print("="*60)

    # Test basic DSL transforms
    print("\n" + "="*60)
    print("BASIC TRANSFORMS")
    print("="*60)

    test_grid = create_simple_test_grid()

    test_transform("Rotate 90°", Rotate(1), test_grid)
    test_transform("Flip Horizontal", Flip('horizontal'), test_grid)
    test_transform("Flip Vertical", Flip('vertical'), test_grid)

    # Test custom transforms
    print("\n" + "="*60)
    print("CUSTOM TRANSFORMS")
    print("="*60)

    # TilePattern
    small_grid = create_pattern_grid()
    test_transform("Tile 2x2", TilePattern(2), small_grid)

    # Mirror
    test_transform("Mirror Right", Mirror('right'), test_grid)

    # ColorMap
    test_transform("Swap Colors 1<->2", SwapColors(1, 2), test_grid)

    # Diagonal Flip
    square_grid = Grid(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]))
    test_transform("Diagonal Flip", DiagonalFlip(), square_grid)

    # Color Gradient
    uniform_grid = Grid(np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]))
    test_transform("Horizontal Gradient", ColorGradient('horizontal'), uniform_grid)

if __name__ == "__main__":
    main()
