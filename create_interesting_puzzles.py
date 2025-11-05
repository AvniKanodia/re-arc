"""
Create interesting, complex puzzles that require insight but are still logical.
These are more challenging than basic transforms but the rules make sense.
"""

import os
import numpy as np
from grid import Grid, Puzzle
from dsl import *
from custom_transforms import *
from visualization import create_puzzle_grid_visualization


def create_puzzles():
    """Create interesting, challenging but logical puzzles."""
    puzzles = []

    # ===================================================================
    # PUZZLE 1: Objects Get Outlined
    # Complex: Finds objects and draws outlines around them
    # ===================================================================
    print("Creating Puzzle 1: Object Outlining...")

    input1 = Grid(np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 3, 0, 0],
        [0, 3, 3, 0, 0],
        [0, 0, 0, 0, 0]
    ]))
    # Manually create outline (1s around the 3s)
    output1 = Grid(np.array([
        [0, 1, 1, 1, 0],
        [1, 3, 3, 1, 0],
        [1, 3, 3, 1, 0],
        [0, 1, 1, 1, 0]
    ]))

    input2 = Grid(np.array([
        [0, 0, 0, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 0]
    ]))
    output2 = Grid(np.array([
        [0, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 0],
        [2, 2, 1, 0]
    ]))

    puzzle1 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]]))],
        metadata={
            'name': 'Draw Outlines Around Objects',
            'difficulty': 'hard',
            'rule': 'Add a border of 1s around each non-zero object'
        }
    )
    puzzles.append(puzzle1)

    # ===================================================================
    # PUZZLE 2: Size-Based Coloring
    # Objects get colored based on their size
    # ===================================================================
    print("Creating Puzzle 2: Color By Size...")

    input1 = Grid(np.array([
        [5, 0, 0, 5, 5],
        [0, 0, 0, 5, 5],
        [6, 6, 6, 0, 0]
    ]))
    # Small object (size 1) -> color 1
    # Medium object (size 4) -> color 2
    # Large object (size 3) -> color 2
    output1 = Grid(np.array([
        [1, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
        [2, 2, 2, 0, 0]
    ]))

    input2 = Grid(np.array([
        [7, 7, 0, 0],
        [0, 0, 0, 8],
        [9, 9, 9, 0]
    ]))
    output2 = Grid(np.array([
        [2, 2, 0, 0],
        [0, 0, 0, 1],
        [2, 2, 2, 0]
    ]))

    puzzle2 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[3, 0, 4, 4], [0, 0, 4, 4]]))],
        metadata={
            'name': 'Color Objects By Size',
            'difficulty': 'hard',
            'rule': 'Size 1 object -> color 1, Size 2+ -> color 2'
        }
    )
    puzzles.append(puzzle2)

    # ===================================================================
    # PUZZLE 3: Pattern Completion (Symmetry)
    # Complete the pattern to make it symmetric
    # ===================================================================
    print("Creating Puzzle 3: Complete to Symmetry...")

    # Half pattern becomes full symmetric pattern
    input1 = Grid(np.array([
        [1, 2, 0, 0, 0],
        [3, 4, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))
    output1 = Grid(np.array([
        [1, 2, 0, 2, 1],
        [3, 4, 0, 4, 3],
        [0, 0, 0, 0, 0]
    ]))

    input2 = Grid(np.array([
        [5, 0, 0],
        [6, 7, 0],
        [0, 0, 0]
    ]))
    output2 = Grid(np.array([
        [5, 0, 5],
        [6, 7, 6],
        [0, 0, 0]
    ]))

    puzzle3 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[8, 9, 0, 0], [0, 0, 0, 0]]))],
        metadata={
            'name': 'Complete to Horizontal Symmetry',
            'difficulty': 'medium',
            'rule': 'Mirror the left half to the right to create symmetry'
        }
    )
    puzzles.append(puzzle3)

    # ===================================================================
    # PUZZLE 4: Gravity with Stacking
    # Objects fall down and stack on each other
    # ===================================================================
    print("Creating Puzzle 4: Gravity Simulation...")

    input1 = Grid(np.array([
        [1, 0, 2],
        [0, 0, 0],
        [0, 3, 0],
        [0, 0, 0]
    ]))
    output1 = Grid(np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 3, 0],
        [1, 0, 2]
    ]))

    input2 = Grid(np.array([
        [4, 5, 0],
        [0, 0, 6],
        [0, 0, 0]
    ]))
    output2 = Grid(np.array([
        [0, 0, 0],
        [0, 0, 6],
        [4, 5, 0]
    ]))

    puzzle4 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[7, 0], [0, 8], [0, 0]]))],
        metadata={
            'name': 'Apply Gravity',
            'difficulty': 'medium',
            'rule': 'All non-zero cells fall to the bottom of their column'
        }
    )
    puzzles.append(puzzle4)

    # ===================================================================
    # PUZZLE 5: Diagonal Pattern Propagation
    # Pattern spreads diagonally
    # ===================================================================
    print("Creating Puzzle 5: Diagonal Spreading...")

    input1 = Grid(np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2]
    ]))
    output1 = Grid(np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 2, 2],
        [0, 0, 2, 2]
    ]))

    input2 = Grid(np.array([
        [3, 0, 0],
        [0, 0, 0],
        [0, 0, 4]
    ]))
    output2 = Grid(np.array([
        [3, 3, 3],
        [3, 3, 4],
        [3, 4, 4]
    ]))

    puzzle5 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[5, 0], [0, 6]]))],
        metadata={
            'name': 'Diagonal Color Spreading',
            'difficulty': 'hard',
            'rule': 'Colors spread diagonally from their source'
        }
    )
    puzzles.append(puzzle5)

    # ===================================================================
    # PUZZLE 6: Count and Multiply
    # Output repeats input N times where N = number of non-zero colors
    # ===================================================================
    print("Creating Puzzle 6: Repeat By Color Count...")

    input1 = Grid(np.array([
        [1, 2]
    ]))
    # 2 colors, so repeat 2 times vertically
    output1 = Grid(np.array([
        [1, 2],
        [1, 2]
    ]))

    input2 = Grid(np.array([
        [3, 4, 5]
    ]))
    # 3 colors, repeat 3 times
    output2 = Grid(np.array([
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5]
    ]))

    input3 = Grid(np.array([
        [6, 0]
    ]))
    # 1 color, repeat 1 time
    output3 = Grid(np.array([
        [6, 0]
    ]))

    puzzle6 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[7, 8, 9, 1]]))],
        metadata={
            'name': 'Repeat By Unique Color Count',
            'difficulty': 'hard',
            'rule': 'Repeat the grid vertically N times, where N = number of unique non-zero colors'
        }
    )
    puzzles.append(puzzle6)

    # ===================================================================
    # PUZZLE 7: Corners to Center
    # Corner values determine center
    # ===================================================================
    print("Creating Puzzle 7: Corners Determine Center...")

    input1 = Grid(np.array([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 4]
    ]))
    # Sum of corners: 1+2+3+4 = 10, mod 10 = 0
    # But that's boring, so let's use max corner value
    output1 = Grid(np.array([
        [1, 0, 2],
        [0, 4, 0],
        [3, 0, 4]
    ]))

    input2 = Grid(np.array([
        [5, 0, 6],
        [0, 0, 0],
        [7, 0, 8]
    ]))
    # Max corner = 8
    output2 = Grid(np.array([
        [5, 0, 6],
        [0, 8, 0],
        [7, 0, 8]
    ]))

    input3 = Grid(np.array([
        [2, 0, 2],
        [0, 0, 0],
        [2, 0, 2]
    ]))
    # Max = 2
    output3 = Grid(np.array([
        [2, 0, 2],
        [0, 2, 0],
        [2, 0, 2]
    ]))

    puzzle7 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 0, 3], [0, 0, 0], [2, 0, 5]]))],
        metadata={
            'name': 'Max Corner to Center',
            'difficulty': 'hard',
            'rule': 'Place the maximum corner value in the center cell'
        }
    )
    puzzles.append(puzzle7)

    return puzzles


def visualize_puzzles(puzzles, output_dir="interesting_puzzles"):
    """Create visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Creating Visualizations")
    print(f"{'='*60}\n")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.metadata.get('name', f'Puzzle {i+1}')
        filename = f"puzzle_{i+1:02d}_{name.lower().replace(' ', '_')}"

        create_puzzle_grid_visualization(
            puzzle,
            output_dir,
            filename,
            visualizer_type='color'
        )

        print(f"  ✓ {name}")
        print(f"    Rule: {puzzle.metadata.get('rule')}")
        print(f"    Difficulty: {puzzle.metadata.get('difficulty')}")
        print()

    print(f"{'='*60}")
    print(f"All puzzles saved to {output_dir}/")
    print(f"{'='*60}")


def main():
    print(f"{'='*60}")
    print("CREATING INTERESTING, COMPLEX PUZZLES")
    print(f"{'='*60}\n")

    puzzles = create_puzzles()

    print(f"\nCreated {len(puzzles)} challenging puzzles:")
    for i, p in enumerate(puzzles):
        print(f"  {i+1}. {p.metadata.get('name')} ({p.metadata.get('difficulty')})")

    visualize_puzzles(puzzles)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"All puzzles require insight but have clear logical rules")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
