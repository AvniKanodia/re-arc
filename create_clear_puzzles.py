"""
Create clear, logical puzzles that make perfect sense to humans.
Each puzzle demonstrates ONE clear transformation rule.
"""

import os
import numpy as np
from grid import Grid, Puzzle
from dsl import *
from custom_transforms import *
from visualization import create_puzzle_grid_visualization

def create_puzzles():
    """Create a set of clear, logical puzzles."""
    puzzles = []

    # ===================================================================
    # PUZZLE 1: Rotate 90 degrees clockwise
    # ===================================================================
    print("Creating Puzzle 1: Rotate 90° Clockwise...")

    # Training example 1
    input1 = Grid(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]))
    output1 = Rotate(1).apply(input1)

    # Training example 2
    input2 = Grid(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]))
    output2 = Rotate(1).apply(input2)

    # Training example 3
    input3 = Grid(np.array([
        [2, 2],
        [3, 3]
    ]))
    output3 = Rotate(1).apply(input3)

    puzzle1 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 2], [3, 4]]))],
        metadata={
            'name': 'Rotate 90° Clockwise',
            'difficulty': 'easy',
            'rule': 'Rotate the entire grid 90 degrees clockwise'
        }
    )
    puzzles.append(puzzle1)

    # ===================================================================
    # PUZZLE 2: Mirror Horizontally (Flip)
    # ===================================================================
    print("Creating Puzzle 2: Mirror Horizontally...")

    input1 = Grid(np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]))
    output1 = Flip('vertical').apply(input1)

    input2 = Grid(np.array([
        [1, 0, 0],
        [2, 2, 0],
        [3, 3, 3]
    ]))
    output2 = Flip('vertical').apply(input2)

    input3 = Grid(np.array([
        [5, 6],
        [7, 8]
    ]))
    output3 = Flip('vertical').apply(input3)

    puzzle2 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 2], [3, 4]]))],
        metadata={
            'name': 'Mirror Horizontally',
            'difficulty': 'easy',
            'rule': 'Flip the grid horizontally (left becomes right)'
        }
    )
    puzzles.append(puzzle2)

    # ===================================================================
    # PUZZLE 3: Tile Pattern 2x2
    # ===================================================================
    print("Creating Puzzle 3: Tile Pattern 2x2...")

    input1 = Grid(np.array([
        [1, 2],
        [3, 4]
    ]))
    output1 = TilePattern(2).apply(input1)

    input2 = Grid(np.array([
        [5, 0],
        [0, 6]
    ]))
    output2 = TilePattern(2).apply(input2)

    puzzle3 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[7, 8], [9, 1]]))],
        metadata={
            'name': 'Tile Pattern 2×2',
            'difficulty': 'medium',
            'rule': 'Repeat the pattern in a 2x2 grid'
        }
    )
    puzzles.append(puzzle3)

    # ===================================================================
    # PUZZLE 4: Swap Two Colors
    # ===================================================================
    print("Creating Puzzle 4: Swap Colors 1 and 2...")

    input1 = Grid(np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4]
    ]))
    output1 = SwapColors(1, 2).apply(input1)

    input2 = Grid(np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1]
    ]))
    output2 = SwapColors(1, 2).apply(input2)

    input3 = Grid(np.array([
        [1, 0, 2],
        [2, 0, 1]
    ]))
    output3 = SwapColors(1, 2).apply(input3)

    puzzle4 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 1], [2, 2]]))],
        metadata={
            'name': 'Swap Colors 1 and 2',
            'difficulty': 'easy',
            'rule': 'Swap all 1s with 2s and all 2s with 1s'
        }
    )
    puzzles.append(puzzle4)

    # ===================================================================
    # PUZZLE 5: Add Border
    # ===================================================================
    print("Creating Puzzle 5: Add Color Border...")

    input1 = Grid(np.array([
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0]
    ]))
    # Add border by replacing edge 0s with 1
    output1_data = input1.data.copy()
    # Top and bottom rows
    for c in range(output1_data.shape[1]):
        if output1_data[0, c] == 0:
            output1_data[0, c] = 1
        if output1_data[-1, c] == 0:
            output1_data[-1, c] = 1
    # Left and right columns
    for r in range(output1_data.shape[0]):
        if output1_data[r, 0] == 0:
            output1_data[r, 0] = 1
        if output1_data[r, -1] == 0:
            output1_data[r, -1] = 1
    output1 = Grid(output1_data)

    input2 = Grid(np.array([
        [0, 0, 0, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 0, 0, 0]
    ]))
    output2_data = input2.data.copy()
    for c in range(output2_data.shape[1]):
        if output2_data[0, c] == 0:
            output2_data[0, c] = 1
        if output2_data[-1, c] == 0:
            output2_data[-1, c] = 1
    for r in range(output2_data.shape[0]):
        if output2_data[r, 0] == 0:
            output2_data[r, 0] = 1
        if output2_data[r, -1] == 0:
            output2_data[r, -1] = 1
    output2 = Grid(output2_data)

    puzzle5 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]))],
        metadata={
            'name': 'Add Color Border',
            'difficulty': 'medium',
            'rule': 'Color all edge 0s with color 1'
        }
    )
    puzzles.append(puzzle5)

    # ===================================================================
    # PUZZLE 6: Horizontal Gradient
    # ===================================================================
    print("Creating Puzzle 6: Horizontal Color Gradient...")

    input1 = Grid(np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]))
    output1 = ColorGradient('horizontal').apply(input1)

    input2 = Grid(np.array([
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
    ]))
    output2 = ColorGradient('horizontal').apply(input2)

    puzzle6 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[3, 3, 3, 3], [3, 3, 3, 3]]))],
        metadata={
            'name': 'Horizontal Color Gradient',
            'difficulty': 'medium',
            'rule': 'Increase color by 1 for each column to the right'
        }
    )
    puzzles.append(puzzle6)

    # ===================================================================
    # PUZZLE 7: Count and Replace
    # ===================================================================
    print("Creating Puzzle 7: Fill Grid with Most Common Color...")

    input1 = Grid(np.array([
        [1, 1, 1, 2],
        [1, 1, 2, 2],
        [1, 3, 3, 0]
    ]))
    # Most common is 1 (appears 6 times)
    output1 = Grid(np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ]))

    input2 = Grid(np.array([
        [2, 2, 2],
        [3, 3, 0],
        [3, 3, 0]
    ]))
    # Most common is 3 (appears 4 times)
    output2 = Grid(np.array([
        [3, 3, 3],
        [3, 3, 0],
        [3, 3, 0]
    ]))

    puzzle7 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[5, 5, 6], [5, 7, 7], [0, 0, 0]]))],
        metadata={
            'name': 'Most Common Color Wins',
            'difficulty': 'hard',
            'rule': 'Replace all non-zero colors with the most common color (ignore 0)'
        }
    )
    puzzles.append(puzzle7)

    return puzzles


def visualize_puzzles(puzzles, output_dir="clear_puzzles"):
    """Create visualizations for all puzzles."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Creating Visualizations")
    print(f"{'='*60}\n")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.metadata.get('name', f'Puzzle {i+1}')
        filename = f"puzzle_{i+1:02d}_{name.lower().replace(' ', '_')}"

        output_path = create_puzzle_grid_visualization(
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
    print("CREATING CLEAR, LOGICAL PUZZLES")
    print(f"{'='*60}\n")

    puzzles = create_puzzles()

    print(f"\nCreated {len(puzzles)} puzzles:")
    for i, p in enumerate(puzzles):
        print(f"  {i+1}. {p.metadata.get('name')} ({p.metadata.get('difficulty')})")

    visualize_puzzles(puzzles)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
