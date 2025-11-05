"""
Quick Demo - Generate a small set of interesting puzzles quickly.
"""

import os
import numpy as np
from grid import Grid, Puzzle
from puzzle_generator import PuzzleGenerator, GeneratorConfig, GenerationStrategy
from visualization import create_puzzle_grid_visualization, PuzzleVisualizer
from custom_transforms import *
from custom_priors import *

def create_demo_puzzles():
    """Create a few hand-crafted interesting puzzles for demonstration."""
    puzzles = []

    # Puzzle 1: Tiling Pattern
    print("Creating Puzzle 1: Tiling Pattern...")
    input1 = Grid(np.array([
        [1, 2],
        [3, 4]
    ]))
    transform = TilePattern(2)
    output1 = transform.apply(input1)

    input2 = Grid(np.array([
        [5, 0],
        [0, 5]
    ]))
    output2 = transform.apply(input2)

    puzzle1 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[2, 3], [1, 2]]))],
        test_outputs=[transform.apply(Grid(np.array([[2, 3], [1, 2]])))],
        metadata={
            'name': 'Tiling Pattern',
            'strategy': 'custom_transform',
            'transform': 'tile_2x2',
            'difficulty': 'medium'
        }
    )
    puzzles.append(puzzle1)

    # Puzzle 2: Mirror Effect
    print("Creating Puzzle 2: Mirror Reflection...")
    input1 = Grid(np.array([
        [1, 0, 0],
        [1, 2, 0],
        [1, 2, 3]
    ]))
    transform = Mirror('right')
    output1 = transform.apply(input1)

    input2 = Grid(np.array([
        [4, 4, 0],
        [0, 4, 0],
        [0, 0, 0]
    ]))
    output2 = transform.apply(input2)

    puzzle2 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[2, 0, 0], [2, 3, 0], [2, 3, 4]]))],
        test_outputs=[transform.apply(Grid(np.array([[2, 0, 0], [2, 3, 0], [2, 3, 4]])))],
        metadata={
            'name': 'Mirror Reflection',
            'strategy': 'custom_transform',
            'transform': 'mirror_right',
            'difficulty': 'easy'
        }
    )
    puzzles.append(puzzle2)

    # Puzzle 3: Spiral Color Rotation
    print("Creating Puzzle 3: Spiral Color Rotation...")
    input1 = Grid(np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1]
    ]))
    transform = SpiralRotate()
    output1 = transform.apply(input1)

    input2 = Grid(np.array([
        [4, 4, 4],
        [4, 5, 4],
        [4, 4, 4]
    ]))
    output2 = transform.apply(input2)

    puzzle3 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]]))],
        test_outputs=[transform.apply(Grid(np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]])))],
        metadata={
            'name': 'Spiral Color Rotation',
            'strategy': 'custom_transform',
            'transform': 'spiral_rotate',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle3)

    # Puzzle 4: Color Gradient
    print("Creating Puzzle 4: Color Gradient...")
    input1 = Grid(np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]))
    transform = ColorGradient('horizontal')
    output1 = transform.apply(input1)

    input2 = Grid(np.array([
        [2, 0, 2],
        [0, 2, 0],
        [2, 0, 2]
    ]))
    output2 = transform.apply(input2)

    puzzle4 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[3, 3, 3], [3, 3, 3]]))],
        test_outputs=[transform.apply(Grid(np.array([[3, 3, 3], [3, 3, 3]])))],
        metadata={
            'name': 'Horizontal Color Gradient',
            'strategy': 'custom_transform',
            'transform': 'gradient_horizontal',
            'difficulty': 'medium'
        }
    )
    puzzles.append(puzzle4)

    # Puzzle 5: Diagonal Flip
    print("Creating Puzzle 5: Diagonal Flip...")
    input1 = Grid(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]))
    transform = DiagonalFlip()
    output1 = transform.apply(input1)

    puzzle5 = Puzzle(
        train_pairs=[(input1, output1)],
        test_inputs=[Grid(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))],
        test_outputs=[transform.apply(Grid(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])))],
        metadata={
            'name': 'Diagonal Flip',
            'strategy': 'custom_transform',
            'transform': 'diagonal_flip_anti',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle5)

    return puzzles


def visualize_demo_puzzles(puzzles, output_dir="demo_visualizations"):
    """Create visualizations for demo puzzles."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}/...")

    for i, puzzle in enumerate(puzzles):
        filename = f"demo_puzzle_{i+1}_{puzzle.metadata.get('name', 'unknown').replace(' ', '_').lower()}"

        output_path = create_puzzle_grid_visualization(
            puzzle,
            output_dir,
            filename,
            visualizer_type='color'
        )

        print(f"  ✓ Created: {filename}.png")

    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    print("=" * 70)
    print("QUICK DEMO - Creating Interesting Puzzles")
    print("=" * 70)
    print()

    # Create demo puzzles
    puzzles = create_demo_puzzles()

    print(f"\nCreated {len(puzzles)} demo puzzles:")
    for i, p in enumerate(puzzles):
        print(f"  {i+1}. {p.metadata.get('name')} - {p.metadata.get('difficulty')}")

    # Visualize
    visualize_demo_puzzles(puzzles)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nCheck the 'demo_visualizations' folder for PNG images")
    print("Each puzzle shows 2-3 training examples with input/output pairs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
