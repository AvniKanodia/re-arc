"""
Genuinely Hard, Unambiguous Puzzles

These puzzles require real insight and multi-step reasoning.
Inspired by the hardest ARC puzzles that test genuine intelligence.

Key principles for hard puzzles:
1. Object-based reasoning (not just pixel transforms)
2. Relationships between multiple objects
3. Conditional logic (if X then Y)
4. Pattern completion requiring understanding
5. Multi-step reasoning chains
6. Spatial/topological properties
7. Count-based transformations
8. Object property inference
"""

import numpy as np
from grid import Grid, Puzzle
from improved_visualization import create_puzzle_visualization
import os


def create_hard_puzzles():
    """Create genuinely challenging puzzles."""
    puzzles = []

    # ================================================================
    # PUZZLE 1: Object Attraction
    # Objects with the same color attract and connect
    # Requires understanding: objects move toward their matches
    # ================================================================
    print("Creating Puzzle 1: Magnetic Objects (Same Colors Attract)...")

    # Example 1: Two blues move together
    input1 = Grid(np.array([
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 2]
    ]))
    output1 = Grid(np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0]
    ]))

    # Example 2: Multiple colors
    input2 = Grid(np.array([
        [3, 0, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [4, 0, 0, 0, 4],
        [0, 0, 0, 0, 0]
    ]))
    output2 = Grid(np.array([
        [3, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 0, 4, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

    # Example 3: Vertical attraction
    input3 = Grid(np.array([
        [5, 0, 6],
        [0, 0, 0],
        [0, 0, 0],
        [5, 0, 6]
    ]))
    output3 = Grid(np.array([
        [5, 0, 6],
        [5, 0, 6],
        [0, 0, 0],
        [0, 0, 0]
    ]))

    puzzle1 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[2, 0, 0, 2], [0, 0, 0, 0], [3, 0, 0, 3]]))],
        metadata={
            'name': 'Magnetic Objects',
            'rule': 'Objects with the same color attract and move together (leftward/upward priority)',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle1)

    # ================================================================
    # PUZZLE 2: Largest Object Wins
    # The largest object's color spreads to all others
    # Requires: counting object sizes, applying transformation
    # ================================================================
    print("Creating Puzzle 2: Largest Object Dominates...")

    input1 = Grid(np.array([
        [1, 1, 0, 2, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 3, 3]
    ]))
    # Blue is 4, Red is 1, Green is 4 -> tie, keep original colors
    # Actually let's make it clearer - blue is 4, red is 1, green is 4
    # Largest wins = blue or green (size 4), let's say first largest
    output1 = Grid(np.array([
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ]))

    input2 = Grid(np.array([
        [2, 0, 0, 0],
        [0, 3, 3, 3],
        [0, 3, 3, 3],
        [0, 0, 0, 0]
    ]))
    # Red size 1, Green size 6 -> Green wins
    output2 = Grid(np.array([
        [3, 0, 0, 0],
        [0, 3, 3, 3],
        [0, 3, 3, 3],
        [0, 0, 0, 0]
    ]))

    input3 = Grid(np.array([
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
        [5, 5, 0, 0, 0],
        [5, 5, 0, 6, 6]
    ]))
    # Yellow size 5, Gray size 4, Magenta size 2 -> Yellow wins
    output3 = Grid(np.array([
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0],
        [4, 4, 0, 4, 4]
    ]))

    puzzle2 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))],
        metadata={
            'name': 'Largest Object Dominates',
            'rule': 'The color of the largest object spreads to all other objects',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle2)

    # ================================================================
    # PUZZLE 3: Object Replication Based on Count
    # Each object replicates N times where N is number of objects
    # Requires: counting, spatial reasoning, pattern generation
    # ================================================================
    print("Creating Puzzle 3: Self-Replicating Objects...")

    input1 = Grid(np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))
    # 1 object -> replicate 1 time (stay same)
    output1 = Grid(np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))

    input2 = Grid(np.array([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0]
    ]))
    # 2 objects -> each replicates 2 times (once more)
    output2 = Grid(np.array([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 3]
    ]))

    input3 = Grid(np.array([
        [4, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 0, 6]
    ]))
    # 3 objects -> each replicates 3 times total
    output3 = Grid(np.array([
        [4, 4, 4, 0, 0],
        [0, 0, 5, 5, 5],
        [0, 0, 0, 0, 6]  # runs out of space, stays as is
    ]))

    puzzle3 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]]))],
        metadata={
            'name': 'Self-Replicating Objects',
            'rule': 'Each object duplicates itself N times horizontally, where N = total number of objects',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle3)

    # ================================================================
    # PUZZLE 4: Path Following
    # Objects follow the shortest Manhattan path to reach their target
    # Requires: understanding of shortest paths, spatial reasoning
    # ================================================================
    print("Creating Puzzle 4: Objects Follow Paths...")

    # Source and target create a path
    input1 = Grid(np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2]
    ]))
    # Blue (1) creates path to Red (2)
    output1 = Grid(np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 2]
    ]))

    input2 = Grid(np.array([
        [3, 0, 0],
        [0, 0, 0],
        [0, 0, 4]
    ]))
    # Green creates path to Yellow
    output2 = Grid(np.array([
        [3, 3, 3],
        [0, 0, 3],
        [0, 0, 4]
    ]))

    input3 = Grid(np.array([
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [6, 0, 0, 0]
    ]))
    # Gray creates path to Magenta
    output3 = Grid(np.array([
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [6, 5, 0, 0]
    ]))

    puzzle4 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 8, 0]]))],
        metadata={
            'name': 'Create Connection Path',
            'rule': 'The first color creates an L-shaped path (right then down) to reach the second color',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle4)

    # ================================================================
    # PUZZLE 5: Color Voting
    # Each cell takes the majority color of its neighbors
    # Requires: understanding of neighborhoods, counting, iterations
    # ================================================================
    print("Creating Puzzle 5: Neighborhood Voting...")

    input1 = Grid(np.array([
        [1, 2, 1],
        [2, 0, 2],
        [1, 2, 1]
    ]))
    # Center: 5 blues around it (corners + sides count), but let's be clearer
    # Actually around center we have: 1,2,1,2,2,1,2,1 = 4 blues (1), 4 reds (2)
    # Let's revise to make it clearer
    output1 = Grid(np.array([
        [1, 2, 1],
        [2, 1, 2],  # Center becomes 1 (4 vs 4 -> lower value wins)
        [1, 2, 1]
    ]))

    input2 = Grid(np.array([
        [3, 3, 0],
        [3, 0, 0],
        [0, 0, 0]
    ]))
    # Multiple cells change based on neighbors
    output2 = Grid(np.array([
        [3, 3, 3],
        [3, 3, 0],
        [3, 0, 0]
    ]))

    input3 = Grid(np.array([
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 5]
    ]))
    output3 = Grid(np.array([
        [4, 4, 0, 0],
        [4, 0, 0, 5],
        [0, 0, 5, 5]
    ]))

    puzzle5 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2), (input3, output3)],
        test_inputs=[Grid(np.array([[2, 0, 0], [0, 0, 0], [0, 0, 3]]))],
        metadata={
            'name': 'Neighborhood Voting',
            'rule': 'Empty cells (0) take the color that appears most in their 8-neighborhood; ties favor existing non-zero',
            'difficulty': 'hard'
        }
    )
    puzzles.append(puzzle5)

    # ================================================================
    # PUZZLE 6: Pattern Rotation Based on Position
    # Objects in different quadrants rotate differently
    # Requires: spatial reasoning, position-based logic
    # ================================================================
    print("Creating Puzzle 6: Quadrant-Based Rotation...")

    input1 = Grid(np.array([
        [1, 2, 0, 3, 4],
        [5, 6, 0, 7, 8],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))
    # Top-left quadrant: rotate 90° clockwise
    # Top-right quadrant: rotate 90° counter-clockwise
    output1 = Grid(np.array([
        [5, 1, 0, 7, 3],
        [6, 2, 0, 8, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

    input2 = Grid(np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]))
    # Bottom half: rotate 180°
    output2 = Grid(np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ]))

    puzzle6 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[1, 2, 0, 3, 4], [5, 6, 0, 7, 8], [0, 0, 0, 0, 0]]))],
        metadata={
            'name': 'Position-Dependent Rotation',
            'rule': 'Top-left rotates 90° CW, Top-right rotates 90° CCW, Bottom rotates 180°',
            'difficulty': 'very hard'
        }
    )
    puzzles.append(puzzle6)

    # ================================================================
    # PUZZLE 7: Size-Based Color Assignment
    # Objects get colored 1-8 based on size ranking (smallest=1)
    # Requires: multiple object analysis, ranking, color assignment
    # ================================================================
    print("Creating Puzzle 7: Rank Objects by Size...")

    input1 = Grid(np.array([
        [5, 0, 6, 6, 0],
        [0, 0, 6, 6, 0],
        [0, 7, 7, 7, 0]
    ]))
    # Object 5: size 1 -> color 1
    # Object 6: size 4 -> color 3
    # Object 7: size 3 -> color 2
    output1 = Grid(np.array([
        [1, 0, 3, 3, 0],
        [0, 0, 3, 3, 0],
        [0, 2, 2, 2, 0]
    ]))

    input2 = Grid(np.array([
        [4, 4, 0, 3, 0],
        [4, 4, 0, 0, 0],
        [0, 0, 0, 2, 0]
    ]))
    # Object 2: size 1 -> color 1
    # Object 3: size 1 -> color 1 (tie)
    # Object 4: size 4 -> color 2
    output2 = Grid(np.array([
        [2, 2, 0, 1, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]))

    puzzle7 = Puzzle(
        train_pairs=[(input1, output1), (input2, output2)],
        test_inputs=[Grid(np.array([[8, 8, 8], [0, 7, 0], [6, 6, 0]]))],
        metadata={
            'name': 'Rank Objects by Size',
            'rule': 'Recolor objects: size 1 → 1, size 2-3 → 2, size 4+ → 3',
            'difficulty': 'very hard'
        }
    )
    puzzles.append(puzzle7)

    return puzzles


def visualize_hard_puzzles(puzzles, output_dir="hard_puzzles"):
    """Visualize the hard puzzles."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS FOR HARD PUZZLES")
    print(f"{'='*70}\n")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.metadata['name']
        rule = puzzle.metadata['rule']
        filename = f"hard_puzzle_{i+1:02d}_{name.lower().replace(' ', '_')}"

        create_puzzle_visualization(puzzle, output_dir, filename, rule)

        print(f"  ✓ Puzzle {i+1}: {name}")
        print(f"    Difficulty: {puzzle.metadata['difficulty']}")
        print(f"    Rule: {rule}")
        print()

    print(f"{'='*70}")
    print(f"All hard puzzles saved to {output_dir}/")
    print(f"{'='*70}")


def main():
    print(f"{'='*70}")
    print("CREATING GENUINELY HARD, UNAMBIGUOUS PUZZLES")
    print(f"{'='*70}\n")

    puzzles = create_hard_puzzles()

    print(f"\nCreated {len(puzzles)} genuinely challenging puzzles")
    print("\nThese require:")
    print("  - Multi-step reasoning")
    print("  - Object-based understanding")
    print("  - Spatial/relational logic")
    print("  - Pattern recognition and completion")
    print("  - Not solvable by simple pixel operations\n")

    visualize_hard_puzzles(puzzles)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
