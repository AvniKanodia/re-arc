"""
Generate Puzzles using Graph State Space Mining

Automatically discovers interesting transformation sequences by exploring
the space of possible grid transformations.
"""

import os
import numpy as np
from state_space_mining import GridStateSpace, TransformationPath
from improved_visualization import create_puzzle_visualization
from grid import Grid, Puzzle


def generate_and_visualize_puzzles(
    num_puzzles: int = 10,
    output_dir: str = "mined_puzzles"
):
    """
    Generate puzzles using state space mining and create visualizations.
    """
    print("=" * 70)
    print("AUTOMATIC PUZZLE GENERATION VIA STATE SPACE MINING")
    print("=" * 70)
    print()
    print("Approach:")
    print("  1. Model grid transformations as a state space graph")
    print("  2. Mine for paths with high 'interestingness':")
    print("     - High visual difference (looks very different)")
    print("     - Moderate path length (not too trivial or arbitrary)")
    print("     - Low branching factor (few obvious moves)")
    print("  3. Generate training examples with variations")
    print()
    print("=" * 70)
    print()

    os.makedirs(output_dir, exist_ok=True)

    space = GridStateSpace(max_colors=8)
    generated_puzzles = []

    print(f"Mining for {num_puzzles} interesting puzzles...")
    print()

    attempts = 0
    max_attempts = num_puzzles * 5  # Try harder to find good puzzles

    while len(generated_puzzles) < num_puzzles and attempts < max_attempts:
        attempts += 1

        # Vary grid sizes for diversity
        grid_sizes = [(4, 4), (5, 5), (6, 6), (4, 6), (6, 4), (5, 6), (6, 5)]
        size = grid_sizes[attempts % len(grid_sizes)]

        # Generate puzzle
        puzzle = space.generate_puzzle(grid_size=size, num_examples=3)

        if puzzle is None:
            continue

        # Filter for quality
        score = puzzle.metadata['interestingness']
        visual_dist = puzzle.metadata['visual_distance']

        # Require minimum quality thresholds
        if score < 0.15 or visual_dist < 0.3:
            continue

        # Success!
        puzzle_num = len(generated_puzzles) + 1
        transformation = puzzle.metadata['transformation']
        path_length = puzzle.metadata['path_length']

        print(f"Puzzle {puzzle_num}/{num_puzzles} FOUND!")
        print(f"  Grid Size: {size[0]}×{size[1]}")
        print(f"  Transformation: {' → '.join(transformation[:3])}")
        if len(transformation) > 3:
            print(f"                  → {' → '.join(transformation[3:])}")
        print(f"  Path Length: {path_length}")
        print(f"  Interestingness Score: {score:.3f}")
        print(f"  Visual Distance: {visual_dist:.3f}")
        print()

        # Add metadata for visualization
        puzzle.metadata['name'] = f"Mined Puzzle #{puzzle_num}"
        puzzle.metadata['rule'] = f"Transformation: {' → '.join(transformation)}"

        generated_puzzles.append(puzzle)

    print("=" * 70)
    print(f"Successfully mined {len(generated_puzzles)} puzzles!")
    print("=" * 70)
    print()

    # Visualize all puzzles
    print("Creating visualizations...")
    print()

    for idx, puzzle in enumerate(generated_puzzles):
        puzzle_num = idx + 1
        filename = f"puzzle_{puzzle_num:02d}"

        # Create title with transformation and metrics
        trans = puzzle.metadata['transformation']
        score = puzzle.metadata['interestingness']

        title_parts = [
            f"Mined Puzzle #{puzzle_num}",
            f"Transformation: {' → '.join(trans)}",
            f"(Interestingness: {score:.2f})"
        ]
        title = "\n".join(title_parts)

        # Update puzzle metadata
        puzzle.metadata['name'] = title_parts[0]

        # Create visualization
        output_path = create_puzzle_visualization(
            puzzle,
            output_dir,
            filename,
            rule=title_parts[1]
        )

        print(f"  ✓ Saved: {filename}.png")

    print()
    print("=" * 70)
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 70)
    print()

    # Print summary statistics
    print("SUMMARY STATISTICS:")
    print()

    interestingness_scores = [p.metadata['interestingness'] for p in generated_puzzles]
    visual_distances = [p.metadata['visual_distance'] for p in generated_puzzles]
    path_lengths = [p.metadata['path_length'] for p in generated_puzzles]

    print(f"  Average Interestingness: {np.mean(interestingness_scores):.3f}")
    print(f"  Average Visual Distance: {np.mean(visual_distances):.3f}")
    print(f"  Average Path Length: {np.mean(path_lengths):.1f}")
    print()

    print("  Most Complex Transformation:")
    most_complex = max(generated_puzzles, key=lambda p: p.metadata['path_length'])
    print(f"    {' → '.join(most_complex.metadata['transformation'])}")
    print()

    print("  Highest Visual Change:")
    highest_visual = max(generated_puzzles, key=lambda p: p.metadata['visual_distance'])
    print(f"    {' → '.join(highest_visual.metadata['transformation'])}")
    print()

    print("=" * 70)
    print("COMPLETE!")
    print()
    print("Key Insight: These puzzles were automatically discovered by mining")
    print("the state space for transformations that are compositionally complex")
    print("but appear visually simple - the hallmark of genuinely hard puzzles.")
    print("=" * 70)

    return generated_puzzles


def analyze_puzzle_quality(puzzles):
    """Analyze the quality of generated puzzles."""
    print()
    print("=" * 70)
    print("QUALITY ANALYSIS")
    print("=" * 70)
    print()

    for idx, puzzle in enumerate(puzzles[:3], 1):
        print(f"Puzzle {idx}:")
        print(f"  Transformation sequence: {puzzle.metadata['transformation']}")
        print(f"  Why it's interesting:")

        score = puzzle.metadata['interestingness']
        visual_dist = puzzle.metadata['visual_distance']
        path_len = puzzle.metadata['path_length']

        reasons = []

        if visual_dist > 0.5:
            reasons.append(f"High visual change ({visual_dist:.2f})")

        if path_len >= 3:
            reasons.append(f"Multi-step transformation ({path_len} operations)")

        if score > 0.3:
            reasons.append(f"Low branching factor (non-obvious moves)")

        for reason in reasons:
            print(f"    • {reason}")

        print()


def main():
    """Main entry point."""
    np.random.seed(42)  # For reproducibility

    # Generate puzzles
    puzzles = generate_and_visualize_puzzles(num_puzzles=10, output_dir="mined_puzzles")

    # Analyze quality
    if puzzles:
        analyze_puzzle_quality(puzzles)


if __name__ == "__main__":
    main()
