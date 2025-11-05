"""
Test visualization to ensure colors show correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid import Grid
from visualization import PuzzleVisualizer, ARC_COLORS

def test_color_palette():
    """Test that all colors display correctly."""
    print("Testing color palette...")

    # Create a grid with all colors
    grid_data = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ])
    grid = Grid(grid_data)

    print("\nGrid data:")
    print(grid_data)

    print("\nColors that should appear:")
    for i in range(10):
        color_name = {
            0: "Black (background)",
            1: "Blue",
            2: "Red",
            3: "Green",
            4: "Yellow",
            5: "Gray",
            6: "Magenta",
            7: "Orange",
            8: "Light Blue",
            9: "Brown/Maroon"
        }
        print(f"  {i}: {color_name[i]} - {ARC_COLORS[i]}")

    # Create visualization
    visualizer = PuzzleVisualizer()
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    visualizer._draw_grid(ax, grid, "Color Palette Test")

    plt.savefig("test_colors.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSaved test_colors.png")
    print("Check this file to verify colors are displaying correctly")

if __name__ == "__main__":
    test_color_palette()
