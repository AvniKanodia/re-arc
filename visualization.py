"""
Visualization Module for Puzzle PNG Generation

Creates beautiful, human-readable visualizations of puzzles with:
- Color-based grids
- Shape-based grids (circles, triangles, squares, etc.)
- Combined color + shape grids
- Clear input/output pairs
- Professional formatting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyBboxPatch, Wedge
from grid import Grid, Puzzle
from typing import List, Tuple, Optional
import os


# ARC-style color palette (visually distinct colors)
ARC_COLORS = {
    0: '#000000',  # Black (background)
    1: '#0074D9',  # Blue
    2: '#FF4136',  # Red
    3: '#2ECC40',  # Green
    4: '#FFDC00',  # Yellow
    5: '#AAAAAA',  # Gray
    6: '#F012BE',  # Magenta
    7: '#FF851B',  # Orange
    8: '#7FDBFF',  # Light Blue
    9: '#870C25',  # Brown/Maroon
}


# Shape types for shape-based puzzles
SHAPE_TYPES = {
    0: None,           # Empty
    1: 'square',       # Square
    2: 'circle',       # Circle
    3: 'triangle',     # Triangle (pointing up)
    4: 'diamond',      # Diamond
    5: 'hexagon',      # Hexagon
    6: 'star',         # Star
    7: 'cross',        # Plus/Cross
    8: 'heart',        # Heart
    9: 'pentagon',     # Pentagon
}


class PuzzleVisualizer:
    """Visualize puzzles as beautiful PNG images."""

    def __init__(self, cell_size: float = 0.8, figsize_per_cell: float = 0.6):
        """
        Args:
            cell_size: Size of each cell relative to grid spacing (0-1)
            figsize_per_cell: Figure size per grid cell (inches)
        """
        self.cell_size = cell_size
        self.figsize_per_cell = figsize_per_cell

    def visualize_puzzle(self, puzzle: Puzzle, output_path: str,
                        title: Optional[str] = None,
                        show_test_output: bool = True):
        """
        Create a comprehensive visualization of a puzzle.

        Args:
            puzzle: Puzzle to visualize
            output_path: Path to save PNG
            title: Optional title for the puzzle
            show_test_output: Whether to show test outputs (if available)
        """
        num_train = puzzle.num_train
        num_test = puzzle.num_test if show_test_output and puzzle.test_outputs else 0
        total_pairs = num_train + num_test

        # Calculate figure layout
        fig, axes = plt.subplots(total_pairs, 2, figsize=(
            12,
            6 * total_pairs
        ))

        if total_pairs == 1:
            axes = axes.reshape(1, -1)

        # Set main title
        if title:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)

        # Visualize training pairs
        for i, (input_grid, output_grid) in enumerate(puzzle.train_pairs):
            ax_in = axes[i, 0]
            ax_out = axes[i, 1]

            self._draw_grid(ax_in, input_grid, f"Training {i+1} - Input")
            self._draw_grid(ax_out, output_grid, f"Training {i+1} - Output")

        # Visualize test pairs
        if show_test_output and puzzle.test_outputs:
            for i, (test_input, test_output) in enumerate(
                zip(puzzle.test_inputs, puzzle.test_outputs)
            ):
                ax_in = axes[num_train + i, 0]
                ax_out = axes[num_train + i, 1]

                self._draw_grid(ax_in, test_input, f"Test {i+1} - Input")
                self._draw_grid(ax_out, test_output, f"Test {i+1} - Output")
        elif num_test > 0:
            # Show test input without output
            for i, test_input in enumerate(puzzle.test_inputs):
                ax_in = axes[num_train + i, 0]
                ax_out = axes[num_train + i, 1]

                self._draw_grid(ax_in, test_input, f"Test {i+1} - Input")
                ax_out.text(0.5, 0.5, '?', fontsize=100,
                          ha='center', va='center', color='gray', alpha=0.3)
                ax_out.set_xlim(0, 1)
                ax_out.set_ylim(0, 1)
                ax_out.axis('off')
                ax_out.set_title(f"Test {i+1} - Output", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _draw_grid(self, ax, grid: Grid, title: str):
        """Draw a single grid on an axis."""
        h, w = grid.height, grid.width

        # Set up axis
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Top-left origin
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Draw grid lines
        for i in range(h + 1):
            ax.axhline(i, color='lightgray', linewidth=0.5, alpha=0.5)
        for j in range(w + 1):
            ax.axvline(j, color='lightgray', linewidth=0.5, alpha=0.5)

        # Draw cells
        for r in range(h):
            for c in range(w):
                color_value = grid.get(r, c)
                self._draw_cell(ax, r, c, color_value)

        # Add dimensions label
        ax.text(w - 0.5, h + 0.3, f'{h}×{w}',
               fontsize=9, ha='right', va='top', color='gray')

    def _draw_cell(self, ax, row: int, col: int, color_value: int):
        """Draw a single cell."""
        if color_value == 0:
            # Background - draw light cell
            rect = Rectangle((col, row), 1, 1,
                           facecolor='#F0F0F0',
                           edgecolor='none')
            ax.add_patch(rect)
        else:
            # Colored cell
            color = ARC_COLORS.get(color_value, '#808080')
            rect = Rectangle((col + (1 - self.cell_size)/2,
                            row + (1 - self.cell_size)/2),
                           self.cell_size, self.cell_size,
                           facecolor=color,
                           edgecolor='white',
                           linewidth=1)
            ax.add_patch(rect)

    def visualize_grid_comparison(self, grids: List[Tuple[Grid, str]],
                                  output_path: str, title: str = "Grid Comparison"):
        """
        Visualize multiple grids in a row for comparison.

        Args:
            grids: List of (grid, label) tuples
            output_path: Path to save PNG
            title: Title for the comparison
        """
        num_grids = len(grids)
        fig, axes = plt.subplots(1, num_grids, figsize=(4 * num_grids, 4))

        if num_grids == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight='bold')

        for i, (grid, label) in enumerate(grids):
            self._draw_grid(axes[i], grid, label)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class ShapeVisualizer(PuzzleVisualizer):
    """Visualizer that renders shapes instead of colored squares."""

    def _draw_cell(self, ax, row: int, col: int, value: int):
        """Draw a cell with a shape."""
        if value == 0:
            # Background
            rect = Rectangle((col, row), 1, 1,
                           facecolor='#F5F5F5',
                           edgecolor='none')
            ax.add_patch(rect)
            return

        # Draw shape
        center_x = col + 0.5
        center_y = row + 0.5
        size = self.cell_size * 0.5

        shape_type = SHAPE_TYPES.get(value, 'square')
        color = ARC_COLORS.get(value, '#808080')

        if shape_type == 'square':
            rect = Rectangle((center_x - size/2, center_y - size/2),
                           size, size,
                           facecolor=color,
                           edgecolor='white',
                           linewidth=1.5)
            ax.add_patch(rect)

        elif shape_type == 'circle':
            circle = Circle((center_x, center_y), size/2,
                          facecolor=color,
                          edgecolor='white',
                          linewidth=1.5)
            ax.add_patch(circle)

        elif shape_type == 'triangle':
            triangle = RegularPolygon((center_x, center_y), 3, size/2,
                                    orientation=0,
                                    facecolor=color,
                                    edgecolor='white',
                                    linewidth=1.5)
            ax.add_patch(triangle)

        elif shape_type == 'diamond':
            diamond = RegularPolygon((center_x, center_y), 4, size/2,
                                   orientation=np.pi/4,
                                   facecolor=color,
                                   edgecolor='white',
                                   linewidth=1.5)
            ax.add_patch(diamond)

        elif shape_type == 'hexagon':
            hexagon = RegularPolygon((center_x, center_y), 6, size/2,
                                   facecolor=color,
                                   edgecolor='white',
                                   linewidth=1.5)
            ax.add_patch(hexagon)

        elif shape_type == 'star':
            self._draw_star(ax, center_x, center_y, size/2, color)

        elif shape_type == 'cross':
            self._draw_cross(ax, center_x, center_y, size/2, color)

        elif shape_type == 'pentagon':
            pentagon = RegularPolygon((center_x, center_y), 5, size/2,
                                    orientation=np.pi/2,
                                    facecolor=color,
                                    edgecolor='white',
                                    linewidth=1.5)
            ax.add_patch(pentagon)

        elif shape_type == 'heart':
            self._draw_heart(ax, center_x, center_y, size/2, color)

    def _draw_star(self, ax, x, y, size, color):
        """Draw a 5-pointed star."""
        angles = np.linspace(0, 2*np.pi, 11)
        radii = np.array([size, size*0.4] * 5 + [size])
        xs = x + radii * np.cos(angles - np.pi/2)
        ys = y + radii * np.sin(angles - np.pi/2)

        star = plt.Polygon(list(zip(xs, ys)),
                          facecolor=color,
                          edgecolor='white',
                          linewidth=1.5)
        ax.add_patch(star)

    def _draw_cross(self, ax, x, y, size, color):
        """Draw a cross/plus shape."""
        arm_width = size * 0.3
        rects = [
            Rectangle((x - arm_width/2, y - size), arm_width, size*2,
                     facecolor=color, edgecolor='white', linewidth=1.5),
            Rectangle((x - size, y - arm_width/2), size*2, arm_width,
                     facecolor=color, edgecolor='white', linewidth=1.5)
        ]
        for rect in rects:
            ax.add_patch(rect)

    def _draw_heart(self, ax, x, y, size, color):
        """Draw a heart shape (simplified)."""
        # Two circles and a triangle
        r = size * 0.25

        circle1 = Circle((x - r*0.8, y - r*0.5), r,
                        facecolor=color, edgecolor='none')
        circle2 = Circle((x + r*0.8, y - r*0.5), r,
                        facecolor=color, edgecolor='none')

        triangle_points = [
            (x - size*0.7, y - r*0.3),
            (x + size*0.7, y - r*0.3),
            (x, y + size*0.8)
        ]
        triangle = plt.Polygon(triangle_points,
                              facecolor=color,
                              edgecolor='white',
                              linewidth=1.5)

        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(triangle)


class ColorShapeVisualizer(PuzzleVisualizer):
    """
    Visualizer that uses both color AND shape.
    This creates the most interesting puzzles where both dimensions matter.
    """

    def __init__(self, cell_size: float = 0.8, figsize_per_cell: float = 0.6):
        super().__init__(cell_size, figsize_per_cell)
        self.shape_visualizer = ShapeVisualizer(cell_size, figsize_per_cell)

    def _draw_cell(self, ax, row: int, col: int, value: int):
        """
        Draw cell with shape determined by value, but color can vary.
        For now, we map value to both shape and color.
        """
        if value == 0:
            rect = Rectangle((col, row), 1, 1,
                           facecolor='#F5F5F5',
                           edgecolor='none')
            ax.add_patch(rect)
            return

        # Use shape drawing but with distinct styling
        self.shape_visualizer._draw_cell(ax, row, col, value)


def create_puzzle_grid_visualization(puzzle: Puzzle, output_dir: str,
                                     filename: str, visualizer_type: str = 'color'):
    """
    Helper function to create puzzle visualization.

    Args:
        puzzle: Puzzle to visualize
        output_dir: Directory to save image
        filename: Filename (without extension)
        visualizer_type: 'color', 'shape', or 'colorshape'
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.png")

    if visualizer_type == 'shape':
        visualizer = ShapeVisualizer()
    elif visualizer_type == 'colorshape':
        visualizer = ColorShapeVisualizer()
    else:
        visualizer = PuzzleVisualizer()

    # Create title with metadata
    metadata = puzzle.metadata or {}
    title_parts = []

    if 'strategy' in metadata:
        title_parts.append(f"Strategy: {metadata['strategy']}")

    if 'difficulty' in metadata:
        title_parts.append(f"Difficulty: {metadata['difficulty']}")

    title = " | ".join(title_parts) if title_parts else "ARC-Style Puzzle"

    visualizer.visualize_puzzle(puzzle, output_path, title=title)

    return output_path


def create_dataset_summary_visualization(puzzles: List[Puzzle],
                                        output_path: str,
                                        samples: int = 4):
    """
    Create a summary visualization showing samples from a dataset.

    Args:
        puzzles: List of puzzles
        output_path: Path to save summary image
        samples: Number of sample puzzles to show
    """
    import random

    if len(puzzles) > samples:
        selected = random.sample(puzzles, samples)
    else:
        selected = puzzles

    fig = plt.figure(figsize=(16, 4 * samples))
    gs = fig.add_gridspec(samples, 4, hspace=0.3, wspace=0.2)

    visualizer = PuzzleVisualizer()

    for i, puzzle in enumerate(selected):
        if puzzle.num_train > 0:
            # Show first training pair
            input_grid, output_grid = puzzle.train_pairs[0]

            ax_in = fig.add_subplot(gs[i, 0:2])
            ax_out = fig.add_subplot(gs[i, 2:4])

            visualizer._draw_grid(ax_in, input_grid, f"Puzzle {i+1} - Input")
            visualizer._draw_grid(ax_out, output_grid, f"Puzzle {i+1} - Output")

    fig.suptitle(f"Dataset Sample ({len(puzzles)} total puzzles)",
                fontsize=18, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
