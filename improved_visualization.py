"""
Improved Visualization Module with Better Aesthetics

Uses only the core ARC colors (0-8) that appear most frequently.
Larger cells, better grid lines, clearer labeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches as mpatches
from grid import Grid, Puzzle
from typing import List, Tuple, Optional
import os

# Core ARC color palette (most commonly used colors)
# Using only 0-8 for better consistency with actual ARC puzzles
CORE_ARC_COLORS = {
    0: '#0a0a0a',  # Black (background) - slightly lighter for visibility
    1: '#0074D9',  # Blue - primary
    2: '#FF4136',  # Red - primary
    3: '#2ECC40',  # Green - primary
    4: '#FFDC00',  # Yellow - primary
    5: '#AAAAAA',  # Gray - neutral
    6: '#F012BE',  # Magenta - accent
    7: '#FF851B',  # Orange - accent
    8: '#7FDBFF',  # Light Blue - accent
}

# Nice names for display
COLOR_NAMES = {
    0: 'Black', 1: 'Blue', 2: 'Red', 3: 'Green',
    4: 'Yellow', 5: 'Gray', 6: 'Magenta', 7: 'Orange', 8: 'Cyan'
}


class ImprovedVisualizer:
    """Enhanced visualizer with better aesthetics."""

    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size

    def visualize_puzzle(self, puzzle: Puzzle, output_path: str,
                        title: Optional[str] = None):
        """Create a beautiful puzzle visualization."""

        num_pairs = len(puzzle.train_pairs)

        # Create figure with better proportions
        fig = plt.figure(figsize=(14, 4.5 * num_pairs), facecolor='white')

        # Add overall title
        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold',
                        family='sans-serif', y=0.98)

        # Calculate grid layout
        for idx, (input_grid, output_grid) in enumerate(puzzle.train_pairs):
            # Input grid
            ax_in = plt.subplot(num_pairs, 2, idx * 2 + 1)
            self._draw_enhanced_grid(ax_in, input_grid,
                                    f"Example {idx + 1}: Input",
                                    is_input=True)

            # Output grid
            ax_out = plt.subplot(num_pairs, 2, idx * 2 + 2)
            self._draw_enhanced_grid(ax_out, output_grid,
                                    f"Example {idx + 1}: Output",
                                    is_input=False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def _draw_enhanced_grid(self, ax, grid: Grid, title: str, is_input: bool = True):
        """Draw a grid with enhanced visual quality."""
        h, w = grid.height, grid.width

        # Set up the axis
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Remove default spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set background
        ax.set_facecolor('#f8f8f8')

        # Add title with better styling
        title_color = '#2c3e50'
        ax.set_title(title, fontsize=13, fontweight='bold',
                    pad=15, color=title_color, family='sans-serif')

        # Draw grid background
        for r in range(h):
            for c in range(w):
                # Light gray cell background
                bg_rect = Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                                   facecolor='white',
                                   edgecolor='#e0e0e0',
                                   linewidth=0.5,
                                   zorder=1)
                ax.add_patch(bg_rect)

        # Draw colored cells with better styling
        for r in range(h):
            for c in range(w):
                color_value = grid.get(r, c)
                if color_value == 0:
                    # Background stays light
                    continue
                else:
                    # Draw colored cell
                    color = CORE_ARC_COLORS[color_value]

                    # Main colored rectangle (slightly inset for nice borders)
                    cell_rect = Rectangle((c - 0.42, r - 0.42), 0.84, 0.84,
                                         facecolor=color,
                                         edgecolor='white',
                                         linewidth=2,
                                         zorder=2)
                    ax.add_patch(cell_rect)

                    # Add subtle shadow for depth
                    shadow_rect = Rectangle((c - 0.44, r - 0.44), 0.84, 0.84,
                                           facecolor='black',
                                           alpha=0.1,
                                           zorder=1.5)
                    ax.add_patch(shadow_rect)

        # Add outer border
        border = Rectangle((-0.5, -0.5), w, h,
                          fill=False,
                          edgecolor='#34495e',
                          linewidth=2.5,
                          zorder=3)
        ax.add_patch(border)

        # Add dimension label
        dim_text = f"{h}×{w}"
        ax.text(w - 0.5, h + 0.2, dim_text,
               fontsize=9, ha='right', va='top',
               color='#7f8c8d', style='italic')

        # Add arrow between input and output
        if not is_input:
            # Add a subtle indicator
            ax.text(-0.8, h/2, '←', fontsize=20, ha='right', va='center',
                   color='#3498db', alpha=0.6)


def create_puzzle_visualization(puzzle: Puzzle, output_dir: str,
                               filename: str, rule: str = ""):
    """Helper to create improved visualization."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.png")

    # Create title with rule
    title = puzzle.metadata.get('name', 'Puzzle')
    if rule:
        title = f"{title}\n{rule}"

    visualizer = ImprovedVisualizer()
    visualizer.visualize_puzzle(puzzle, output_path, title=title)

    return output_path
