"""
Custom Transformations - Extended DSL Operations

This module adds domain-specific transformations that create more
interesting and complex puzzle patterns beyond the basic primitives.
"""

import numpy as np
from dsl import Transform, TransformType
from grid import Grid, Object
from typing import Tuple, List


class TilePattern(Transform):
    """Tile a pattern across the grid."""

    def __init__(self, tile_factor: int = 2):
        super().__init__(f"tile_{tile_factor}x{tile_factor}", TransformType.GEOMETRIC)
        self.tile_factor = tile_factor

    def apply(self, grid: Grid) -> Grid:
        # Create a larger grid by tiling the input
        new_height = grid.height * self.tile_factor
        new_width = grid.width * self.tile_factor
        new_data = np.zeros((new_height, new_width), dtype=np.int8)

        for i in range(self.tile_factor):
            for j in range(self.tile_factor):
                start_r = i * grid.height
                start_c = j * grid.width
                new_data[start_r:start_r + grid.height,
                        start_c:start_c + grid.width] = grid.data

        return Grid(new_data)

    def description_length(self) -> int:
        return 2


class DiagonalFlip(Transform):
    """Flip along the anti-diagonal (top-right to bottom-left)."""

    def __init__(self):
        super().__init__("diagonal_flip_anti", TransformType.GEOMETRIC)

    def apply(self, grid: Grid) -> Grid:
        if grid.height != grid.width:
            # Pad to square
            max_dim = max(grid.height, grid.width)
            padded = np.zeros((max_dim, max_dim), dtype=np.int8)
            padded[:grid.height, :grid.width] = grid.data
        else:
            padded = grid.data

        # Flip along anti-diagonal: rotate 90° then transpose
        flipped = np.rot90(padded.T, k=2)

        # Crop back to original size
        return Grid(flipped[:grid.height, :grid.width])

    def description_length(self) -> int:
        return 2


class SpiralRotate(Transform):
    """Apply different rotations in a spiral pattern from center."""

    def __init__(self):
        super().__init__("spiral_rotate", TransformType.GEOMETRIC)

    def apply(self, grid: Grid) -> Grid:
        new_grid = grid.copy()
        center_r, center_c = grid.height // 2, grid.width // 2

        for r in range(grid.height):
            for c in range(grid.width):
                # Distance from center determines rotation
                dist = max(abs(r - center_r), abs(c - center_c))

                # Map distance to color rotation
                if dist > 0:
                    color = grid.get(r, c)
                    if color != 0:
                        new_color = (color + dist) % 10
                        new_grid.set(r, c, new_color)

        return new_grid

    def description_length(self) -> int:
        return 3


class FillEnclosed(Transform):
    """Fill enclosed regions with a specific color."""

    def __init__(self, fill_color: int = 1, background: int = 0):
        super().__init__(f"fill_enclosed_{fill_color}", TransformType.SPATIAL)
        self.fill_color = fill_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        from scipy.ndimage import binary_fill_holes

        new_grid = grid.copy()

        # For each non-background color, find and fill holes
        for color in range(1, 10):
            if color == self.background:
                continue

            mask = grid.data == color
            if not mask.any():
                continue

            # Fill holes in this color's regions
            filled = binary_fill_holes(mask)

            # Mark newly filled cells
            newly_filled = filled & ~mask
            new_grid.data[newly_filled] = self.fill_color

        return new_grid

    def description_length(self) -> int:
        return 3


class Mirror(Transform):
    """Mirror the grid in a direction (creates double-size output)."""

    def __init__(self, direction: str = 'right'):
        super().__init__(f"mirror_{direction}", TransformType.GEOMETRIC)
        self.direction = direction

    def apply(self, grid: Grid) -> Grid:
        if self.direction == 'right':
            mirrored = np.fliplr(grid.data)
            new_data = np.hstack([grid.data, mirrored])
        elif self.direction == 'left':
            mirrored = np.fliplr(grid.data)
            new_data = np.hstack([mirrored, grid.data])
        elif self.direction == 'down':
            mirrored = np.flipud(grid.data)
            new_data = np.vstack([grid.data, mirrored])
        elif self.direction == 'up':
            mirrored = np.flipud(grid.data)
            new_data = np.vstack([mirrored, grid.data])
        else:
            return grid.copy()

        return Grid(new_data)

    def description_length(self) -> int:
        return 2


class OutlineObjects(Transform):
    """Replace objects with their outlines."""

    def __init__(self, outline_color: int = 1, background: int = 0):
        super().__init__(f"outline_objects", TransformType.OBJECT)
        self.outline_color = outline_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        from scipy.ndimage import binary_dilation

        new_grid = Grid.empty(grid.height, grid.width, self.background)
        objects = grid.extract_objects(background=self.background)

        for obj in objects:
            # Create mask for this object
            mask = np.zeros(grid.shape, dtype=bool)
            for r, c in obj.cells:
                mask[r, c] = True

            # Dilate and subtract to get outline
            dilated = binary_dilation(mask)
            outline = dilated & ~mask

            # Draw outline
            outline_coords = np.where(outline)
            for r, c in zip(*outline_coords):
                new_grid.set(r, c, self.outline_color)

        return new_grid

    def description_length(self) -> int:
        return 2


class ConnectObjects(Transform):
    """Draw lines connecting object centers."""

    def __init__(self, line_color: int = 1, background: int = 0):
        super().__init__("connect_objects", TransformType.SPATIAL)
        self.line_color = line_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        new_grid = grid.copy()
        objects = grid.extract_objects(background=self.background)

        if len(objects) < 2:
            return new_grid

        # Get centers
        centers = []
        for obj in objects:
            rows, cols = zip(*obj.cells)
            center_r, center_c = int(np.mean(rows)), int(np.mean(cols))
            centers.append((center_r, center_c))

        # Connect consecutive centers
        for i in range(len(centers) - 1):
            r1, c1 = centers[i]
            r2, c2 = centers[i + 1]
            self._draw_line(new_grid, r1, c1, r2, c2, self.line_color)

        return new_grid

    def _draw_line(self, grid: Grid, r1: int, c1: int, r2: int, c2: int, color: int):
        """Bresenham's line algorithm."""
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r_step = 1 if r1 < r2 else -1
        c_step = 1 if c1 < c2 else -1

        if dr > dc:
            err = dr / 2
            c = c1
            for r in range(r1, r2 + r_step, r_step):
                if 0 <= r < grid.height and 0 <= c < grid.width:
                    grid.set(r, c, color)
                err -= dc
                if err < 0:
                    c += c_step
                    err += dr
        else:
            err = dc / 2
            r = r1
            for c in range(c1, c2 + c_step, c_step):
                if 0 <= r < grid.height and 0 <= c < grid.width:
                    grid.set(r, c, color)
                err -= dr
                if err < 0:
                    r += r_step
                    err += dc

    def description_length(self) -> int:
        return 3


class ZoomCenter(Transform):
    """Zoom into the center of the grid."""

    def __init__(self, zoom_factor: float = 2.0):
        super().__init__(f"zoom_center_{zoom_factor}", TransformType.GEOMETRIC)
        self.zoom_factor = zoom_factor

    def apply(self, grid: Grid) -> Grid:
        center_r, center_c = grid.height // 2, grid.width // 2

        # Calculate the window to extract
        window_h = int(grid.height / self.zoom_factor)
        window_w = int(grid.width / self.zoom_factor)

        start_r = max(0, center_r - window_h // 2)
        start_c = max(0, center_c - window_w // 2)
        end_r = min(grid.height, start_r + window_h)
        end_c = min(grid.width, start_c + window_w)

        # Extract and scale up
        cropped = grid.data[start_r:end_r, start_c:end_c]

        # Simple nearest-neighbor scaling
        scaled = np.repeat(np.repeat(cropped, int(self.zoom_factor), axis=0),
                          int(self.zoom_factor), axis=1)

        # Crop to original size
        return Grid(scaled[:grid.height, :grid.width])

    def description_length(self) -> int:
        return 2


class ColorGradient(Transform):
    """Create a color gradient based on position."""

    def __init__(self, direction: str = 'horizontal'):
        super().__init__(f"gradient_{direction}", TransformType.COLOR)
        self.direction = direction

    def apply(self, grid: Grid) -> Grid:
        new_grid = grid.copy()

        for r in range(grid.height):
            for c in range(grid.width):
                color = grid.get(r, c)
                if color != 0:
                    if self.direction == 'horizontal':
                        offset = c
                    elif self.direction == 'vertical':
                        offset = r
                    elif self.direction == 'diagonal':
                        offset = r + c
                    else:
                        offset = 0

                    new_color = (color + offset) % 10
                    if new_color == 0:
                        new_color = 1
                    new_grid.set(r, c, new_color)

        return new_grid

    def description_length(self) -> int:
        return 2


class MostCommonColor(Transform):
    """Replace all colors with the most common non-background color."""

    def __init__(self, background: int = 0):
        super().__init__("most_common_color", TransformType.COLOR)
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        counts = grid.count_colors()

        # Remove background from counts
        if self.background in counts:
            del counts[self.background]

        if not counts:
            return grid.copy()

        # Find most common
        most_common = max(counts.items(), key=lambda x: x[1])[0]

        new_grid = grid.copy()
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.get(r, c) != self.background:
                    new_grid.set(r, c, most_common)

        return new_grid

    def description_length(self) -> int:
        return 2


class ReflectBorder(Transform):
    """Add reflected border around the grid."""

    def __init__(self, border_width: int = 1):
        super().__init__(f"reflect_border_{border_width}", TransformType.GEOMETRIC)
        self.border_width = border_width

    def apply(self, grid: Grid) -> Grid:
        # Create larger grid
        new_height = grid.height + 2 * self.border_width
        new_width = grid.width + 2 * self.border_width
        new_grid = Grid.empty(new_height, new_width, 0)

        # Copy original to center
        for r in range(grid.height):
            for c in range(grid.width):
                new_grid.set(r + self.border_width, c + self.border_width,
                           grid.get(r, c))

        # Add reflected borders
        # Top and bottom
        for i in range(self.border_width):
            for c in range(grid.width):
                # Top border
                new_grid.set(i, c + self.border_width,
                           grid.get(self.border_width - i - 1, c))
                # Bottom border
                new_grid.set(new_height - i - 1, c + self.border_width,
                           grid.get(grid.height - self.border_width + i, c))

        # Left and right
        for r in range(new_height):
            for i in range(self.border_width):
                # Left border
                c_src = min(self.border_width - i - 1, grid.width - 1)
                r_src = max(0, min(r - self.border_width, grid.height - 1))
                if 0 <= r_src < grid.height:
                    new_grid.set(r, i, grid.get(r_src, c_src))

                # Right border
                c_src = min(grid.width - self.border_width + i, grid.width - 1)
                r_src = max(0, min(r - self.border_width, grid.height - 1))
                if 0 <= r_src < grid.height:
                    new_grid.set(r, new_width - i - 1, grid.get(r_src, c_src))

        return new_grid

    def description_length(self) -> int:
        return 2


# Registry of custom transforms
CUSTOM_TRANSFORMS = [
    TilePattern(2),
    TilePattern(3),
    DiagonalFlip(),
    SpiralRotate(),
    FillEnclosed(),
    Mirror('right'),
    Mirror('down'),
    OutlineObjects(),
    ConnectObjects(),
    ZoomCenter(2.0),
    ColorGradient('horizontal'),
    ColorGradient('vertical'),
    MostCommonColor(),
    ReflectBorder(1),
]


def get_all_custom_transforms() -> List[Transform]:
    """Get all custom transformations."""
    return CUSTOM_TRANSFORMS.copy()
