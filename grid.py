"""
Grid representation for ARC-style puzzles.

This module provides the fundamental data structure for representing
2D grid-based puzzles with colored cells.
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Object:
    """Represents a connected component of cells with the same color."""
    color: int
    cells: Set[Tuple[int, int]]

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Returns (min_row, min_col, max_row, max_col)."""
        if not self.cells:
            return (0, 0, 0, 0)
        rows, cols = zip(*self.cells)
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def size(self) -> int:
        """Number of cells in the object."""
        return len(self.cells)

    def __hash__(self):
        return hash((self.color, frozenset(self.cells)))


class Grid:
    """2D grid with colored cells (0-9, where 0 is typically background)."""

    def __init__(self, data: np.ndarray):
        """
        Initialize a grid.

        Args:
            data: 2D numpy array of integers (0-9)
        """
        self.data = np.array(data, dtype=np.int8)
        if len(self.data.shape) != 2:
            raise ValueError("Grid must be 2D")
        if self.data.min() < 0 or self.data.max() > 9:
            raise ValueError("Grid values must be in range 0-9")

    @classmethod
    def empty(cls, height: int, width: int, fill: int = 0) -> 'Grid':
        """Create an empty grid filled with a color."""
        return cls(np.full((height, width), fill, dtype=np.int8))

    @classmethod
    def random(cls, height: int, width: int, num_colors: int = 10) -> 'Grid':
        """Create a random grid."""
        return cls(np.random.randint(0, num_colors, (height, width)))

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def copy(self) -> 'Grid':
        """Create a deep copy of this grid."""
        return Grid(self.data.copy())

    def get(self, row: int, col: int) -> int:
        """Get the color at position (row, col)."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return int(self.data[row, col])
        return -1  # Out of bounds

    def set(self, row: int, col: int, color: int):
        """Set the color at position (row, col)."""
        if 0 <= row < self.height and 0 <= col < self.width:
            self.data[row, col] = color

    def extract_objects(self, background: int = 0,
                       connectivity: int = 4) -> List[Object]:
        """
        Extract connected components as objects.

        Args:
            background: Color to treat as background
            connectivity: 4 or 8 for connectivity type

        Returns:
            List of Object instances
        """
        from scipy.ndimage import label

        objects = []
        for color in range(10):
            if color == background:
                continue

            mask = (self.data == color)
            if not mask.any():
                continue

            # Label connected components
            structure = np.ones((3, 3)) if connectivity == 8 else \
                       np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            labeled, num_features = label(mask, structure=structure)

            for i in range(1, num_features + 1):
                cells = set(zip(*np.where(labeled == i)))
                objects.append(Object(color=color, cells=cells))

        return objects

    def count_colors(self) -> Dict[int, int]:
        """Count occurrences of each color."""
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def has_symmetry(self, axis: str = 'horizontal') -> bool:
        """Check if grid has symmetry along an axis."""
        if axis == 'horizontal':
            return np.array_equal(self.data, np.flipud(self.data))
        elif axis == 'vertical':
            return np.array_equal(self.data, np.fliplr(self.data))
        elif axis == 'diagonal':
            if self.height != self.width:
                return False
            return np.array_equal(self.data, self.data.T)
        return False

    def rotate(self, k: int = 1) -> 'Grid':
        """Rotate grid 90 degrees k times clockwise."""
        return Grid(np.rot90(self.data, k=-k))

    def flip(self, axis: str = 'horizontal') -> 'Grid':
        """Flip grid along axis."""
        if axis == 'horizontal':
            return Grid(np.flipud(self.data))
        elif axis == 'vertical':
            return Grid(np.fliplr(self.data))
        return self.copy()

    def crop(self, row: int, col: int, height: int, width: int) -> 'Grid':
        """Extract a rectangular region."""
        return Grid(self.data[row:row+height, col:col+width])

    def to_dict(self) -> dict:
        """Convert to dictionary format (ARC-compatible)."""
        return {
            'height': self.height,
            'width': self.width,
            'grid': self.data.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Grid':
        """Create grid from dictionary format."""
        return cls(np.array(data['grid']))

    def __eq__(self, other: 'Grid') -> bool:
        """Check equality with another grid."""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def __repr__(self) -> str:
        return f"Grid({self.height}x{self.width})"

    def __str__(self) -> str:
        """String representation with colored blocks."""
        # Use Unicode block characters for visualization
        colors = ['⬛', '🟦', '🟥', '🟩', '🟨', '⬜', '🟪', '🟧', '🟫', '⬜']
        lines = []
        for row in self.data:
            line = ''.join(colors[c] for c in row)
            lines.append(line)
        return '\n'.join(lines)


@dataclass
class Puzzle:
    """Represents a puzzle with input/output pairs."""
    train_pairs: List[Tuple[Grid, Grid]]
    test_inputs: List[Grid]
    test_outputs: Optional[List[Grid]] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_train(self) -> int:
        return len(self.train_pairs)

    @property
    def num_test(self) -> int:
        return len(self.test_inputs)

    def to_dict(self) -> dict:
        """Convert to ARC JSON format."""
        return {
            'train': [
                {'input': inp.to_dict()['grid'],
                 'output': out.to_dict()['grid']}
                for inp, out in self.train_pairs
            ],
            'test': [
                {'input': inp.to_dict()['grid'],
                 'output': out.to_dict()['grid'] if self.test_outputs else []}
                for inp, out in zip(
                    self.test_inputs,
                    self.test_outputs or [None] * len(self.test_inputs)
                )
            ],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Puzzle':
        """Create puzzle from ARC JSON format."""
        train_pairs = [
            (Grid(np.array(pair['input'])),
             Grid(np.array(pair['output'])))
            for pair in data['train']
        ]
        test_inputs = [Grid(np.array(pair['input'])) for pair in data['test']]
        test_outputs = None
        if data['test'] and data['test'][0].get('output'):
            test_outputs = [Grid(np.array(pair['output'])) for pair in data['test']]

        return cls(
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            metadata=data.get('metadata', {})
        )

    def __repr__(self) -> str:
        return f"Puzzle(train={self.num_train}, test={self.num_test})"
