"""
Domain-Specific Language (DSL) for puzzle transformations.

This module implements a composable DSL for describing transformations
on grids, supporting:
- Primitive operations
- Composition
- Conditional application
- Object-based transformations
- Minimum Description Length (MDL) calculation
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Set, Tuple, Dict
from grid import Grid, Object
import numpy as np
from enum import Enum


class TransformType(Enum):
    """Categories of transformations for analysis."""
    GEOMETRIC = "geometric"
    COLOR = "color"
    OBJECT = "object"
    SPATIAL = "spatial"
    LOGICAL = "logical"
    CELLULAR = "cellular"


class Transform(ABC):
    """Base class for all transformations in the DSL."""

    def __init__(self, name: str, transform_type: TransformType):
        self.name = name
        self.transform_type = transform_type

    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """Apply the transformation to a grid."""
        pass

    @abstractmethod
    def description_length(self) -> int:
        """
        Return the Minimum Description Length (MDL) of this transformation.
        Used for measuring complexity.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# ============================================================================
# GEOMETRIC TRANSFORMATIONS
# ============================================================================

class Rotate(Transform):
    """Rotate grid by k*90 degrees."""

    def __init__(self, k: int = 1):
        super().__init__(f"rotate_{k*90}", TransformType.GEOMETRIC)
        self.k = k

    def apply(self, grid: Grid) -> Grid:
        return grid.rotate(self.k)

    def description_length(self) -> int:
        return 2  # Operation + parameter


class Flip(Transform):
    """Flip grid along an axis."""

    def __init__(self, axis: str = 'horizontal'):
        super().__init__(f"flip_{axis}", TransformType.GEOMETRIC)
        self.axis = axis

    def apply(self, grid: Grid) -> Grid:
        return grid.flip(self.axis)

    def description_length(self) -> int:
        return 2


class Transpose(Transform):
    """Transpose the grid (swap rows and columns)."""

    def __init__(self):
        super().__init__("transpose", TransformType.GEOMETRIC)

    def apply(self, grid: Grid) -> Grid:
        return Grid(grid.data.T)

    def description_length(self) -> int:
        return 1


class Scale(Transform):
    """Scale grid by a factor."""

    def __init__(self, factor: int):
        super().__init__(f"scale_{factor}", TransformType.GEOMETRIC)
        self.factor = factor

    def apply(self, grid: Grid) -> Grid:
        if self.factor == 1:
            return grid.copy()
        new_data = np.repeat(np.repeat(grid.data, self.factor, axis=0),
                            self.factor, axis=1)
        return Grid(new_data)

    def description_length(self) -> int:
        return 2


# ============================================================================
# COLOR TRANSFORMATIONS
# ============================================================================

class ColorMap(Transform):
    """Map colors according to a dictionary."""

    def __init__(self, mapping: Dict[int, int]):
        super().__init__("color_map", TransformType.COLOR)
        self.mapping = mapping

    def apply(self, grid: Grid) -> Grid:
        new_data = grid.data.copy()
        for old_color, new_color in self.mapping.items():
            new_data[grid.data == old_color] = new_color
        return Grid(new_data)

    def description_length(self) -> int:
        return 1 + len(self.mapping)


class SwapColors(Transform):
    """Swap two colors."""

    def __init__(self, color1: int, color2: int):
        super().__init__(f"swap_{color1}_{color2}", TransformType.COLOR)
        self.color1 = color1
        self.color2 = color2

    def apply(self, grid: Grid) -> Grid:
        new_data = grid.data.copy()
        mask1 = grid.data == self.color1
        mask2 = grid.data == self.color2
        new_data[mask1] = self.color2
        new_data[mask2] = self.color1
        return Grid(new_data)

    def description_length(self) -> int:
        return 3


class InvertColors(Transform):
    """Invert colors (color -> 9 - color)."""

    def __init__(self):
        super().__init__("invert_colors", TransformType.COLOR)

    def apply(self, grid: Grid) -> Grid:
        return Grid(9 - grid.data)

    def description_length(self) -> int:
        return 1


class ReplaceColor(Transform):
    """Replace all instances of one color with another."""

    def __init__(self, old_color: int, new_color: int):
        super().__init__(f"replace_{old_color}_with_{new_color}", TransformType.COLOR)
        self.old_color = old_color
        self.new_color = new_color

    def apply(self, grid: Grid) -> Grid:
        new_data = grid.data.copy()
        new_data[grid.data == self.old_color] = self.new_color
        return Grid(new_data)

    def description_length(self) -> int:
        return 3


# ============================================================================
# OBJECT-BASED TRANSFORMATIONS
# ============================================================================

class MoveObjects(Transform):
    """Move all objects by a delta."""

    def __init__(self, delta_row: int, delta_col: int, background: int = 0):
        super().__init__(f"move_objects_{delta_row}_{delta_col}", TransformType.OBJECT)
        self.delta_row = delta_row
        self.delta_col = delta_col
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        objects = grid.extract_objects(background=self.background)
        new_grid = Grid.empty(grid.height, grid.width, self.background)

        for obj in objects:
            for r, c in obj.cells:
                new_r = r + self.delta_row
                new_c = c + self.delta_col
                if 0 <= new_r < grid.height and 0 <= new_c < grid.width:
                    new_grid.set(new_r, new_c, obj.color)

        return new_grid

    def description_length(self) -> int:
        return 3


class ColorBySize(Transform):
    """Color objects based on their size."""

    def __init__(self, size_to_color: Dict[int, int], background: int = 0):
        super().__init__("color_by_size", TransformType.OBJECT)
        self.size_to_color = size_to_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        objects = grid.extract_objects(background=self.background)
        new_grid = Grid.empty(grid.height, grid.width, self.background)

        for obj in objects:
            size = obj.size
            new_color = self.size_to_color.get(size, obj.color)
            for r, c in obj.cells:
                new_grid.set(r, c, new_color)

        return new_grid

    def description_length(self) -> int:
        return 2 + len(self.size_to_color)


class FilterObjectsBySize(Transform):
    """Keep only objects within size range."""

    def __init__(self, min_size: int, max_size: int, background: int = 0):
        super().__init__(f"filter_size_{min_size}_{max_size}", TransformType.OBJECT)
        self.min_size = min_size
        self.max_size = max_size
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        objects = grid.extract_objects(background=self.background)
        new_grid = Grid.empty(grid.height, grid.width, self.background)

        for obj in objects:
            if self.min_size <= obj.size <= self.max_size:
                for r, c in obj.cells:
                    new_grid.set(r, c, obj.color)

        return new_grid

    def description_length(self) -> int:
        return 3


# ============================================================================
# SPATIAL TRANSFORMATIONS
# ============================================================================

class Gravity(Transform):
    """Apply gravity - objects fall to bottom."""

    def __init__(self, direction: str = 'down', background: int = 0):
        super().__init__(f"gravity_{direction}", TransformType.SPATIAL)
        self.direction = direction
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        new_grid = Grid.empty(grid.height, grid.width, self.background)

        if self.direction == 'down':
            for col in range(grid.width):
                write_row = grid.height - 1
                for row in range(grid.height - 1, -1, -1):
                    if grid.get(row, col) != self.background:
                        new_grid.set(write_row, col, grid.get(row, col))
                        write_row -= 1
        elif self.direction == 'up':
            for col in range(grid.width):
                write_row = 0
                for row in range(grid.height):
                    if grid.get(row, col) != self.background:
                        new_grid.set(write_row, col, grid.get(row, col))
                        write_row += 1

        return new_grid

    def description_length(self) -> int:
        return 2


class Extend(Transform):
    """Extend objects in a direction until hitting another object."""

    def __init__(self, direction: str = 'right', background: int = 0):
        super().__init__(f"extend_{direction}", TransformType.SPATIAL)
        self.direction = direction
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        new_grid = grid.copy()

        directions = {
            'right': (0, 1),
            'left': (0, -1),
            'down': (1, 0),
            'up': (-1, 0)
        }
        dr, dc = directions.get(self.direction, (0, 1))

        for r in range(grid.height):
            for c in range(grid.width):
                if grid.get(r, c) != self.background:
                    color = grid.get(r, c)
                    curr_r, curr_c = r + dr, c + dc
                    while (0 <= curr_r < grid.height and
                           0 <= curr_c < grid.width and
                           new_grid.get(curr_r, curr_c) == self.background):
                        new_grid.set(curr_r, curr_c, color)
                        curr_r += dr
                        curr_c += dc

        return new_grid

    def description_length(self) -> int:
        return 2


# ============================================================================
# LOGICAL TRANSFORMATIONS
# ============================================================================

class ConditionalTransform(Transform):
    """Apply transformation only if condition is met."""

    def __init__(self, condition: Callable[[Grid], bool],
                 transform: Transform, name: str = "conditional"):
        super().__init__(name, TransformType.LOGICAL)
        self.condition = condition
        self.transform = transform

    def apply(self, grid: Grid) -> Grid:
        if self.condition(grid):
            return self.transform.apply(grid)
        return grid.copy()

    def description_length(self) -> int:
        return 2 + self.transform.description_length()


class Compose(Transform):
    """Compose multiple transformations sequentially."""

    def __init__(self, transforms: List[Transform], name: str = "compose"):
        super().__init__(name, TransformType.LOGICAL)
        self.transforms = transforms

    def apply(self, grid: Grid) -> Grid:
        result = grid
        for transform in self.transforms:
            result = transform.apply(result)
        return result

    def description_length(self) -> int:
        return sum(t.description_length() for t in self.transforms) + 1


class Branch(Transform):
    """Apply different transforms based on a condition."""

    def __init__(self, condition: Callable[[Grid], bool],
                 true_transform: Transform,
                 false_transform: Transform):
        super().__init__("branch", TransformType.LOGICAL)
        self.condition = condition
        self.true_transform = true_transform
        self.false_transform = false_transform

    def apply(self, grid: Grid) -> Grid:
        if self.condition(grid):
            return self.true_transform.apply(grid)
        return self.false_transform.apply(grid)

    def description_length(self) -> int:
        return (2 + self.true_transform.description_length() +
                self.false_transform.description_length())


# ============================================================================
# CELLULAR AUTOMATA TRANSFORMATIONS
# ============================================================================

class CellularAutomaton(Transform):
    """Apply a cellular automaton rule for N steps."""

    def __init__(self, rule: Callable[[Grid, int, int], int],
                 steps: int = 1, name: str = "ca"):
        super().__init__(name, TransformType.CELLULAR)
        self.rule = rule
        self.steps = steps

    def apply(self, grid: Grid) -> Grid:
        result = grid.copy()
        for _ in range(self.steps):
            new_grid = Grid.empty(grid.height, grid.width)
            for r in range(grid.height):
                for c in range(grid.width):
                    new_grid.set(r, c, self.rule(result, r, c))
            result = new_grid
        return result

    def description_length(self) -> int:
        return 2 + self.steps  # Rule + steps


class GameOfLife(Transform):
    """Conway's Game of Life."""

    def __init__(self, steps: int = 1, alive_color: int = 1, dead_color: int = 0):
        super().__init__("game_of_life", TransformType.CELLULAR)
        self.steps = steps
        self.alive_color = alive_color
        self.dead_color = dead_color

    def apply(self, grid: Grid) -> Grid:
        def rule(g: Grid, r: int, c: int) -> int:
            # Count alive neighbors
            alive = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < g.height and 0 <= nc < g.width:
                        if g.get(nr, nc) == self.alive_color:
                            alive += 1

            current = g.get(r, c)
            if current == self.alive_color:
                return self.alive_color if alive in [2, 3] else self.dead_color
            else:
                return self.alive_color if alive == 3 else self.dead_color

        result = grid.copy()
        for _ in range(self.steps):
            new_grid = Grid.empty(grid.height, grid.width, self.dead_color)
            for r in range(grid.height):
                for c in range(grid.width):
                    new_grid.set(r, c, rule(result, r, c))
            result = new_grid
        return result

    def description_length(self) -> int:
        return 2


# ============================================================================
# DSL UTILITIES
# ============================================================================

class DSL:
    """Factory and utilities for working with the DSL."""

    @staticmethod
    def parse(program_str: str) -> Transform:
        """Parse a simple string representation into a Transform."""
        # Simple parser for basic operations
        # Format: "op(args)" or "op1(args) | op2(args)" for composition

        if '|' in program_str:
            # Composition
            parts = [p.strip() for p in program_str.split('|')]
            transforms = [DSL.parse(p) for p in parts]
            return Compose(transforms)

        # Parse single operation
        if '(' in program_str:
            op, args = program_str.split('(', 1)
            args = args.rstrip(')').split(',')
            args = [a.strip() for a in args if a.strip()]
        else:
            op = program_str
            args = []

        # Create transform based on operation
        if op == 'rotate':
            k = int(args[0]) if args else 1
            return Rotate(k)
        elif op == 'flip':
            axis = args[0] if args else 'horizontal'
            return Flip(axis)
        elif op == 'transpose':
            return Transpose()
        elif op == 'invert_colors':
            return InvertColors()
        elif op == 'gravity':
            direction = args[0] if args else 'down'
            return Gravity(direction)
        else:
            raise ValueError(f"Unknown operation: {op}")

    @staticmethod
    def primitives() -> List[Transform]:
        """Return list of primitive transformations."""
        return [
            Rotate(1), Rotate(2), Rotate(3),
            Flip('horizontal'), Flip('vertical'),
            Transpose(),
            InvertColors(),
            Gravity('down'), Gravity('up'),
        ]

    @staticmethod
    def calculate_search_space_size(max_depth: int,
                                    num_primitives: int = 9) -> int:
        """
        Calculate the size of the search space for finding transformations.
        This is exponential in depth.
        """
        return num_primitives ** max_depth
