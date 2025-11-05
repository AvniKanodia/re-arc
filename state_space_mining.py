"""
Graph State Space Mining for Puzzle Generation

Key Insight: Hard puzzles have transformations that are compositionally complex
but visually seem simple. We mine the state space graph to find these automatically.

Approach:
1. Model grid space as a graph (nodes = grids, edges = operations)
2. Define primitive operations (rotate, flip, shift, color ops)
3. Search for transformation paths that maximize "interestingness"
4. Interestingness = high visual difference / path length / branching factor
5. Generate puzzles from discovered interesting paths
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
from grid import Grid, Puzzle
import copy


# ==============================================================================
# PRIMITIVE OPERATIONS
# ==============================================================================

class PrimitiveOp:
    """A single atomic operation on a grid."""

    @staticmethod
    def rotate_90_cw(grid: Grid) -> Grid:
        """Rotate 90 degrees clockwise."""
        return Grid(np.rot90(grid.data, k=-1))

    @staticmethod
    def rotate_90_ccw(grid: Grid) -> Grid:
        """Rotate 90 degrees counter-clockwise."""
        return Grid(np.rot90(grid.data, k=1))

    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        """Rotate 180 degrees."""
        return Grid(np.rot90(grid.data, k=2))

    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        """Flip horizontally (left-right)."""
        return Grid(np.fliplr(grid.data))

    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        """Flip vertically (top-bottom)."""
        return Grid(np.flipud(grid.data))

    @staticmethod
    def flip_diagonal(grid: Grid) -> Grid:
        """Flip along main diagonal."""
        return Grid(np.transpose(grid.data))

    @staticmethod
    def flip_antidiagonal(grid: Grid) -> Grid:
        """Flip along anti-diagonal."""
        return Grid(np.fliplr(np.transpose(grid.data)))

    @staticmethod
    def shift_up(grid: Grid) -> Grid:
        """Shift content up, wrap around."""
        return Grid(np.roll(grid.data, shift=-1, axis=0))

    @staticmethod
    def shift_down(grid: Grid) -> Grid:
        """Shift content down, wrap around."""
        return Grid(np.roll(grid.data, shift=1, axis=0))

    @staticmethod
    def shift_left(grid: Grid) -> Grid:
        """Shift content left, wrap around."""
        return Grid(np.roll(grid.data, shift=-1, axis=1))

    @staticmethod
    def shift_right(grid: Grid) -> Grid:
        """Shift content right, wrap around."""
        return Grid(np.roll(grid.data, shift=1, axis=1))

    @staticmethod
    def invert_colors(grid: Grid, max_color: int = 8) -> Grid:
        """Invert all colors: c -> max_color - c."""
        return Grid(max_color - grid.data)

    @staticmethod
    def increment_colors(grid: Grid, max_color: int = 8) -> Grid:
        """Increment all non-zero colors, wrap at max_color."""
        result = grid.data.copy()
        mask = result > 0
        result[mask] = (result[mask] % max_color) + 1
        return Grid(result)

    @staticmethod
    def swap_colors(grid: Grid, color1: int, color2: int) -> Grid:
        """Swap two specific colors."""
        result = grid.data.copy()
        mask1 = result == color1
        mask2 = result == color2
        result[mask1] = color2
        result[mask2] = color1
        return Grid(result)

    @staticmethod
    def fill_border(grid: Grid, color: int = 1) -> Grid:
        """Fill the border with a color."""
        result = grid.data.copy()
        result[0, :] = color
        result[-1, :] = color
        result[:, 0] = color
        result[:, -1] = color
        return Grid(result)

    @staticmethod
    def hollow_fill(grid: Grid) -> Grid:
        """Fill hollow regions in objects."""
        result = grid.data.copy()
        h, w = result.shape

        # Find background-connected regions from edges
        from scipy import ndimage
        background = (result == 0)
        # Start from edges
        edge_background = np.zeros_like(background)
        edge_background[0, :] = background[0, :]
        edge_background[-1, :] = background[-1, :]
        edge_background[:, 0] = background[:, 0]
        edge_background[:, -1] = background[:, -1]

        # Dilate from edges to find all external background
        structure = np.ones((3, 3))
        external_bg = ndimage.binary_dilation(edge_background, structure=structure, iterations=max(h, w))
        external_bg = external_bg & background

        # Internal background = background - external background
        internal_bg = background & ~external_bg

        # Fill internal background with adjacent color
        if np.any(internal_bg):
            # Find a non-background color to fill with
            non_bg_colors = result[result > 0]
            if len(non_bg_colors) > 0:
                fill_color = np.bincount(non_bg_colors).argmax()
                result[internal_bg] = fill_color

        return Grid(result)

    @staticmethod
    def extract_largest_object(grid: Grid, background: int = 0) -> Grid:
        """Keep only the largest connected component."""
        objects = grid.extract_objects(background=background)
        if not objects:
            return grid

        largest = max(objects, key=lambda obj: len(obj.cells))
        result = np.full_like(grid.data, background)

        for cell in largest.cells:
            result[cell] = grid.data[cell]

        return Grid(result)

    @staticmethod
    def zoom_in_2x(grid: Grid) -> Grid:
        """Zoom in 2x (each cell becomes 2x2)."""
        h, w = grid.data.shape
        result = np.zeros((h * 2, w * 2), dtype=np.int8)
        for i in range(h):
            for j in range(w):
                result[i*2:i*2+2, j*2:j*2+2] = grid.data[i, j]
        return Grid(result)

    @staticmethod
    def zoom_out_2x(grid: Grid) -> Grid:
        """Zoom out 2x (every 2x2 becomes 1 cell, using majority vote)."""
        h, w = grid.data.shape
        if h % 2 != 0 or w % 2 != 0:
            return grid  # Can't zoom out non-even dimensions

        result = np.zeros((h // 2, w // 2), dtype=np.int8)
        for i in range(h // 2):
            for j in range(w // 2):
                block = grid.data[i*2:i*2+2, j*2:j*2+2]
                # Majority vote
                values, counts = np.unique(block, return_counts=True)
                result[i, j] = values[np.argmax(counts)]

        return Grid(result)


# ==============================================================================
# VISUAL DISTANCE METRICS
# ==============================================================================

class VisualDistance:
    """Measures how visually different two grids are."""

    @staticmethod
    def structural_similarity(grid1: Grid, grid2: Grid) -> float:
        """
        Measure structural similarity (0 = identical, 1 = completely different).

        Considers:
        - Color distribution
        - Spatial structure
        - Object count and sizes
        """
        if grid1.data.shape != grid2.data.shape:
            return 1.0  # Different shapes = maximally different

        # Color distribution distance (histogram)
        hist1 = np.bincount(grid1.data.flatten(), minlength=9)
        hist2 = np.bincount(grid2.data.flatten(), minlength=9)
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        color_dist = np.sum(np.abs(hist1 - hist2)) / 2  # Normalize to [0, 1]

        # Pixel-wise difference
        pixel_diff = np.mean(grid1.data != grid2.data)

        # Object count difference
        objs1 = len(grid1.extract_objects())
        objs2 = len(grid2.extract_objects())
        obj_count_diff = abs(objs1 - objs2) / max(objs1, objs2, 1)

        # Combine metrics
        distance = (
            0.4 * pixel_diff +
            0.3 * color_dist +
            0.3 * obj_count_diff
        )

        return min(distance, 1.0)

    @staticmethod
    def perceptual_distance(grid1: Grid, grid2: Grid) -> float:
        """
        How different do the grids LOOK to a human?

        High-level features:
        - Symmetry changes
        - Pattern regularity
        - Visual complexity
        """
        if grid1.data.shape != grid2.data.shape:
            return 1.0

        # Symmetry score
        def symmetry_score(g: Grid) -> float:
            h_sym = np.array_equal(g.data, np.fliplr(g.data))
            v_sym = np.array_equal(g.data, np.flipud(g.data))
            d_sym = np.array_equal(g.data, g.data.T) if g.data.shape[0] == g.data.shape[1] else False
            return sum([h_sym, v_sym, d_sym]) / 3

        sym1 = symmetry_score(grid1)
        sym2 = symmetry_score(grid2)
        symmetry_change = abs(sym1 - sym2)

        # Visual entropy (complexity)
        def visual_entropy(g: Grid) -> float:
            hist = np.bincount(g.data.flatten(), minlength=9)
            hist = hist[hist > 0] / hist.sum()
            return -np.sum(hist * np.log2(hist + 1e-10))

        entropy1 = visual_entropy(grid1)
        entropy2 = visual_entropy(grid2)
        entropy_change = abs(entropy1 - entropy2) / 3.17  # Normalize by max entropy (log2(9))

        # Combine
        distance = (
            0.5 * VisualDistance.structural_similarity(grid1, grid2) +
            0.3 * symmetry_change +
            0.2 * entropy_change
        )

        return min(distance, 1.0)


# ==============================================================================
# STATE SPACE GRAPH
# ==============================================================================

@dataclass
class GridNode:
    """A node in the state space graph."""
    grid: Grid
    hash_key: str

    def __hash__(self):
        return hash(self.hash_key)

    def __eq__(self, other):
        return self.hash_key == other.hash_key


@dataclass
class TransformationPath:
    """A sequence of operations transforming one grid to another."""
    states: List[Grid]
    operations: List[Tuple[str, Callable]]
    interestingness: float = 0.0

    def __len__(self):
        return len(self.operations)


class GridStateSpace:
    """
    Models the space of possible grids as a graph.

    Nodes = grid states
    Edges = primitive operations

    Mines the graph for interesting transformation paths.
    """

    def __init__(self, max_colors: int = 8):
        self.max_colors = max_colors

        # All available primitive operations
        self.operations = [
            ("rotate_90_cw", PrimitiveOp.rotate_90_cw),
            ("rotate_90_ccw", PrimitiveOp.rotate_90_ccw),
            ("rotate_180", PrimitiveOp.rotate_180),
            ("flip_horizontal", PrimitiveOp.flip_horizontal),
            ("flip_vertical", PrimitiveOp.flip_vertical),
            ("flip_diagonal", PrimitiveOp.flip_diagonal),
            ("flip_antidiagonal", PrimitiveOp.flip_antidiagonal),
            ("shift_up", PrimitiveOp.shift_up),
            ("shift_down", PrimitiveOp.shift_down),
            ("shift_left", PrimitiveOp.shift_left),
            ("shift_right", PrimitiveOp.shift_right),
            ("invert_colors", PrimitiveOp.invert_colors),
            ("increment_colors", PrimitiveOp.increment_colors),
            ("fill_border", lambda g: PrimitiveOp.fill_border(g, color=1)),
            ("hollow_fill", PrimitiveOp.hollow_fill),
            ("extract_largest", PrimitiveOp.extract_largest_object),
            # Add zoom operations for even-sized grids
        ]

        self.visited_states: Set[str] = set()

    def grid_hash(self, grid: Grid) -> str:
        """Create a unique hash for a grid state."""
        return grid.data.tobytes().hex()

    def get_neighbors(self, grid: Grid) -> List[Tuple[str, Callable, Grid]]:
        """Get all states reachable by applying one operation."""
        neighbors = []

        for op_name, op_func in self.operations:
            try:
                next_grid = op_func(grid)

                # Skip if same as input (operation had no effect)
                if self.grid_hash(next_grid) == self.grid_hash(grid):
                    continue

                neighbors.append((op_name, op_func, next_grid))
            except Exception:
                # Some operations may fail on certain grids
                continue

        return neighbors

    def calculate_branching_factor(self, grid: Grid) -> float:
        """
        Calculate the branching factor (how many "reasonable" next moves).

        Lower branching = fewer obvious moves = harder puzzle
        """
        neighbors = self.get_neighbors(grid)

        # Count distinct visual outcomes (some ops may produce same result)
        unique_outcomes = set()
        for _, _, next_grid in neighbors:
            unique_outcomes.add(self.grid_hash(next_grid))

        return len(unique_outcomes)

    def interestingness_score(self, path: TransformationPath) -> float:
        """
        Score how interesting a transformation path is.

        High score = good puzzle:
        - High visual difference (output looks very different from input)
        - Moderate path length (not too short/trivial, not too long/arbitrary)
        - Low branching factor (few obvious moves at each step)
        """
        if len(path) == 0:
            return 0.0

        start_grid = path.states[0]
        end_grid = path.states[-1]

        # Visual distance (higher = more interesting)
        visual_diff = VisualDistance.perceptual_distance(start_grid, end_grid)

        # Path length penalty (prefer 2-5 operations)
        path_len = len(path)
        ideal_length = 3
        length_penalty = np.exp(-((path_len - ideal_length) ** 2) / 4)

        # Branching factor (lower average branching = harder)
        avg_branching = np.mean([
            self.calculate_branching_factor(state)
            for state in path.states[:-1]
        ])
        # Normalize: typical branching is 10-20, want low values to score high
        branching_score = 1.0 / (1.0 + avg_branching / 10.0)

        # Combine scores
        score = (
            visual_diff ** 1.5 *  # Visual difference is most important
            length_penalty *       # Moderate length is good
            branching_score ** 0.5 # Low branching is bonus
        )

        return score

    def find_interesting_path(
        self,
        start_grid: Grid,
        max_depth: int = 5,
        beam_width: int = 10,
        target_score: float = 0.5
    ) -> Optional[TransformationPath]:
        """
        Find a transformation path that maximizes interestingness.

        Uses beam search to explore promising paths.
        """
        # Initial path
        initial_path = TransformationPath(
            states=[start_grid],
            operations=[],
            interestingness=0.0
        )

        # Beam search
        beam = [initial_path]
        best_path = initial_path
        best_score = 0.0

        self.visited_states = {self.grid_hash(start_grid)}

        for depth in range(max_depth):
            # Expand all paths in beam
            candidates = []

            for path in beam:
                current_grid = path.states[-1]

                # Get all possible next states
                neighbors = self.get_neighbors(current_grid)

                for op_name, op_func, next_grid in neighbors:
                    # Skip if already visited
                    next_hash = self.grid_hash(next_grid)
                    if next_hash in self.visited_states:
                        continue

                    # Create new path
                    new_path = TransformationPath(
                        states=path.states + [next_grid],
                        operations=path.operations + [(op_name, op_func)],
                    )

                    # Score it
                    score = self.interestingness_score(new_path)
                    new_path.interestingness = score

                    candidates.append(new_path)
                    self.visited_states.add(next_hash)

                    # Track best
                    if score > best_score:
                        best_score = score
                        best_path = new_path

                        # Early termination if we found something good
                        if score >= target_score:
                            return best_path

            if not candidates:
                break

            # Keep top beam_width paths
            candidates.sort(key=lambda p: p.interestingness, reverse=True)
            beam = candidates[:beam_width]

        return best_path if best_score > 0.1 else None

    def generate_puzzle(
        self,
        start_grid: Optional[Grid] = None,
        grid_size: Tuple[int, int] = (5, 5),
        num_examples: int = 3
    ) -> Optional[Puzzle]:
        """
        Generate a puzzle by finding an interesting transformation path.

        Creates training examples by showing the transformation on different inputs.
        """
        # Generate random start state if not provided
        if start_grid is None:
            start_grid = self._generate_random_grid(grid_size)

        # Find interesting transformation
        path = self.find_interesting_path(start_grid)

        if path is None or len(path) == 0:
            return None

        # Extract the transformation (sequence of operations)
        transformation = path.operations

        # Generate training examples with variations
        train_pairs = []

        # Example 1: Original transformation
        train_pairs.append((path.states[0], path.states[-1]))

        # Generate variations
        for _ in range(num_examples - 1):
            # Create a variation of the start grid
            varied_start = self._create_variation(start_grid)

            # Apply same transformation sequence
            current = varied_start
            for op_name, op_func in transformation:
                try:
                    current = op_func(current)
                except:
                    break

            # Add if transformation succeeded
            if self.grid_hash(current) != self.grid_hash(varied_start):
                train_pairs.append((varied_start, current))

        # Test input: Another variation
        test_input = self._create_variation(start_grid)

        # Create puzzle
        puzzle = Puzzle(
            train_pairs=train_pairs,
            test_inputs=[test_input],
            metadata={
                'transformation': [op_name for op_name, _ in transformation],
                'path_length': len(path),
                'interestingness': path.interestingness,
                'visual_distance': VisualDistance.perceptual_distance(
                    path.states[0], path.states[-1]
                )
            }
        )

        return puzzle

    def _generate_random_grid(self, size: Tuple[int, int]) -> Grid:
        """Generate a random grid with interesting structure."""
        h, w = size
        data = np.zeros((h, w), dtype=np.int8)

        # Generate random patterns
        pattern_type = np.random.choice(['objects', 'lines', 'regions'])

        if pattern_type == 'objects':
            # Place 2-4 small objects
            num_objects = np.random.randint(2, 5)
            for _ in range(num_objects):
                color = np.random.randint(1, self.max_colors + 1)
                size = np.random.randint(1, 3)
                y = np.random.randint(0, h - size + 1)
                x = np.random.randint(0, w - size + 1)
                data[y:y+size, x:x+size] = color

        elif pattern_type == 'lines':
            # Draw 1-3 lines
            num_lines = np.random.randint(1, 4)
            for _ in range(num_lines):
                color = np.random.randint(1, self.max_colors + 1)
                if np.random.random() < 0.5:
                    # Horizontal line
                    y = np.random.randint(0, h)
                    data[y, :] = color
                else:
                    # Vertical line
                    x = np.random.randint(0, w)
                    data[:, x] = color

        else:  # regions
            # Divide into regions
            mid_h = h // 2
            mid_w = w // 2
            data[:mid_h, :mid_w] = np.random.randint(0, self.max_colors + 1)
            data[mid_h:, :mid_w] = np.random.randint(0, self.max_colors + 1)
            data[:mid_h, mid_w:] = np.random.randint(0, self.max_colors + 1)
            data[mid_h:, mid_w:] = np.random.randint(0, self.max_colors + 1)

        return Grid(data)

    def _create_variation(self, grid: Grid) -> Grid:
        """Create a variation of a grid (different colors, slight modifications)."""
        data = grid.data.copy()

        # Color remapping
        color_map = np.arange(9)
        np.random.shuffle(color_map[1:])  # Keep background (0) as 0

        result = np.zeros_like(data)
        for i in range(9):
            result[data == i] = color_map[i]

        return Grid(result)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Demo the state space mining approach."""
    print("=" * 70)
    print("GRAPH STATE SPACE MINING FOR PUZZLE GENERATION")
    print("=" * 70)
    print()

    space = GridStateSpace(max_colors=8)

    print("Generating 5 puzzles using state space mining...")
    print()

    puzzles = []
    for i in range(5):
        print(f"Mining puzzle {i+1}/5...")

        # Try different grid sizes
        size = np.random.choice([(4, 4), (5, 5), (6, 6), (4, 6)])

        puzzle = space.generate_puzzle(grid_size=size, num_examples=3)

        if puzzle:
            puzzles.append(puzzle)

            trans = puzzle.metadata['transformation']
            score = puzzle.metadata['interestingness']
            visual_dist = puzzle.metadata['visual_distance']

            print(f"  ✓ Found interesting path!")
            print(f"    Transformation: {' → '.join(trans)}")
            print(f"    Interestingness: {score:.3f}")
            print(f"    Visual Distance: {visual_dist:.3f}")
            print()

    print(f"Generated {len(puzzles)} puzzles successfully!")
    print()

    return puzzles


if __name__ == "__main__":
    puzzles = main()
