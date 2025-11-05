"""
Custom Semantic Priors - Domain-Specific Constraints

These priors create puzzles with the "aha moment" quality of ARC:
- Non-obvious patterns that require insight
- Multiple layers of reasoning
- Elegant solutions that aren't immediately apparent
"""

import numpy as np
from semantic_priors import SemanticPrior, SemanticPriorType
from grid import Grid
from typing import Dict


class PatternRepetition(SemanticPrior):
    """
    Output should have a repeating pattern (tiling, mirroring).
    Creates "aha" moment: "Oh, it's tiling the input!"
    """

    def __init__(self):
        super().__init__("pattern_repetition", SemanticPriorType.PATTERN_CONTINUATION)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # Check if output contains repeated versions of input-like patterns
        if output_grid.height < input_grid.height or output_grid.width < input_grid.width:
            return 0.0

        # Check for 2x2 tiling
        if (output_grid.height == 2 * input_grid.height and
            output_grid.width == 2 * input_grid.width):
            # Check if it's a proper tiling
            matches = 0
            total = 4

            for i in range(2):
                for j in range(2):
                    start_r = i * input_grid.height
                    start_c = j * input_grid.width
                    match = True

                    for r in range(input_grid.height):
                        for c in range(input_grid.width):
                            if output_grid.get(start_r + r, start_c + c) != input_grid.get(r, c):
                                match = False
                                break
                        if not match:
                            break

                    if match:
                        matches += 1

            return matches / total

        return 0.5

    def generate_constraint(self) -> Dict:
        return {
            'type': 'pattern_repetition',
            'requirement': 'tile_or_mirror'
        }


class DiagonalSymmetry(SemanticPrior):
    """
    Diagonal symmetry (main or anti-diagonal).
    Creates insight moment: recognizing diagonal patterns.
    """

    def __init__(self):
        super().__init__("diagonal_symmetry", SemanticPriorType.SYMMETRY)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        if output_grid.height != output_grid.width:
            return 0.0

        # Check main diagonal symmetry
        main_symmetric = True
        anti_symmetric = True

        for r in range(output_grid.height):
            for c in range(output_grid.width):
                # Main diagonal
                if output_grid.get(r, c) != output_grid.get(c, r):
                    main_symmetric = False

                # Anti-diagonal
                anti_r = output_grid.height - 1 - c
                anti_c = output_grid.width - 1 - r
                if 0 <= anti_r < output_grid.height and 0 <= anti_c < output_grid.width:
                    if output_grid.get(r, c) != output_grid.get(anti_r, anti_c):
                        anti_symmetric = False

        if main_symmetric or anti_symmetric:
            return 1.0

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'diagonal_symmetry',
            'requirement': 'main_or_anti'
        }


class CenterFocused(SemanticPrior):
    """
    Transformation focuses on center region.
    Creates insight: "The rule applies differently based on distance from center."
    """

    def __init__(self):
        super().__init__("center_focused", SemanticPriorType.SPATIAL_RELATIONSHIP)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        if input_grid.shape != output_grid.shape:
            return 0.0

        center_r, center_c = input_grid.height // 2, input_grid.width // 2

        # Check if center region changed more/differently than edges
        center_changes = 0
        edge_changes = 0
        center_count = 0
        edge_count = 0

        for r in range(input_grid.height):
            for c in range(input_grid.width):
                dist = max(abs(r - center_r), abs(c - center_c))

                if input_grid.get(r, c) != output_grid.get(r, c):
                    if dist <= min(input_grid.height, input_grid.width) // 4:
                        center_changes += 1
                        center_count += 1
                    else:
                        edge_changes += 1
                        edge_count += 1
                else:
                    if dist <= min(input_grid.height, input_grid.width) // 4:
                        center_count += 1
                    else:
                        edge_count += 1

        # Center should change more than edges (or vice versa)
        if center_count > 0 and edge_count > 0:
            center_ratio = center_changes / center_count
            edge_ratio = edge_changes / edge_count

            # Strong difference indicates center-focused transformation
            if abs(center_ratio - edge_ratio) > 0.3:
                return 1.0

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'center_focused',
            'requirement': 'differential_transformation'
        }


class ObjectRelationship(SemanticPrior):
    """
    Objects have a relationship (e.g., smallest becomes red, largest becomes blue).
    Creates "aha": "It's sorting/categorizing objects by property!"
    """

    def __init__(self):
        super().__init__("object_relationship", SemanticPriorType.GROUPING)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_objs = input_grid.extract_objects(background=0)
        output_objs = output_grid.extract_objects(background=0)

        if len(input_objs) < 2 or len(output_objs) < 2:
            return 0.5

        # Check if objects are sorted by size in output
        input_sizes = [obj.size for obj in input_objs]
        output_sizes = [obj.size for obj in output_objs]

        # Check if output colors correlate with sizes
        if len(output_objs) == len(input_objs):
            # Sort by position and check color-size correlation
            input_sorted = sorted(zip(input_sizes, input_objs),
                                key=lambda x: x[0])
            output_sorted = sorted(zip(output_sizes, output_objs),
                                 key=lambda x: x[0])

            # Check if color increases/decreases with size
            output_colors = [obj.color for _, obj in output_sorted]

            if len(set(output_colors)) > 1:
                # Colors vary - check for pattern
                monotonic_increase = all(output_colors[i] <= output_colors[i+1]
                                       for i in range(len(output_colors)-1))
                monotonic_decrease = all(output_colors[i] >= output_colors[i+1]
                                       for i in range(len(output_colors)-1))

                if monotonic_increase or monotonic_decrease:
                    return 1.0

        return 0.3

    def generate_constraint(self) -> Dict:
        return {
            'type': 'object_relationship',
            'requirement': 'property_based_transformation'
        }


class BorderSpecial(SemanticPrior):
    """
    Border cells treated differently than interior.
    Creates insight: "The rule only applies to the border/interior!"
    """

    def __init__(self):
        super().__init__("border_special", SemanticPriorType.SPATIAL_RELATIONSHIP)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        if input_grid.shape != output_grid.shape:
            return 0.3

        h, w = input_grid.height, input_grid.width

        if h < 3 or w < 3:
            return 0.0

        # Count changes in border vs interior
        border_changes = 0
        interior_changes = 0
        border_count = 0
        interior_count = 0

        for r in range(h):
            for c in range(w):
                is_border = (r == 0 or r == h-1 or c == 0 or c == w-1)

                if input_grid.get(r, c) != output_grid.get(r, c):
                    if is_border:
                        border_changes += 1
                    else:
                        interior_changes += 1

                if is_border:
                    border_count += 1
                else:
                    interior_count += 1

        # Calculate change ratios
        if border_count > 0 and interior_count > 0:
            border_ratio = border_changes / border_count
            interior_ratio = interior_changes / interior_count

            # Strong difference indicates border-specific rule
            diff = abs(border_ratio - interior_ratio)
            return min(1.0, diff * 2)

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'border_special',
            'requirement': 'differential_by_position'
        }


class EnclosureRule(SemanticPrior):
    """
    Enclosed regions filled or transformed.
    Creates "aha": "It fills/transforms enclosed spaces!"
    """

    def __init__(self):
        super().__init__("enclosure_rule", SemanticPriorType.CONTAINMENT)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        if input_grid.shape != output_grid.shape:
            return 0.3

        # Check if previously empty (0) enclosed regions are now filled
        # This is complex to detect perfectly, so we use a heuristic
        input_zeros = np.sum(input_grid.data == 0)
        output_zeros = np.sum(output_grid.data == 0)

        if input_zeros > 0:
            # Check if some zeros were filled
            filled_ratio = (input_zeros - output_zeros) / input_zeros

            if 0.1 < filled_ratio < 0.9:  # Partial filling suggests selective rule
                return 1.0
            elif filled_ratio > 0:
                return 0.6

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'enclosure_rule',
            'requirement': 'fill_or_transform_enclosed'
        }


class ConnectivityChange(SemanticPrior):
    """
    Objects connect or separate.
    Creates insight: "Objects that touch merge/separate!"
    """

    def __init__(self):
        super().__init__("connectivity_change", SemanticPriorType.OBJECT_TRACKING)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_objs = input_grid.extract_objects(background=0)
        output_objs = output_grid.extract_objects(background=0)

        if len(input_objs) == 0:
            return 0.5

        # Check if number of objects changed (merging or splitting)
        obj_count_change = abs(len(output_objs) - len(input_objs))

        if obj_count_change > 0:
            # Significant change in object count suggests connectivity rule
            ratio = obj_count_change / max(len(input_objs), 1)
            return min(1.0, ratio)

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'connectivity_change',
            'requirement': 'merge_or_split'
        }


class ProportionalScaling(SemanticPrior):
    """
    Output size relates to input properties.
    Creates "aha": "The output size depends on [some property]!"
    """

    def __init__(self):
        super().__init__("proportional_scaling", SemanticPriorType.PATTERN_CONTINUATION)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        input_area = input_grid.height * input_grid.width
        output_area = output_grid.height * output_grid.width

        if input_area == 0:
            return 0.0

        ratio = output_area / input_area

        # Check for common scaling factors (2x, 3x, 4x, 0.5x, etc.)
        common_ratios = [0.25, 0.5, 2.0, 3.0, 4.0, 9.0]

        for common_ratio in common_ratios:
            if abs(ratio - common_ratio) < 0.1:
                return 1.0

        # Any size change is somewhat interesting
        if ratio != 1.0:
            return 0.5

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'proportional_scaling',
            'requirement': 'size_transformation'
        }


class ColorPropagation(SemanticPrior):
    """
    Colors "spread" or "propagate" based on proximity.
    Creates insight: "Colors spread from certain points!"
    """

    def __init__(self):
        super().__init__("color_propagation", SemanticPriorType.SPATIAL_RELATIONSHIP)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        if input_grid.shape != output_grid.shape:
            return 0.3

        # Count cells that gained color (were 0, now non-0)
        gained_color = 0
        total_changed = 0

        for r in range(input_grid.height):
            for c in range(input_grid.width):
                inp = input_grid.get(r, c)
                out = output_grid.get(r, c)

                if inp != out:
                    total_changed += 1
                    if inp == 0 and out != 0:
                        gained_color += 1

        if total_changed > 0:
            propagation_ratio = gained_color / total_changed

            # High ratio suggests color propagation
            if propagation_ratio > 0.5:
                return min(1.0, propagation_ratio)

        return 0.0

    def generate_constraint(self) -> Dict:
        return {
            'type': 'color_propagation',
            'requirement': 'spreading'
        }


# Registry of custom priors emphasizing "aha moments"
CUSTOM_PRIORS = [
    PatternRepetition(),
    DiagonalSymmetry(),
    CenterFocused(),
    ObjectRelationship(),
    BorderSpecial(),
    EnclosureRule(),
    ConnectivityChange(),
    ProportionalScaling(),
    ColorPropagation(),
]


def get_all_custom_priors():
    """Get all custom semantic priors."""
    return CUSTOM_PRIORS.copy()


def get_aha_moment_priors():
    """
    Get a curated set of priors that create the strongest "aha moments".
    These create puzzles where the solution requires insight, not brute force.
    """
    return [
        PatternRepetition(),      # "It's tiling!"
        ObjectRelationship(),     # "It sorts by size!"
        BorderSpecial(),          # "Only the border changes!"
        EnclosureRule(),         # "It fills enclosed regions!"
        DiagonalSymmetry(),      # "It's diagonal symmetry!"
    ]
