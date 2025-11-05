"""
Custom Cellular Automata Rules

These CA rules create interesting emergent patterns that require
insight to understand. Each rule has been designed to create
non-obvious but elegant transformations.
"""

from cellular_automata import CAConfig, CARule
from typing import Dict, List


def get_custom_ca_rules() -> Dict[str, CAConfig]:
    """
    Get custom CA rules that create interesting "aha moment" patterns.

    These rules are designed to:
    - Create non-trivial patterns from simple rules
    - Require insight to deduce from input/output
    - Have bounded complexity (not too chaotic)
    """

    rules = {}

    # Rule 1: "Majority" - Cell takes color of majority of neighbors
    # Creates smooth, blob-like patterns
    rules["majority"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=5,
        survive_conditions=[4, 5, 6, 7, 8],  # Survive with many neighbors
        birth_conditions=[5, 6, 7, 8],        # Birth with strong majority
        neighborhood_type="moore"
    )

    # Rule 2: "Diamond" - Creates diamond-shaped propagation patterns
    # Uses Von Neumann neighborhood for cross-shaped spreading
    rules["diamond"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=6,
        survive_conditions=[2, 3],
        birth_conditions=[2],
        neighborhood_type="von_neumann"
    )

    # Rule 3: "Crystal Growth" - Stable structures with controlled growth
    # Creates snowflake-like patterns
    rules["crystal"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=8,
        survive_conditions=[2, 3, 4],
        birth_conditions=[2, 3],
        neighborhood_type="moore"
    )

    # Rule 4: "Waves" - Oscillating patterns
    # Similar to Day & Night rule
    rules["waves"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=6,
        survive_conditions=[3, 4, 6, 7, 8],
        birth_conditions=[3, 6, 7, 8],
        neighborhood_type="moore"
    )

    # Rule 5: "Erosion" - Structures erode from edges
    # High survival requirements create shrinking patterns
    rules["erosion"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=5,
        survive_conditions=[5, 6, 7, 8],
        birth_conditions=[],  # No birth
        neighborhood_type="moore"
    )

    # Rule 6: "Expansion" - Structures grow aggressively
    # Easy birth, hard death
    rules["expansion"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=4,
        survive_conditions=[1, 2, 3, 4, 5, 6, 7, 8],
        birth_conditions=[1, 2, 3],
        neighborhood_type="moore"
    )

    # Rule 7: "Maze Builder" - Creates maze-like structures
    # Classic maze generation rule
    rules["maze_builder"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=7,
        survive_conditions=[1, 2, 3, 4, 5],
        birth_conditions=[3],
        neighborhood_type="moore"
    )

    # Rule 8: "Checkerboard" - Creates alternating patterns
    rules["checkerboard"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=5,
        survive_conditions=[2, 4, 6, 8],  # Even number of neighbors
        birth_conditions=[2, 4],
        neighborhood_type="moore"
    )

    # Rule 9: "Coral Reef" - Organic branching structures
    rules["coral_reef"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=10,
        survive_conditions=[4, 5, 6, 7, 8],
        birth_conditions=[3],
        neighborhood_type="moore"
    )

    # Rule 10: "Vote" - Democratic rule (majority wins)
    rules["vote"] = CAConfig(
        rule=CARule.CUSTOM,
        steps=6,
        survive_conditions=[4, 5, 6, 7, 8],
        birth_conditions=[5, 6, 7, 8],
        neighborhood_type="moore"
    )

    return rules


def get_rule_description(rule_name: str) -> str:
    """Get human-readable description of what each rule does."""
    descriptions = {
        "majority": "Cells adopt the state of the majority of their neighbors, creating smooth blobs",
        "diamond": "Cross-shaped propagation creates diamond patterns",
        "crystal": "Controlled growth creates snowflake-like crystal structures",
        "waves": "Oscillating patterns that create wave-like effects",
        "erosion": "Structures shrink from the edges, eroding inward",
        "expansion": "Aggressive growth causes structures to expand rapidly",
        "maze_builder": "Creates maze-like labyrinthine structures",
        "checkerboard": "Alternating patterns emerge based on even neighbor counts",
        "coral_reef": "Organic branching like coral growth",
        "vote": "Democratic rule where majority state wins"
    }
    return descriptions.get(rule_name, "Custom CA rule")


def get_aha_moment_rules() -> List[str]:
    """
    Get rule names that create the strongest "aha moments".

    These rules create patterns that are:
    - Non-obvious from the rule description
    - Elegant and understandable once seen
    - Require insight to deduce
    """
    return [
        "maze_builder",   # Creates complex mazes from simple rules
        "crystal",        # Beautiful snowflake patterns
        "diamond",        # Distinctive cross-shaped spreading
        "coral_reef",     # Organic-looking growth
        "waves",          # Interesting oscillations
    ]


class MultiColorCA:
    """
    Multi-color cellular automata that create more complex puzzles.

    Instead of binary alive/dead, uses multiple colors with interaction rules.
    This creates richer "aha moments" where color interactions must be deduced.
    """

    @staticmethod
    def color_competition_rule(grid, r, c):
        """
        Color competition: dominant neighboring color wins.
        "Aha": Different colors compete for territory!
        """
        from collections import Counter

        current = grid.get(r, c)
        neighbors = []

        # Get all neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.height and 0 <= nc < grid.width:
                    neighbor_color = grid.get(nr, nc)
                    if neighbor_color != 0:  # Ignore background
                        neighbors.append(neighbor_color)

        if not neighbors:
            return current

        # Find most common neighbor color
        counter = Counter(neighbors)
        most_common = counter.most_common(1)[0][0]
        most_common_count = counter.most_common(1)[0][1]

        # If majority is strong enough, convert
        if most_common_count >= 4:
            return most_common

        return current

    @staticmethod
    def color_cycle_rule(grid, r, c):
        """
        Colors cycle: 1→2→3→1 based on neighbors.
        "Aha": Colors follow a cycle pattern!
        """
        current = grid.get(r, c)

        if current == 0:
            return 0

        # Count neighbors of next color in cycle
        next_color = (current % 3) + 1 if current <= 3 else current

        next_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.height and 0 <= nc < grid.width:
                    if grid.get(nr, nc) == next_color:
                        next_count += 1

        # If enough neighbors of next color, cycle forward
        if next_count >= 3:
            return next_color

        return current

    @staticmethod
    def color_merge_rule(grid, r, c):
        """
        Colors merge: 1+2→3, 2+3→4, etc.
        "Aha": Colors combine like mixing paint!
        """
        current = grid.get(r, c)

        if current == 0:
            # Check if surrounded by colors that merge
            neighbors = set()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.height and 0 <= nc < grid.width:
                        n_color = grid.get(nr, nc)
                        if n_color != 0:
                            neighbors.add(n_color)

            # If two different colors adjacent, create sum
            if len(neighbors) >= 2:
                colors = sorted(list(neighbors))[:2]
                merged = min(colors[0] + colors[1], 9)
                return merged

        return current

    @staticmethod
    def color_dominance_rule(grid, r, c):
        """
        Higher numbers dominate lower numbers.
        "Aha": Larger values consume smaller values!
        """
        current = grid.get(r, c)

        if current == 0:
            return 0

        # Find maximum neighbor
        max_neighbor = current
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.height and 0 <= nc < grid.width:
                    neighbor = grid.get(nr, nc)
                    if neighbor > max_neighbor:
                        max_neighbor = neighbor

        # If dominated by higher value, may convert
        if max_neighbor > current:
            # Probabilistic conversion based on difference
            diff = max_neighbor - current
            if diff >= 2:  # Strong dominance
                return max_neighbor

        return current


def get_multicolor_rules() -> Dict[str, callable]:
    """Get multi-color CA rules for richer puzzles."""
    return {
        "color_competition": MultiColorCA.color_competition_rule,
        "color_cycle": MultiColorCA.color_cycle_rule,
        "color_merge": MultiColorCA.color_merge_rule,
        "color_dominance": MultiColorCA.color_dominance_rule,
    }


def get_multicolor_rule_descriptions() -> Dict[str, str]:
    """Get descriptions of multi-color rules."""
    return {
        "color_competition": "Dominant neighboring color wins - colors compete for territory",
        "color_cycle": "Colors cycle through a sequence based on neighbors",
        "color_merge": "Different colors merge to create new colors, like mixing paint",
        "color_dominance": "Higher numbered colors dominate and consume lower numbers",
    }
