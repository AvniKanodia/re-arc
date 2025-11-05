"""
re-arc: A Generalized Puzzle Generator for AGI Benchmarks

This package implements a principled approach to generating ARC-style puzzles
that are:
- Algorithmically complex (embed computational hardness)
- Semantically coherent (respect human priors)
- In the difficulty "sweet spot" (not too easy, not too hard)

Main components:
- grid: Grid and Puzzle data structures
- dsl: Domain-specific language for transformations
- semantic_priors: Human-interpretable constraints
- constraints: Constraint satisfaction system
- cellular_automata: CA-based puzzle generation
- verification: Difficulty assessment and filtering
- puzzle_generator: Main orchestrator

Quick start:
    from puzzle_generator import generate_puzzle_dataset, GenerationStrategy

    # Generate a dataset of puzzles
    dataset = generate_puzzle_dataset(
        num_puzzles=10,
        strategy=GenerationStrategy.HYBRID
    )
"""

__version__ = "0.1.0"

from grid import Grid, Puzzle
from puzzle_generator import (
    PuzzleGenerator,
    GeneratorConfig,
    GenerationStrategy,
    generate_puzzle_dataset
)

__all__ = [
    'Grid',
    'Puzzle',
    'PuzzleGenerator',
    'GeneratorConfig',
    'GenerationStrategy',
    'generate_puzzle_dataset'
]
