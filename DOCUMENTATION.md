# Re-ARC Documentation

## Overview

Re-ARC is a generalized puzzle generator that creates genuinely complex and interesting puzzles for AGI benchmarking. Unlike approaches that hand-craft puzzles or use simple random composition, Re-ARC implements a principled multi-layer architecture based on:

1. **Computational Hardness**: Embedding NP-hard problems to prevent brute-force solutions
2. **Emergence-Based Generation**: Using cellular automata with semantic filtering
3. **Hierarchical Composition**: Constraint-guided transformation synthesis
4. **Difficulty Calibration**: Program synthesis evaluation for the "Goldilocks zone"
5. **Meta-Learning**: Adversarial filtering to avoid pattern-matching solutions

## Architecture

The system follows this pipeline:

```
Semantic Goals → Constraint System → Transformation Search → Grid Generation
    → Apply Transform → Solvability Test → Filter → Valid Puzzle
```

### Core Components

#### 1. Grid (`grid.py`)

**Purpose**: Fundamental data structures for representing 2D colored grids and puzzles.

**Key Classes**:
- `Grid`: 2D array of colors (0-9) with operations like rotate, flip, extract_objects
- `Object`: Connected component of cells with same color
- `Puzzle`: Collection of input/output pairs (train + test)

**Example**:
```python
from grid import Grid
import numpy as np

# Create a grid
data = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
grid = Grid(data)

# Extract objects
objects = grid.extract_objects(background=0)
print(f"Found {len(objects)} objects")

# Transform
rotated = grid.rotate(1)
flipped = grid.flip('horizontal')
```

#### 2. DSL (`dsl.py`)

**Purpose**: Domain-specific language for composable transformations with MDL calculation.

**Transform Categories**:
- **Geometric**: Rotate, Flip, Transpose, Scale
- **Color**: ColorMap, SwapColors, InvertColors, ReplaceColor
- **Object**: MoveObjects, ColorBySize, FilterObjectsBySize
- **Spatial**: Gravity, Extend
- **Logical**: ConditionalTransform, Compose, Branch
- **Cellular**: CellularAutomaton, GameOfLife

**Key Concept - Description Length**:
Each transformation has a Minimum Description Length (MDL) that measures its complexity. Good puzzles have small MDL but large search space.

**Example**:
```python
from dsl import Rotate, Flip, Compose

# Single transform
transform = Rotate(1)
mdl = transform.description_length()  # 2

# Composition
composite = Compose([Rotate(1), Flip('horizontal')])
mdl = composite.description_length()  # 5 (2 + 2 + 1 for composition)

# Apply
output = composite.apply(input_grid)
```

#### 3. Semantic Priors (`semantic_priors.py`)

**Purpose**: Human-interpretable concepts that make puzzles logically coherent.

**Prior Categories**:
- **Symmetry**: PreserveSymmetry, CreateSymmetry
- **Object Tracking**: PreserveObjectCount, PreserveObjectShapes, MaintainObjectCohesion
- **Color Consistency**: PreserveColorPalette, ConsistentColorMapping
- **Spatial Relationships**: PreserveRelativePositions, AlignObjects
- **Pattern Continuation**: ExtendPattern
- **Containment**: PartitionSpace
- **Counting**: CountBasedTransformation

**Evaluation**: Each prior scores a transformation from 0.0 (violates) to 1.0 (perfectly respects).

**Example**:
```python
from semantic_priors import PreserveObjectCount, PreserveSymmetry

prior = PreserveObjectCount()
score = prior.evaluate(input_grid, output_grid)
# Returns 1.0 if object count preserved, 0.0 otherwise
```

#### 4. Constraints (`constraints.py`)

**Purpose**: Convert semantic priors into constraints and find satisfying transformations.

**Key Classes**:
- `ConstraintSystem`: Manages hard (must satisfy) and soft (prefer) constraints
- `ConstraintSynthesizer`: Converts priors to constraints
- `TransformationFinder`: Searches for transforms satisfying constraints
- `ComplexityAnalyzer`: Evaluates MDL and search space size
- `HardnessEmbedding`: Embeds NP-hard problems (graph coloring, SAT, etc.)

**Search Algorithms**:
- Random search over primitives
- Beam search for compositions
- Constraint-guided pruning

**Example**:
```python
from constraints import ConstraintSynthesizer, TransformationFinder
from semantic_priors import PreserveObjectCount, PreserveColorPalette

priors = [PreserveObjectCount(), PreserveColorPalette()]
constraint_system = ConstraintSynthesizer.from_priors(priors)

finder = TransformationFinder(constraint_system)
transform = finder.find_composite_transform(
    input_grid,
    max_depth=3,
    beam_width=5
)
```

#### 5. Cellular Automata (`cellular_automata.py`)

**Purpose**: Generate puzzles through emergent complexity from simple rules.

**Features**:
- Game of Life, Seeds, Brian's Brain, Wireworld
- Custom rules with configurable survive/birth conditions
- Semantic filtering for "interesting" patterns
- History tracking for dynamics analysis

**Filtering Criteria**:
- Object cohesion (connected components)
- Bounded complexity (not too simple/chaotic)
- Interesting dynamics (neither static nor exploding)
- Stable/periodic endpoints

**Example**:
```python
from cellular_automata import InterestingRules, CellularAutomaton

config = InterestingRules.get_rule("life")
config.steps = 10

ca = CellularAutomaton(config)
final = ca.run(initial_grid, steps=10)
```

#### 6. Verification (`verification.py`)

**Purpose**: Implement "Goldilocks zone" filtering - reject puzzles that are too easy or too hard.

**Key Classes**:
- `ProgramSynthesisSolver`: Exhaustive search baseline solver
- `PuzzleVerifier`: Determines puzzle difficulty
- `AdversarialFilter`: Rejects pattern-matching solutions
- `DiversityMetrics`: Measures dataset diversity
- `QualityMetrics`: Consistency and elegance scoring

**Difficulty Levels**:
- `TOO_EASY`: Solved instantly (< 100ms)
- `EASY`: Solved quickly (100-1500ms)
- `MEDIUM`: Moderate difficulty (1500-3500ms)
- `HARD`: Challenging (3500-5000ms)
- `TOO_HARD`: Unsolvable in timeout (> 5000ms)

**Goldilocks Criteria**:
```
Good puzzle: small MDL + large search space
- MDL: 2-6 (elegant solution)
- Search time: 100-5000ms (hard but solvable)
```

**Example**:
```python
from verification import PuzzleVerifier

verifier = PuzzleVerifier(
    min_search_time_ms=100,
    max_search_time_ms=5000,
    min_mdl=2,
    max_mdl=6
)

result = verifier.verify(puzzle)
if result.is_valid:
    print(f"Difficulty: {result.difficulty}")
    print(f"Solution: {result.solution}")
```

#### 7. Puzzle Generator (`puzzle_generator.py`)

**Purpose**: Main orchestrator implementing the full pipeline.

**Generation Strategies**:
- `CONSTRAINT_BASED`: Sample priors → constraints → find transform
- `CELLULAR_AUTOMATA`: Generate patterns through CA evolution
- `HYBRID`: Mix of both strategies
- `HARDNESS_EMBEDDING`: Embed NP-hard problems

**Configuration**:
```python
from puzzle_generator import GeneratorConfig, GenerationStrategy

config = GeneratorConfig(
    strategy=GenerationStrategy.HYBRID,
    num_train_examples=3,        # Training pairs
    num_test_examples=1,         # Test inputs
    min_grid_size=5,
    max_grid_size=15,
    max_transform_depth=3,       # Max composition depth
    beam_width=5,                # Search beam width
    min_mdl=2,                   # Minimum description length
    max_mdl=6,                   # Maximum description length
    use_adversarial_filtering=True
)
```

## Usage Patterns

### Quick Start

```python
from puzzle_generator import generate_puzzle_dataset, GenerationStrategy

# Generate 10 puzzles
dataset = generate_puzzle_dataset(
    num_puzzles=10,
    strategy=GenerationStrategy.HYBRID,
    output_path="puzzles.json"
)

# Access puzzles
for puzzle, verification in dataset.puzzles:
    print(f"Difficulty: {verification.difficulty}")
    print(f"Training examples: {puzzle.num_train}")
```

### Custom Prior Configuration

```python
from puzzle_generator import PuzzleGenerator, GeneratorConfig
from semantic_priors import (
    PreserveObjectCount, PreserveSymmetry, ConsistentColorMapping
)

# Define custom priors
priors = [
    PreserveObjectCount(),
    PreserveSymmetry('horizontal'),
    ConsistentColorMapping()
]

# Generate with these priors
generator = PuzzleGenerator(GeneratorConfig())
puzzles = generator.generate(num_puzzles=5, priors=priors)
```

### Programmatic Puzzle Creation

```python
from grid import Grid, Puzzle
from dsl import Rotate, Compose, Flip

# Define transformation
transform = Compose([Rotate(1), Flip('horizontal')])

# Generate training examples
train_pairs = []
for _ in range(3):
    input_grid = Grid.random(8, 8, num_colors=5)
    output_grid = transform.apply(input_grid)
    train_pairs.append((input_grid, output_grid))

# Create puzzle
puzzle = Puzzle(
    train_pairs=train_pairs,
    test_inputs=[Grid.random(8, 8, num_colors=5)],
    metadata={'transform': str(transform)}
)
```

## Design Principles

### 1. Computational Hardness

Puzzles embed NP-hard subproblems to prevent brute-force solutions:
- Graph coloring constraints
- SAT encoding
- Hamiltonian path requirements

### 2. Semantic Coherence

Human priors ensure solutions "make sense":
- Objects maintain cohesion
- Colors transform consistently
- Spatial relationships are meaningful
- Symmetries are preserved or created purposefully

### 3. Conceptual Compression

Good puzzles have high compression ratio:
```
Compression = Execution Complexity / Description Complexity
```

Small program (low MDL) that produces complex change → high compression → interesting puzzle.

### 4. Adversarial Filtering

Meta-learning approach:
1. Generate puzzle
2. If solver finds it too easily → reject
3. Learn which patterns to avoid
4. Iterate

This fights against pattern-matching approaches that transformers use.

### 5. Goldilocks Zone

```
Too Easy: MDL < 2 OR search_time < 100ms → REJECT
Just Right: 2 ≤ MDL ≤ 6 AND 100ms ≤ search_time ≤ 5000ms → ACCEPT
Too Hard: MDL > 6 OR search_time > 5000ms → REJECT
```

## Extending the System

### Adding New Transformations

```python
from dsl import Transform, TransformType

class MyTransform(Transform):
    def __init__(self, param: int):
        super().__init__(f"my_transform_{param}", TransformType.GEOMETRIC)
        self.param = param

    def apply(self, grid: Grid) -> Grid:
        # Implement transformation logic
        new_data = ...
        return Grid(new_data)

    def description_length(self) -> int:
        return 2  # Operation + parameter
```

### Adding New Semantic Priors

```python
from semantic_priors import SemanticPrior, SemanticPriorType

class MyPrior(SemanticPrior):
    def __init__(self):
        super().__init__("my_prior", SemanticPriorType.CUSTOM)

    def evaluate(self, input_grid: Grid, output_grid: Grid) -> float:
        # Return score 0.0-1.0
        return score

    def generate_constraint(self) -> Dict:
        return {'type': 'my_constraint', 'params': {...}}
```

### Adding New CA Rules

```python
from cellular_automata import CAConfig, CARule

config = CAConfig(
    rule=CARule.CUSTOM,
    steps=10,
    survive_conditions=[2, 3],
    birth_conditions=[3]
)
```

## Performance Considerations

### Search Space Size

For depth `d` and `n` primitives:
```
Search space = n^1 + n^2 + ... + n^d = O(n^d)
```

With 9 primitives and depth 3:
- Depth 1: 9 possibilities
- Depth 2: 81 possibilities
- Depth 3: 729 possibilities
- Total: 819 candidates

### Optimization Strategies

1. **Beam Search**: Keep only top-k candidates at each level
2. **Early Pruning**: Reject candidates violating hard constraints
3. **Caching**: Store evaluated transforms
4. **Parallel Generation**: Generate multiple puzzles concurrently

## Theoretical Foundations

### Why This Creates AGI Benchmarks

1. **Systematic Coverage**: Enumerate all combinations of semantic priors
2. **Provable Hardness**: Embedded NP-hard problems resist brute force
3. **Human-Interpretable**: Semantic priors ensure solutions make sense
4. **Scalable Difficulty**: Parameterize by constraint complexity and depth
5. **No Overfitting**: Generator creates infinite variations

### The Counter-Intuitive Insight

> "Generalization happens at the wrong level if you just compose transformations. True generalization is not about having one function that generates all puzzles, but about having a meta-framework where puzzle families emerge from principled combinations of hardness + semantics + verification."

### Measuring Puzzle Quality

```python
from verification import QualityMetrics

# Consistency (do all examples follow same rule?)
consistency = QualityMetrics.consistency_score(puzzle)

# Elegance (simple description, complex execution?)
elegance = QualityMetrics.elegance_score(puzzle, solution)

# Combined quality
quality = 0.5 * consistency + 0.5 * elegance
```

## References

The system implements concepts from:
- ARC-AGI benchmark design
- Program synthesis and DSL research
- Cellular automata theory (Conway's Life, etc.)
- Constraint satisfaction problems
- Computational complexity theory
- Meta-learning and adversarial training
