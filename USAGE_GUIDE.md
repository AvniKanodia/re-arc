# Re-ARC Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/AvniKanodia/re-arc.git
cd re-arc

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### 1. Generate Your First Puzzle

```python
from puzzle_generator import generate_puzzle_dataset, GenerationStrategy

# Generate a single puzzle
dataset = generate_puzzle_dataset(
    num_puzzles=1,
    strategy=GenerationStrategy.CONSTRAINT_BASED
)

# Access the puzzle
puzzle, verification = dataset[0]
print(f"Difficulty: {verification.difficulty.value}")

# View training example
input_grid, output_grid = puzzle.train_pairs[0]
print("Input:")
print(input_grid)
print("\nOutput:")
print(output_grid)
```

### 2. Generate a Dataset

```python
# Generate 100 puzzles and save to file
dataset = generate_puzzle_dataset(
    num_puzzles=100,
    strategy=GenerationStrategy.HYBRID,
    output_path="my_puzzles.json"
)

print(f"Generated {len(dataset)} puzzles")
print("Statistics:", dataset.get_statistics())
```

### 3. Run Examples

```python
# Run the example file
python examples.py

# Or import specific examples
from examples import (
    example_1_basic_grid_operations,
    example_5_generate_simple_puzzle
)

example_1_basic_grid_operations()
example_5_generate_simple_puzzle()
```

## Common Use Cases

### Generate Puzzles with Specific Difficulty

```python
from puzzle_generator import PuzzleGenerator, GeneratorConfig, GenerationStrategy
from verification import DifficultyLevel

config = GeneratorConfig(
    strategy=GenerationStrategy.CONSTRAINT_BASED,
    min_search_time_ms=1000,   # Harder puzzles
    max_search_time_ms=5000,
    min_mdl=3,
    max_mdl=5
)

generator = PuzzleGenerator(config)
puzzles = generator.generate(num_puzzles=10)

# Filter by difficulty
hard_puzzles = [
    (p, v) for p, v in puzzles
    if v.difficulty == DifficultyLevel.HARD
]
```

### Generate Puzzles with Specific Themes

```python
from semantic_priors import (
    PreserveSymmetry, PreserveObjectCount, ConsistentColorMapping
)

# Symmetry-focused puzzles
symmetry_priors = [
    PreserveSymmetry('horizontal'),
    PreserveSymmetry('vertical'),
    PreserveObjectCount()
]

puzzles = generator.generate(num_puzzles=5, priors=symmetry_priors)
```

### Generate CA-Based Puzzles

```python
from puzzle_generator import GeneratorConfig, GenerationStrategy

config = GeneratorConfig(
    strategy=GenerationStrategy.CELLULAR_AUTOMATA,
    ca_rule="life",      # Game of Life
    ca_steps=10,
    min_grid_size=8,
    max_grid_size=12
)

generator = PuzzleGenerator(config)
ca_puzzles = generator.generate(num_puzzles=5)
```

### Custom Transformation Pipeline

```python
from grid import Grid, Puzzle
from dsl import Rotate, Flip, Compose, ColorMap
import numpy as np

# Define a complex transformation
transform = Compose([
    Rotate(1),                          # Rotate 90° clockwise
    Flip('horizontal'),                 # Flip horizontally
    ColorMap({1: 2, 2: 3, 3: 1})       # Cycle colors
])

# Generate examples
train_pairs = []
for i in range(3):
    # Create input with structure
    input_grid = Grid.empty(6, 6, 0)
    # Add some objects
    for _ in range(3):
        r, c = np.random.randint(0, 4), np.random.randint(0, 4)
        color = np.random.randint(1, 4)
        input_grid.set(r, c, color)
        input_grid.set(r+1, c, color)

    output_grid = transform.apply(input_grid)
    train_pairs.append((input_grid, output_grid))

# Create test input
test_input = Grid.random(6, 6, num_colors=4)

puzzle = Puzzle(
    train_pairs=train_pairs,
    test_inputs=[test_input],
    metadata={
        'custom': True,
        'transform': str(transform),
        'mdl': transform.description_length()
    }
)

# Verify difficulty
from verification import PuzzleVerifier
verifier = PuzzleVerifier()
result = verifier.verify(puzzle)
print(f"Custom puzzle difficulty: {result.difficulty}")
```

### Batch Processing

```python
import multiprocessing as mp
from puzzle_generator import PuzzleGenerator, GeneratorConfig

def generate_batch(batch_size):
    """Generate a batch of puzzles."""
    generator = PuzzleGenerator(GeneratorConfig())
    return generator.generate(num_puzzles=batch_size)

# Generate 100 puzzles using 4 processes
with mp.Pool(4) as pool:
    results = pool.map(generate_batch, [25, 25, 25, 25])

# Combine results
all_puzzles = []
for batch in results:
    all_puzzles.extend(batch)

print(f"Generated {len(all_puzzles)} puzzles in parallel")
```

### Analyze Generated Puzzles

```python
from verification import DiversityMetrics, QualityMetrics

# Generate dataset
dataset = generate_puzzle_dataset(num_puzzles=50)

# Measure diversity
diversity = DiversityMetrics.semantic_diversity([p for p, _ in dataset.puzzles])
print(f"Semantic diversity: {diversity:.2f}")

transform_diversity = DiversityMetrics.transformation_diversity(
    [(p, v.solution) for p, v in dataset.puzzles if v.solution]
)
print(f"Transformation diversity: {transform_diversity:.2f}")

# Analyze quality
for puzzle, verification in dataset.puzzles[:5]:
    consistency = QualityMetrics.consistency_score(puzzle)
    elegance = QualityMetrics.elegance_score(puzzle, verification.solution)
    print(f"Puzzle: consistency={consistency:.2f}, elegance={elegance:.2f}")
```

### Export to ARC Format

```python
import json

# Generate puzzles
dataset = generate_puzzle_dataset(num_puzzles=10)

# Export each puzzle as separate JSON file
for i, (puzzle, verification) in enumerate(dataset.puzzles):
    arc_format = puzzle.to_dict()

    with open(f"puzzle_{i:03d}.json", "w") as f:
        json.dump(arc_format, f, indent=2)

    print(f"Saved puzzle {i} (difficulty: {verification.difficulty.value})")
```

### Create Training/Test Split

```python
from puzzle_generator import generate_puzzle_dataset

# Generate large dataset
full_dataset = generate_puzzle_dataset(num_puzzles=1000)

# Split into train/validation/test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))

train_puzzles = full_dataset.puzzles[:train_size]
val_puzzles = full_dataset.puzzles[train_size:train_size + val_size]
test_puzzles = full_dataset.puzzles[train_size + val_size:]

# Save splits
from puzzle_generator import PuzzleDataset

train_dataset = PuzzleDataset()
for p, v in train_puzzles:
    train_dataset.add_puzzle(p, v)
train_dataset.save("train.json")

# Similar for val and test...
```

## Testing Your Solver

### Evaluate Solver Performance

```python
def my_solver(puzzle):
    """Your solver implementation."""
    # Return predicted output for test input
    pass

# Test on generated puzzles
correct = 0
total = 0

dataset = generate_puzzle_dataset(num_puzzles=20)

for puzzle, verification in dataset.puzzles:
    test_input = puzzle.test_inputs[0]
    expected_output = puzzle.test_outputs[0] if puzzle.test_outputs else None

    predicted_output = my_solver(puzzle)

    if predicted_output == expected_output:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Solver accuracy: {accuracy:.1%}")
```

## Configuration Options

### GeneratorConfig Parameters

```python
from puzzle_generator import GeneratorConfig, GenerationStrategy

config = GeneratorConfig(
    # Strategy
    strategy=GenerationStrategy.HYBRID,  # or CONSTRAINT_BASED, CELLULAR_AUTOMATA

    # Puzzle structure
    num_train_examples=3,      # Number of input-output training pairs
    num_test_examples=1,       # Number of test inputs

    # Grid dimensions
    min_grid_size=5,
    max_grid_size=15,
    num_colors=10,             # Max colors to use (0-9)

    # Search parameters
    max_transform_depth=3,     # Max composition depth
    beam_width=5,              # Beam search width

    # CA parameters
    ca_steps=10,
    ca_rule="life",           # or "seeds", "brian_brain", "maze", etc.

    # Verification
    min_search_time_ms=100,   # Minimum solving time
    max_search_time_ms=5000,  # Maximum solving time
    min_mdl=2,                # Minimum description length
    max_mdl=6,                # Maximum description length

    # Filtering
    use_adversarial_filtering=True,
    diversity_threshold=0.8
)
```

## Troubleshooting

### Problem: No Puzzles Generated

```python
# Relax constraints
config = GeneratorConfig(
    min_search_time_ms=50,     # Lower minimum
    max_search_time_ms=10000,  # Higher maximum
    min_mdl=1,                 # Allow simpler
    max_mdl=8                  # Allow more complex
)
```

### Problem: Puzzles Too Easy/Hard

```python
# For harder puzzles
config = GeneratorConfig(
    min_search_time_ms=1000,
    min_mdl=3,
    max_transform_depth=4
)

# For easier puzzles
config = GeneratorConfig(
    max_search_time_ms=2000,
    max_mdl=4,
    max_transform_depth=2
)
```

### Problem: Low Diversity

```python
config = GeneratorConfig(
    use_adversarial_filtering=True,
    diversity_threshold=0.6,    # Lower threshold
    strategy=GenerationStrategy.HYBRID  # Mix strategies
)
```

## Best Practices

1. **Start Small**: Generate 5-10 puzzles first to test configuration
2. **Monitor Statistics**: Check diversity and difficulty distribution
3. **Use Hybrid Strategy**: Mixes constraint-based and CA generation
4. **Enable Adversarial Filtering**: Prevents duplicate patterns
5. **Verify Quality**: Check consistency and elegance scores
6. **Save Regularly**: Large generations can take time
7. **Parallelize**: Use multiprocessing for large datasets

## Next Steps

- Read `DOCUMENTATION.md` for technical details
- Explore `examples.py` for more use cases
- Extend the system with custom transformations
- Create domain-specific puzzle generators
- Benchmark your AGI models

## Support

For issues and questions:
- GitHub Issues: https://github.com/AvniKanodia/re-arc/issues
- Documentation: See `DOCUMENTATION.md`
- Examples: See `examples.py`
