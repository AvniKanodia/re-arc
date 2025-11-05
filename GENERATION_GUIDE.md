# Puzzle Generation Guide

## Quick Start

### Generate Demo Puzzles (Fast)

```bash
python quick_demo.py
```

This creates 5 hand-crafted example puzzles demonstrating different transformations:
- Tiling Pattern
- Mirror Reflection
- Spiral Color Rotation
- Color Gradient
- Diagonal Flip

Output: `demo_visualizations/` folder with PNG images

### Generate Comprehensive Dataset

```bash
python generate_datasets.py
```

This generates a diverse dataset using:
- 15 custom transformation puzzles
- 10 "aha moment" optimized puzzles
- 5 cellular automata puzzles
- 10 standard constraint-based puzzles

Output:
- `puzzle_datasets/comprehensive_dataset.json` - Full dataset in ARC format
- `puzzle_visualizations/*.png` - PNG images for each puzzle
- `puzzle_visualizations/dataset_summary.png` - Overview of samples

**Note**: Full generation takes 5-15 minutes depending on hardware.

## Understanding Puzzle Types

### Custom Transform Puzzles

Use novel transformations like:
- **TilePattern**: Repeats pattern in 2x2 or 3x3 grid
- **Mirror**: Reflects and doubles the grid
- **SpiralRotate**: Colors change based on distance from center
- **DiagonalFlip**: Flips along anti-diagonal
- **ColorGradient**: Colors shift based on position
- **ConnectObjects**: Draws lines between object centers
- **OutlineObjects**: Replaces objects with outlines
- **ZoomCenter**: Zooms into center region

### "Aha Moment" Puzzles

Optimized for insight-based solving using priors:
- **PatternRepetition**: "Oh, it's tiling the input!"
- **ObjectRelationship**: "It sorts objects by size!"
- **BorderSpecial**: "Only the border changes!"
- **EnclosureRule**: "It fills enclosed regions!"
- **DiagonalSymmetry**: "It's diagonal symmetry!"

### Cellular Automata Puzzles

Emergent patterns from simple rules:
- **maze_builder**: Creates labyrinthine structures
- **crystal**: Snowflake-like growth patterns
- **diamond**: Cross-shaped propagation
- **coral_reef**: Organic branching structures
- **waves**: Oscillating patterns

## Customization

### Adjust Difficulty

Edit `GeneratorConfig` in `generate_datasets.py`:

```python
config = GeneratorConfig(
    min_search_time_ms=50,      # Lower = easier puzzles
    max_search_time_ms=10000,   # Higher = harder puzzles
    min_mdl=1,                  # Simpler transformations
    max_mdl=8,                  # More complex transformations
)
```

### Create Custom Puzzles

1. **Add Custom Transform** (`custom_transforms.py`):
```python
class MyTransform(Transform):
    def apply(self, grid: Grid) -> Grid:
        # Your transformation logic
        return new_grid
```

2. **Add Custom Prior** (`custom_priors.py`):
```python
class MyPrior(SemanticPrior):
    def evaluate(self, input_grid, output_grid) -> float:
        # Return 0.0-1.0 score
        return score
```

3. **Add to Registry**:
```python
CUSTOM_TRANSFORMS.append(MyTransform())
CUSTOM_PRIORS.append(MyPrior())
```

### Generate Specific Puzzle Types

```python
from generate_datasets import ExtendedPuzzleGenerator
from puzzle_generator import GeneratorConfig

config = GeneratorConfig()
generator = ExtendedPuzzleGenerator(config)

# Generate only custom transform puzzles
puzzles = generator.generate_with_custom_transforms(num_puzzles=20)

# Generate only "aha moment" puzzles
puzzles = generator.generate_with_aha_priors(num_puzzles=15)

# Generate only CA puzzles
puzzles = generator.generate_with_custom_ca(num_puzzles=10)
```

## Visualization Options

### Visualize Single Puzzle

```python
from visualization import create_puzzle_grid_visualization

create_puzzle_grid_visualization(
    puzzle,
    output_dir="my_puzzles",
    filename="puzzle_001",
    visualizer_type='color'  # or 'shape', 'colorshape'
)
```

### Create Dataset Summary

```python
from visualization import create_dataset_summary_visualization

create_dataset_summary_visualization(
    puzzles,
    output_path="summary.png",
    samples=6  # Number of samples to show
)
```

## Puzzle Quality Metrics

Generated puzzles are evaluated on:

1. **Difficulty**: Based on program synthesis search time
   - TOO_EASY: < 100ms
   - EASY: 100-1500ms
   - MEDIUM: 1500-3500ms
   - HARD: 3500-5000ms
   - TOO_HARD: > 5000ms or unsolvable

2. **Elegance**: Ratio of execution complexity to description complexity
   - High = simple rule, complex result

3. **Consistency**: All training examples follow same rule

4. **Diversity**: Variety across dataset

## Troubleshooting

### "No puzzles generated"

The verification filters may be too strict. Try:
```python
config = GeneratorConfig(
    min_search_time_ms=50,    # Relaxed from 100
    max_search_time_ms=15000, # Increased from 5000
    min_mdl=1,                # Relaxed from 2
    max_mdl=10                # Increased from 6
)
```

### "All puzzles marked as TOO_HARD"

This is common with CA puzzles (they're genuinely hard!). They're still valid and interesting - the verification just couldn't solve them quickly. These make excellent AGI test cases.

### "Generation is slow"

- Use `quick_demo.py` for fast examples
- Reduce `num_puzzles` in `generate_datasets.py`
- Increase `max_attempts` ratio
- Use simpler transforms

## Advanced: Parallel Generation

```python
import multiprocessing as mp

def generate_batch(n):
    generator = ExtendedPuzzleGenerator(config)
    return generator.generate_with_custom_transforms(n)

with mp.Pool(4) as pool:
    results = pool.map(generate_batch, [10, 10, 10, 10])

all_puzzles = [p for batch in results for p in batch]
```

## Dataset Format

Generated JSON follows ARC format:

```json
{
  "train": [
    {"input": [[...]], "output": [[...]]},
    {"input": [[...]], "output": [[...]]}
  ],
  "test": [
    {"input": [[...]], "output": [[...]]}
  ],
  "metadata": {
    "strategy": "custom_transform",
    "transform": "tile_2x2",
    "difficulty": "medium"
  }
}
```

## Examples Gallery

After running `quick_demo.py`, check `demo_visualizations/` for:
- Visual input/output pairs
- Clear grid structure
- ARC-style color palette
- Size annotations

## Contributing New Puzzle Types

1. Create transformation in `custom_transforms.py`
2. Create corresponding prior in `custom_priors.py`
3. Test with quick demo
4. Add to comprehensive generation
5. Verify "aha moment" quality on humans!

## Performance Benchmarks

On typical hardware:
- Quick demo: ~10 seconds
- 10 custom puzzles: ~2-3 minutes
- 10 aha puzzles: ~3-5 minutes
- 5 CA puzzles: ~1-2 minutes
- Full dataset (40 puzzles): ~10-15 minutes

## Citation

If you use this puzzle generator in your research, please cite:

```
re-arc: A Generalized Puzzle Generator for AGI Benchmarks
GitHub: https://github.com/AvniKanodia/re-arc
```
