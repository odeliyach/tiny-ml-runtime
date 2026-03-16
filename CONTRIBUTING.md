# Contributing to Tiny ML Runtime

Thank you for your interest in contributing to Tiny ML Runtime! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Development Setup

### Prerequisites
- GCC or Clang compiler (C11 compatible)
- Python 3.8+
- Git

### Initial Setup

1. **Clone the repository**:
```bash
git clone https://github.com/odeliyach/tiny-ml-runtime.git
cd tiny-ml-runtime
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install pre-commit hooks**:
```bash
pre-commit install
```

4. **Build the project**:
```bash
make clean && make
```

5. **Run tests**:
```bash
make test
python -m pytest tests/
```

## Coding Standards

### C Code Style
- Follow the `.clang-format` configuration
- Use 4 spaces for indentation (no tabs)
- Function names: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Maximum line length: 100 characters
- Always check return values
- Free all allocated memory

**Example**:
```c
/**
 * @brief Performs matrix multiplication: out = W @ in + b
 *
 * @param in Input vector (size: cols)
 * @param W Weight matrix (size: rows × cols, row-major)
 * @param b Bias vector (size: rows)
 * @param out Output vector (size: rows)
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns in weight matrix
 */
void linear(float *in, float *W, float *b, float *out, int rows, int cols) {
    // Implementation
}
```

### Python Code Style
- Follow PEP 8 (enforced by Black and Flake8)
- Use Google-style docstrings
- Type hints for all function signatures
- Maximum line length: 100 characters

**Example**:
```python
def train_model(
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01
) -> nn.Module:
    """Trains a neural network model on the provided data.

    Args:
        data: Input features as numpy array of shape (N, D).
        labels: Target labels as numpy array of shape (N,).
        epochs: Number of training epochs. Defaults to 100.
        learning_rate: Learning rate for optimizer. Defaults to 0.01.

    Returns:
        Trained PyTorch model.

    Raises:
        ValueError: If data and labels have incompatible shapes.
    """
    # Implementation
```

## Commit Message Guidelines

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic changes)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `chore`: Build process or tooling changes

**Examples**:
```
feat(inference): Add support for convolutional layers

Implement conv2d operation with NEON intrinsics for ARM optimization.
Includes unit tests and benchmark results.

Closes #42
```

```
fix(training): Resolve numerical instability in softmax

Subtract max value before exp to prevent overflow.
Added property-based tests to verify sum(softmax) == 1.0
```

## Pull Request Process

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run the test suite**:
```bash
make test
python -m pytest tests/ -v --cov
```

4. **Format your code**:
```bash
black src/python/
clang-format -i src/c/*.c
```

5. **Commit your changes**:
```bash
git add .
git commit -m "feat: your descriptive commit message"
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

7. **Open a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI passes

### PR Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally and in CI
- [ ] Documentation updated
- [ ] No new compiler warnings
- [ ] Memory leaks checked (Valgrind clean)
- [ ] Performance impact assessed (if applicable)

## Testing Guidelines

### Unit Tests (C)
Location: `tests/unit/test_inference.c`

```c
void test_matrix_multiply() {
    float W[] = {1, 2, 3, 4};  // 2x2 matrix
    float in[] = {1, 1};
    float b[] = {0, 0};
    float out[2];

    linear(in, W, b, out, 2, 2);

    assert(fabs(out[0] - 3.0) < 1e-6);  // 1*1 + 2*1 = 3
    assert(fabs(out[1] - 7.0) < 1e-6);  // 3*1 + 4*1 = 7
}
```

### Unit Tests (Python)
Location: `tests/unit/test_*.py`

```python
import pytest
import numpy as np
from src.python.train import train_model

def test_model_training():
    """Test that model training improves accuracy."""
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, 100)

    model = train_model(X, y, epochs=10)

    # Model should achieve >50% accuracy on training data
    predictions = model(torch.tensor(X)).argmax(dim=1)
    accuracy = (predictions == torch.tensor(y)).float().mean()
    assert accuracy > 0.5
```

### Integration Tests
Location: `tests/integration/test_end_to_end.py`

```python
def test_train_export_infer_pipeline():
    """Test complete pipeline from training to inference."""
    # Train model
    train_model(...)

    # Export weights
    export_weights(...)

    # Run C inference
    result = subprocess.run(['./inference', 'test'], capture_output=True)

    # Verify output
    assert result.returncode == 0
    assert "Accuracy" in result.stdout.decode()
```

## Performance Benchmarking

When making performance changes, include benchmark results:

```bash
# Before changes
make inference
./inference iris
# Record: 2,732,240 predictions/sec

# After changes
# Record: 3,100,000 predictions/sec
# Improvement: +13.5%
```

Use `perf` for detailed profiling:
```bash
perf record -g ./inference iris
perf report
```

## Documentation

### Code Documentation
- **C functions**: Use Doxygen comments
- **Python functions**: Use Google-style docstrings
- **Complex algorithms**: Add inline comments explaining the "why", not the "what"

### User Documentation
- Update `README.md` for user-facing changes
- Add examples to `docs/examples/`
- Update `docs/ARCHITECTURE.md` for design changes

## Security

### Reporting Vulnerabilities
Do not open public issues for security vulnerabilities. Email: security@example.com

### Security Best Practices
- Validate all input dimensions
- Check return values from `malloc`, `fread`, etc.
- Avoid buffer overflows (use bounds checking)
- Sanitize file paths
- Run Valgrind and AddressSanitizer:

```bash
# Memory leak detection
valgrind --leak-check=full ./inference iris

# Address sanitizer
gcc -fsanitize=address -g inference.c -o inference -lm
./inference iris
```

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an issue with reproduction steps
- **Feature requests**: Open an issue with use case description

Thank you for contributing! 🚀
