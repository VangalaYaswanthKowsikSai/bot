# Contributing to Prompt Engineering vs. Fine-Tuning ROI

Thank you for your interest in contributing to this project! This guide will help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for:
- New classification methods to compare
- Additional metrics or visualizations
- Performance improvements
- Documentation improvements

### Code Contributions

#### 1. Fork and Clone

```bash
git fork https://github.com/VangalaYaswanthKowsikSai/bot.git
git clone https://github.com/YOUR_USERNAME/bot.git
cd prompt-vs-finetune-roi
```

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

#### 3. Make Your Changes

Follow the project structure:
```
src/
├── method_a_zeroshot.py      # Zero-shot classification
├── method_b_fewshot.py        # Few-shot prompting
├── method_c_setfit.py         # SetFit fine-tuning
└── compare_methods.py         # Comparison & visualization
```

#### 4. Code Style Guidelines

- **PEP 8**: Follow Python style guidelines
- **Docstrings**: Use clear docstrings for functions
- **Comments**: Explain complex logic
- **Type hints**: Use type hints where appropriate

**Example:**
```python
def classify(self, text: str) -> int:
    """
    Classify a single financial tweet.

    Args:
        text (str): The tweet text to classify

    Returns:
        int: Predicted label (0=negative, 1=positive, 2=neutral)
    """
    # Implementation here
    pass
```

#### 5. Testing

Before submitting, test your changes:

```bash
# Run individual methods
python src/method_a_zeroshot.py
python src/method_b_fewshot.py
python src/method_c_setfit.py

# Run full pipeline
python run_all.py

# Verify results are generated
ls results/
```

#### 6. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add: New method for LoRA fine-tuning"
# or
git commit -m "Fix: Handle API timeout errors gracefully"
# or
git commit -m "Docs: Improve installation instructions"
```

**Commit message format:**
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements to existing features
- `Docs:` for documentation changes
- `Refactor:` for code refactoring

#### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then:
1. Go to your fork on GitHub
2. Click "Pull Request"
3. Fill in the PR template with:
   - Description of changes
   - Motivation/rationale
   - Testing performed
   - Screenshots (if applicable)

## Adding New Methods

To add a new classification method:

### 1. Create Method Script

Create `src/method_d_yourmethod.py`:

```python
"""
Method D: Your Method Name
Brief description of the approach.
"""

import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

class YourMethodClassifier:
    def __init__(self):
        """Initialize your classifier."""
        self.latencies = []

    def classify(self, text):
        """Classify a single text."""
        # Your implementation
        pass

    def evaluate(self, df):
        """Evaluate on entire dataset."""
        predictions = []
        for text in df['text']:
            pred = self.classify(text)
            predictions.append(pred)

        # Calculate metrics (use consistent framework)
        y_true = df['label'].values
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')

        results = {
            "method": "Your Method Name",
            "accuracy": accuracy,
            "f1_score": f1,
            "avg_latency_ms": sum(self.latencies) / len(self.latencies) * 1000,
            "predictions": predictions,
            "classification_report": classification_report(
                y_true, predictions,
                target_names=['negative', 'positive', 'neutral'],
                output_dict=True
            )
        }

        return results

def main():
    df = pd.read_csv("data/financial_tweets.csv")
    test_df = df.sample(n=30, random_state=42)

    classifier = YourMethodClassifier()
    results = classifier.evaluate(test_df)

    # Print and save results
    print(f"Accuracy: {results['accuracy']:.4f}")
    with open("results/yourmethod_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

### 2. Update Comparison Script

Edit `src/compare_methods.py`:

```python
# Add to result_files dictionary
result_files = {
    "Zero-Shot": "results/zeroshot_results.json",
    "Few-Shot": "results/fewshot_results.json",
    "SetFit": "results/setfit_results.json",
    "Your Method": "results/yourmethod_results.json",  # Add this
}

# Add pricing if applicable
self.pricing = {
    "Zero-Shot": {"input": 0.05, "output": 0.08},
    "Few-Shot": {"input": 0.05, "output": 0.08},
    "SetFit": {"train": 0.00, "inference": 0.00},
    "Your Method": {"input": 0.XX, "output": 0.XX},  # Add this
}
```

### 3. Update Documentation

- Add method description to `README.md`
- Update comparison table
- Add any new dependencies to `requirements.txt`

### 4. Test Integration

```bash
python src/method_d_yourmethod.py
python src/compare_methods.py
```

Verify:
- Results JSON is generated
- Method appears in comparison visualizations
- No errors in the pipeline

## Adding New Datasets

To support a new dataset:

### 1. Prepare Data Format

Ensure CSV has required columns:
```csv
text,label
"Sample text",0
"Another sample",1
```

### 2. Add to data/ Directory

```bash
data/
├── financial_tweets.csv       # Original
└── your_dataset.csv           # New dataset
```

### 3. Create Dataset-Specific Script (Optional)

```python
# src/run_on_custom_dataset.py
import pandas as pd
from method_a_zeroshot import ZeroShotClassifier
# ... etc

df = pd.read_csv("data/your_dataset.csv")
# Run experiments
```

### 4. Document Dataset

In `README.md`, add:
- Dataset description
- Source/citation
- Statistics (size, classes, etc.)

## Code Review Process

All contributions go through review:

1. **Automated checks**: Code style, basic tests
2. **Manual review**: Logic, efficiency, documentation
3. **Feedback**: Maintainers may request changes
4. **Approval**: Once approved, PR will be merged

## Community Guidelines

- Be respectful and constructive
- Focus on the code, not the person
- Help others learn and grow
- Celebrate contributions of all sizes

## Recognition

Contributors will be:
- Added to `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in documentation

## Questions?

- Create a discussion: https://github.com/VangalaYaswanthKowsikSai/bot/discussions
- Email: [your-email@example.com]


