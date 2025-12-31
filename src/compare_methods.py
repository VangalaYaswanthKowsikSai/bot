"""
ROI Comparison: Prompt Engineering vs Fine-Tuning
Compares Zero-Shot, Few-Shot, and SetFit approaches on key metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class ROIComparison:
    def __init__(self):
        """
        Initialize ROI comparison.

        All models are now 100% FREE and run locally - no API costs!
        """
        # All methods are FREE - using open-source models
        self.api_cost = 0.00

    def load_results(self):
        """Load results from all three methods."""
        results = {}

        result_files = {
            "Zero-Shot": "results/zeroshot_results.json",
            "Few-Shot": "results/fewshot_results.json",
            "SetFit": "results/setfit_results.json"
        }

        for method, filepath in result_files.items():
            try:
                with open(filepath, 'r') as f:
                    results[method] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {filepath} not found. Run the corresponding script first.")
                results[method] = None

        return results

    def calculate_cost_per_1k(self, method, num_samples=1000):
        """
        Calculate cost per 1000 predictions.

        All methods are FREE (open-source models running locally)!
        """
        return 0.00  # All methods are FREE!

    def create_comparison_table(self, results):
        """Create comprehensive comparison table."""
        comparison_data = []

        for method, data in results.items():
            if data is None:
                continue

            row = {
                "Method": method,
                "Accuracy": f"{data['accuracy']:.4f}",
                "F1-Score": f"{data['f1_score']:.4f}",
                "Avg Latency (ms)": f"{data.get('avg_latency_ms', 0):.2f}",
                "API Cost": "FREE! 🔓"
            }

            if method == "SetFit":
                row["Training Time (s)"] = f"{data.get('training_time_sec', 0):.2f}"
            else:
                row["Training Time (s)"] = "0.00"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def plot_accuracy_comparison(self, results, save_path="results/accuracy_comparison.png"):
        """Plot accuracy and F1-score comparison."""
        methods = []
        accuracies = []
        f1_scores = []

        for method, data in results.items():
            if data is None:
                continue
            methods.append(method)
            accuracies.append(data['accuracy'] * 100)
            f1_scores.append(data['f1_score'] * 100)

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71')

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy & F1-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim([0, 105])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    def plot_latency_comparison(self, results, save_path="results/latency_comparison.png"):
        """Plot latency comparison."""
        methods = []
        latencies = []

        for method, data in results.items():
            if data is None:
                continue
            methods.append(method)
            latencies.append(data.get('avg_latency_ms', 0))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, latencies, color=['#e74c3c', '#f39c12', '#9b59b6'])

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    def plot_model_size_comparison(self, results, save_path="results/model_size_comparison.png"):
        """Plot approximate model size comparison."""
        # Approximate model sizes in GB
        model_sizes = {
            "Zero-Shot": 1.6,   # facebook/bart-large-mnli ~1.6GB
            "Few-Shot": 0.85,   # google/flan-t5-base ~850MB
            "SetFit": 0.42      # sentence-transformers/paraphrase-mpnet-base-v2 ~420MB
        }

        methods = []
        sizes = []

        for method in results.keys():
            if results[method] is None:
                continue
            methods.append(method)
            sizes.append(model_sizes.get(method, 0))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, sizes, color=['#1abc9c', '#16a085', '#27ae60'])

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Size (GB)', fontsize=12, fontweight='bold')
        ax.set_title('Model Size Comparison - All FREE! 🔓', fontsize=14, fontweight='bold')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f} GB',
                   ha='center', va='bottom', fontsize=10)

        # Add note that all are FREE
        ax.text(0.5, 0.95, '💰 API Cost: $0.00 for all methods!',
                ha='center', transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    def plot_roi_matrix(self, results, save_path="results/roi_matrix.png"):
        """Create decision matrix: Accuracy vs Latency."""
        methods = []
        accuracies = []
        latencies = []

        for method, data in results.items():
            if data is None:
                continue
            methods.append(method)
            accuracies.append(data['accuracy'] * 100)
            latencies.append(data.get('avg_latency_ms', 0))

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        for i, method in enumerate(methods):
            ax.scatter(latencies[i], accuracies[i], s=500, alpha=0.6,
                      color=colors[i], edgecolors='black', linewidth=2)
            ax.annotate(method, (latencies[i], accuracies[i]),
                       fontsize=12, fontweight='bold',
                       ha='center', va='center')

        ax.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Decision Matrix: Accuracy vs Latency - All FREE! 🔓', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add quadrant lines
        if latencies and accuracies:
            ax.axhline(y=np.mean(accuracies), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.mean(latencies), color='gray', linestyle='--', alpha=0.5)

        # Add note about free models
        ax.text(0.5, 0.02, '💰 All models are FREE and run locally!',
                ha='center', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    def generate_report(self, results, comparison_df):
        """Generate markdown report."""
        report = """# Prompt Engineering vs. Fine-Tuning: ROI Analysis

## 🔓 100% FREE - No API Keys Required!

All methods use **open-source models running locally** - zero API costs!

## Executive Summary

This experiment compares three approaches to financial sentiment analysis:

1. **Zero-Shot Prompting**: Direct classification with no examples (BART-large-mnli)
2. **Few-Shot Prompting**: Classification with 3 in-context examples (FLAN-T5-base)
3. **SetFit Fine-Tuning**: Lightweight fine-tuning on 20 samples (paraphrase-mpnet)

**Key Advantage**: All models run locally without API keys or usage costs!

## Results Summary

"""
        # Add comparison table
        report += "### Performance Comparison\n\n"
        report += comparison_df.to_markdown(index=False)
        report += "\n\n"

        # Key insights
        report += """## Key Insights

### Accuracy Analysis
"""
        for method, data in results.items():
            if data is None:
                continue
            report += f"- **{method}**: {data['accuracy']:.4f} accuracy, {data['f1_score']:.4f} F1-score\n"

        report += """
### Cost Analysis
**All methods are 100% FREE!** 🔓
- **Zero-Shot**: $0.00 (no API costs, runs locally)
- **Few-Shot**: $0.00 (no API costs, runs locally)
- **SetFit**: $0.00 (no API costs, runs locally)

**Total savings**: Unlimited! No API usage fees ever.

### Latency Analysis
"""
        for method, data in results.items():
            if data is None:
                continue
            latency = data.get('avg_latency_ms', 0)
            report += f"- **{method}**: {latency:.2f}ms average latency\n"

        report += """
### Model Size
- **Zero-Shot (BART-large)**: ~1.6GB download (one-time)
- **Few-Shot (FLAN-T5-base)**: ~850MB download (one-time)
- **SetFit (paraphrase-mpnet)**: ~420MB download (one-time)

## Recommendations

**For Production Use:**
- **Best accuracy needed**: SetFit (highest accuracy after training)
- **Quick prototyping**: Few-Shot prompting (good accuracy without training)
- **Zero setup required**: Zero-Shot (instant deployment, no examples needed)
- **Smallest model**: SetFit (~420MB)
- **Fastest inference**: SetFit (typically faster than LLMs)

**Winner**: All methods are FREE! Choose based on accuracy vs latency trade-offs, not cost.

## Visualizations

See the `results/` directory for detailed comparison charts:
- `accuracy_comparison.png` - Accuracy & F1-Score comparison
- `latency_comparison.png` - Inference latency comparison
- `model_size_comparison.png` - Model size comparison
- `roi_matrix.png` - Accuracy vs Latency decision matrix

## Conclusion

**All methods are 100% FREE and run locally!** 🔓

This analysis demonstrates that you don't need expensive API keys to build powerful NLP applications:
- **Zero-Shot**: Great for instant deployment without training data
- **Few-Shot**: Improved accuracy with just 3 examples in the prompt
- **SetFit**: Best accuracy with minimal training data

**The real ROI**: No API costs, full control, and privacy-preserving local execution!

---
*Generated automatically using 100% free, open-source models*
"""

        # Save report
        with open("results/REPORT.md", "w") as f:
            f.write(report)

        print("\nReport saved to results/REPORT.md")


def main():
    """Main execution function."""
    print("="*60)
    print("ROI COMPARISON: PROMPT ENGINEERING VS. FINE-TUNING")
    print("="*60)

    # Initialize comparison
    roi = ROIComparison()

    # Load results
    print("\nLoading results from all methods...")
    results = roi.load_results()

    # Check if we have any results
    if all(v is None for v in results.values()):
        print("\nError: No results found. Please run the individual method scripts first:")
        print("  - python src/method_a_zeroshot.py")
        print("  - python src/method_b_fewshot.py")
        print("  - python src/method_c_setfit.py")
        return

    # Create comparison table
    print("\nGenerating comparison table...")
    comparison_df = roi.create_comparison_table(results)
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv("results/comparison_table.csv", index=False)
    print("\nComparison table saved to results/comparison_table.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    roi.plot_accuracy_comparison(results)
    roi.plot_latency_comparison(results)
    roi.plot_model_size_comparison(results)
    roi.plot_roi_matrix(results)

    # Generate report
    print("\nGenerating comprehensive report...")
    roi.generate_report(results, comparison_df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nAll results saved to the 'results/' directory")
    print("Check results/REPORT.md for the full analysis")


if __name__ == "__main__":
    main()
