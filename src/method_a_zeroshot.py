"""
Method A: Zero-Shot Classification
Uses HuggingFace zero-shot classification pipeline (100% FREE, NO API KEY!)
Model: facebook/bart-large-mnli (runs locally on CPU/GPU)
"""

import time
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')


class ZeroShotClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initialize Zero-Shot classifier with HuggingFace pipeline.

        Model downloads automatically on first run (~1.6GB).
        Runs on CPU (no GPU required, but GPU is faster if available).
        """
        print(f"Loading model: {model_name}")
        print("⏳ First run will download model (~1.6GB)...")
        print("🔓 100% FREE - No API key needed!\n")

        self.model_name = model_name
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1  # -1 for CPU, 0 for GPU
        )
        self.candidate_labels = ["negative", "neutral", "positive"]
        self.latencies = []

        print("✓ Model loaded successfully!\n")

    def classify(self, text):
        """Classify a single financial tweet with zero-shot classification."""
        start_time = time.time()

        try:
            # Run zero-shot classification
            result = self.classifier(
                text,
                candidate_labels=self.candidate_labels,
                hypothesis_template="This financial tweet has {} sentiment."
            )

            latency = time.time() - start_time
            self.latencies.append(latency)

            # Get predicted label
            predicted_label = result['labels'][0]

            # Map to numeric labels
            label_map = {"positive": 1, "negative": 0, "neutral": 2}
            return label_map[predicted_label]

        except Exception as e:
            print(f"Error classifying: {e}")
            self.latencies.append(0)
            return 2  # Default to neutral on error

    def evaluate(self, df):
        """Evaluate on entire dataset."""
        print(f"Starting Zero-Shot Classification with {self.model_name}...")
        print(f"Dataset size: {len(df)} samples")
        print(f"Running on: {'GPU' if self.classifier.device.type == 'cuda' else 'CPU'}\n")

        predictions = []
        for idx, text in enumerate(df['text']):
            pred = self.classify(text)
            predictions.append(pred)
            if (idx + 1) % 10 == 0:
                avg_latency = sum(self.latencies) / len(self.latencies)
                print(f"Processed {idx + 1}/{len(df)} samples (avg: {avg_latency:.3f}s per sample)")

        # Calculate metrics
        y_true = df['label'].values
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

        results = {
            "method": "Zero-Shot (HuggingFace)",
            "model": self.model_name,
            "api_cost": 0.00,  # FREE!
            "accuracy": accuracy,
            "f1_score": f1,
            "avg_latency_ms": avg_latency * 1000,
            "total_time_sec": sum(self.latencies),
            "predictions": predictions,
            "classification_report": classification_report(y_true, predictions,
                                                          target_names=['negative', 'positive', 'neutral'],
                                                          output_dict=True)
        }

        return results


def main():
    """Main execution function."""
    # Load data
    df = pd.read_csv("data/financial_tweets.csv")

    # Use a test set for evaluation (adjust as needed)
    test_df = df.sample(n=30, random_state=42)  # Sample 30 tweets for faster testing

    print("="*60)
    print("ZERO-SHOT CLASSIFICATION")
    print("100% FREE - No API Key Required!")
    print("="*60 + "\n")

    # Initialize and run zero-shot classifier
    classifier = ZeroShotClassifier()
    results = classifier.evaluate(test_df)

    # Print results
    print("\n" + "="*60)
    print("ZERO-SHOT CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"API Cost: $0.00 (FREE!)")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_score']:.4f}")
    print(f"Average Latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Total Time: {results['total_time_sec']:.2f} seconds")
    print("\nDetailed Classification Report:")

    # Print classification report in readable format
    report = results['classification_report']
    for label in ['negative', 'positive', 'neutral']:
        if label in report:
            print(f"\n{label.upper()}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall: {report[label]['recall']:.4f}")
            print(f"  F1-Score: {report[label]['f1-score']:.4f}")

    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")

    # Save results
    with open("results/zeroshot_results.json", "w") as f:
        # Remove predictions for cleaner JSON
        save_results = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(save_results, f, indent=2)

    print("\n✓ Results saved to results/zeroshot_results.json")


if __name__ == "__main__":
    main()
