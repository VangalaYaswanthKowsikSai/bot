"""
Method B: Few-Shot Prompting
Uses HuggingFace text-generation model with few-shot examples (100% FREE, NO API KEY!)
Model: google/flan-t5-base (runs locally on CPU/GPU)
"""

import time
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')


class FewShotClassifier:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initialize Few-Shot classifier with HuggingFace pipeline.

        Model downloads automatically on first run (~850MB).
        Runs on CPU (no GPU required, but GPU is faster if available).
        """
        print(f"Loading model: {model_name}")
        print("⏳ First run will download model (~850MB)...")
        print("🔓 100% FREE - No API key needed!\n")

        self.model_name = model_name
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1,  # -1 for CPU, 0 for GPU
            max_length=10
        )
        self.latencies = []

        # Define few-shot examples (included in every prompt)
        self.examples = """Classify the sentiment of financial tweets as positive, negative, or neutral.

Example 1:
Tweet: "$AAPL beating earnings expectations! Strong quarter ahead"
Sentiment: positive

Example 2:
Tweet: "$TSLA disappointing delivery numbers, below estimates"
Sentiment: negative

Example 3:
Tweet: "Federal Reserve maintains current interest rates"
Sentiment: neutral
"""

        print("✓ Model loaded successfully!\n")

    def classify(self, text):
        """Classify a single financial tweet with few-shot prompting."""
        # Create prompt with examples
        prompt = f"""{self.examples}

Now classify this tweet:
Tweet: {text}
Sentiment:"""

        start_time = time.time()

        try:
            # Generate response
            result = self.generator(
                prompt,
                max_new_tokens=5,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False
            )

            latency = time.time() - start_time
            self.latencies.append(latency)

            # Extract sentiment from generated text
            generated_text = result[0]['generated_text'].strip().lower()

            # Map to numeric labels
            if 'positive' in generated_text:
                return 1
            elif 'negative' in generated_text:
                return 0
            elif 'neutral' in generated_text:
                return 2
            else:
                # Default to neutral if unclear
                return 2

        except Exception as e:
            print(f"Error classifying: {e}")
            self.latencies.append(0)
            return 2  # Default to neutral on error

    def evaluate(self, df):
        """Evaluate on entire dataset."""
        print(f"Starting Few-Shot Classification with {self.model_name}...")
        print(f"Dataset size: {len(df)} samples")
        print(f"Using 3 examples in prompt")
        print(f"Running on: {'GPU' if self.generator.device.type == 'cuda' else 'CPU'}\n")

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
            "method": "Few-Shot (HuggingFace)",
            "model": self.model_name,
            "num_examples": 3,
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

    # Use same test set as zero-shot for fair comparison
    test_df = df.sample(n=30, random_state=42)

    print("="*60)
    print("FEW-SHOT PROMPTING")
    print("100% FREE - No API Key Required!")
    print("="*60 + "\n")

    # Initialize and run few-shot classifier
    classifier = FewShotClassifier()
    results = classifier.evaluate(test_df)

    # Print results
    print("\n" + "="*60)
    print("FEW-SHOT CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Examples in prompt: {results['num_examples']}")
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
    with open("results/fewshot_results.json", "w") as f:
        save_results = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(save_results, f, indent=2)

    print("\n✓ Results saved to results/fewshot_results.json")


if __name__ == "__main__":
    main()
