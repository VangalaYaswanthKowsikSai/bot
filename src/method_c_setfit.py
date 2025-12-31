"""
Method C: Light Fine-Tuning with SetFit
Uses SetFit (Sentence Transformer Fine-tuning) for efficient few-shot learning.
Fast training on CPU/free-tier compute.
"""

import time
import pandas as pd
import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')


class SetFitClassifier:
    def __init__(self, model_name="sentence-transformers/paraphrase-mpnet-base-v2"):
        """Initialize SetFit classifier."""
        self.model_name = model_name
        self.model = None
        self.training_time = 0
        self.inference_latencies = []

    def train(self, train_df, num_epochs=1):
        """Train SetFit model on training data."""
        print(f"Initializing SetFit model: {self.model_name}")
        print(f"Training samples: {len(train_df)}")
        print(f"Epochs: {num_epochs}\n")

        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])

        # Initialize SetFit model
        self.model = SetFitModel.from_pretrained(self.model_name)

        # Training arguments
        args = TrainingArguments(
            batch_size=16,
            num_epochs=num_epochs,
            evaluation_strategy="no",
            save_strategy="no",
            logging_steps=10,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            column_mapping={"text": "text", "label": "label"}
        )

        # Train model
        start_time = time.time()
        trainer.train()
        self.training_time = time.time() - start_time

        print(f"\nTraining completed in {self.training_time:.2f} seconds")

    def predict(self, texts):
        """Predict sentiment for a list of texts."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        start_time = time.time()
        predictions = self.model.predict(texts)
        latency = time.time() - start_time

        # Store individual latencies (approximate)
        avg_latency = latency / len(texts)
        self.inference_latencies.extend([avg_latency] * len(texts))

        return predictions

    def evaluate(self, test_df):
        """Evaluate model on test data."""
        print(f"\nEvaluating on {len(test_df)} test samples...")

        predictions = self.predict(test_df['text'].tolist())
        y_true = test_df['label'].values

        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        avg_latency = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0

        results = {
            "method": "SetFit Fine-Tuning",
            "model": self.model_name,
            "training_time_sec": self.training_time,
            "accuracy": accuracy,
            "f1_score": f1,
            "avg_latency_ms": avg_latency * 1000,
            "total_inference_time_sec": sum(self.inference_latencies),
            "predictions": predictions.tolist(),
            "classification_report": classification_report(y_true, predictions,
                                                          target_names=['negative', 'positive', 'neutral'],
                                                          output_dict=True)
        }

        return results


def main():
    """Main execution function."""
    # Load data
    df = pd.read_csv("data/financial_tweets.csv")

    # Split into train and test (using same test set as other methods)
    test_df = df.sample(n=30, random_state=42)
    remaining_df = df.drop(test_df.index)

    # Use 20 samples for training (few-shot learning scenario)
    train_df = remaining_df.sample(n=20, random_state=42)

    print("="*60)
    print("SETFIT FINE-TUNING")
    print("="*60)
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}\n")

    # Initialize and train classifier
    classifier = SetFitClassifier()
    classifier.train(train_df, num_epochs=1)

    # Evaluate
    results = classifier.evaluate(test_df)

    # Print results
    print("\n" + "="*60)
    print("SETFIT CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Training Time: {results['training_time_sec']:.2f} seconds")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_score']:.4f}")
    print(f"Average Inference Latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Total Inference Time: {results['total_inference_time_sec']:.4f} seconds")
    print("\nDetailed Classification Report:")
    print(json.dumps(results['classification_report'], indent=2))

    # Save results
    with open("results/setfit_results.json", "w") as f:
        save_results = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(save_results, f, indent=2)

    print("\nResults saved to results/setfit_results.json")

    # Optionally save the model
    if classifier.model:
        classifier.model.save_pretrained("results/setfit_model")
        print("Model saved to results/setfit_model")


if __name__ == "__main__":
    main()
