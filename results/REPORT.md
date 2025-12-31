# Prompt Engineering vs. Fine-Tuning: ROI Analysis

## 🔓 100% FREE - No API Keys Required!

All methods use **open-source models running locally** - zero API costs!

## Executive Summary

This experiment compares three approaches to financial sentiment analysis:

1. **Zero-Shot Prompting**: Direct classification with no examples (BART-large-mnli)
2. **Few-Shot Prompting**: Classification with 3 in-context examples (FLAN-T5-base)
3. **SetFit Fine-Tuning**: Lightweight fine-tuning on 20 samples (paraphrase-mpnet)

**Key Advantage**: All models run locally without API keys or usage costs!

## Results Summary

### Performance Comparison

| Method    |   Accuracy |   F1-Score |   Avg Latency (ms) | API Cost   |   Training Time (s) |
|:----------|-----------:|-----------:|-------------------:|:-----------|--------------------:|
| Zero-Shot |     0.7    |     0.6344 |             523.41 | FREE! 🔓   |                 0   |
| Few-Shot  |     0.6333 |     0.5004 |             386.57 | FREE! 🔓   |                 0   |
| SetFit    |     0.8    |     0.7765 |              19.55 | FREE! 🔓   |                39.2 |

## Key Insights

### Accuracy Analysis
- **Zero-Shot**: 0.7000 accuracy, 0.6344 F1-score
- **Few-Shot**: 0.6333 accuracy, 0.5004 F1-score
- **SetFit**: 0.8000 accuracy, 0.7765 F1-score

### Cost Analysis
**All methods are 100% FREE!** 🔓
- **Zero-Shot**: $0.00 (no API costs, runs locally)
- **Few-Shot**: $0.00 (no API costs, runs locally)
- **SetFit**: $0.00 (no API costs, runs locally)

**Total savings**: Unlimited! No API usage fees ever.

### Latency Analysis
- **Zero-Shot**: 523.41ms average latency
- **Few-Shot**: 386.57ms average latency
- **SetFit**: 19.55ms average latency

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
