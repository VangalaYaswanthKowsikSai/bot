# Prompt Engineering vs. Fine-Tuning: ROI Analysis

> **🔓 100% FREE - No API Keys Required!**
> *All models run locally using open-source HuggingFace Transformers*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-green.svg)](https://github.com/Niranjan70957)

## 🎯 Overview

This project provides a **quantitative comparison** of three approaches to LLM-based text classification:

1. **Zero-Shot Prompting** - Direct classification with no examples (BART-large)
2. **Few-Shot Prompting** - Classification with 3 in-context examples (FLAN-T5)
3. **SetFit Fine-Tuning** - Lightweight fine-tuning on 20 samples

**🔥 What Makes This Special:**
- ✅ **100% FREE** - No API keys, no signup, no costs
- ✅ **Runs Locally** - Full control and privacy
- ✅ **Open Source** - Uses HuggingFace Transformers
- ✅ **Production Ready** - Complete pipeline with visualizations

**Key Metrics Evaluated:**
- 📊 Accuracy & F1-Score
- ⚡ Inference Latency
- 💾 Model Size
- 🎯 Accuracy vs Latency Trade-offs

## 🚀 Quick Start (2 Steps!)

### Step 1: Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Everything
```bash
python run_all.py
```

**That's it!** First run will download models (~3GB total), then cached for future use.

**In 5-10 minutes** you'll have:
- ✅ Complete analysis results
- ✅ 4 professional visualizations
- ✅ Comprehensive report in `results/REPORT.md`

---

## 📁 Project Structure

```
prompt-vs-finetune-roi/
├── 📊 data/
│   └── financial_tweets.csv          # 96 labeled financial tweets
│
├── 🔬 src/
│   ├── method_a_zeroshot.py          # Zero-shot (BART-large-mnli)
│   ├── method_b_fewshot.py           # Few-shot (FLAN-T5-base)
│   ├── method_c_setfit.py            # SetFit (paraphrase-mpnet)
│   └── compare_methods.py            # ROI comparison & visualizations
│
├── 📈 results/ (generated when you run)
│   ├── REPORT.md                     # Comprehensive analysis
│   ├── comparison_table.csv          # Metrics comparison
│   ├── accuracy_comparison.png       # Accuracy & F1 chart
│   ├── latency_comparison.png        # Latency chart
│   ├── model_size_comparison.png     # Model size chart
│   └── roi_matrix.png                # Accuracy vs Latency matrix
│
├── 📓 notebooks/
│   └── exploration.ipynb             # Interactive analysis
│
├── requirements.txt                  # Python dependencies
├── run_all.py                        # One-click pipeline runner
└── README.md                         # This file
```

---

## 💡 Why This Project Matters

### Answers a Critical Question
*"Should I use prompt engineering or fine-tuning?"*

Every ML team faces this decision. This project provides **quantitative evidence** to make informed choices.

### No API Costs
Traditional approaches require expensive API keys:
- OpenAI GPT-4: ~$30-60 per 1M tokens
- Anthropic Claude: ~$15-75 per 1M tokens
- **This project: $0.00** 🎉

### Complete Control
- 🔒 **Privacy**: Data never leaves your machine
- 🎮 **Control**: Full customization of models
- 📦 **Offline**: Works without internet (after initial download)

---

## 🔬 Methodology

### Task: Financial Sentiment Analysis

**Dataset**: 96 financial tweets with sentiment labels (positive/negative/neutral)

**Test Set**: 30 samples (consistent across all methods for fair comparison)

### Method A: Zero-Shot Classification

**Model**: `facebook/bart-large-mnli` (~1.6GB)

**Approach**: HuggingFace zero-shot classification pipeline

**Prompt Template**:
```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    text,
    candidate_labels=["negative", "neutral", "positive"],
    hypothesis_template="This financial tweet has {} sentiment."
)
```

**Pros**:
- No training data required
- Instant deployment
- Good baseline accuracy

**Cons**:
- Lower accuracy than few-shot or fine-tuning
- Larger model size (~1.6GB)

### Method B: Few-Shot Prompting

**Model**: `google/flan-t5-base` (~850MB)

**Approach**: Text-generation with 3 examples in prompt

**Example Prompt**:
```
Classify the sentiment of financial tweets as positive, negative, or neutral.

Example 1:
Tweet: "$AAPL beating earnings expectations! Strong quarter ahead"
Sentiment: positive

Example 2:
Tweet: "$TSLA disappointing delivery numbers, below estimates"
Sentiment: negative

Example 3:
Tweet: "Federal Reserve maintains current interest rates"
Sentiment: neutral

Now classify this tweet:
Tweet: {text}
Sentiment:
```

**Pros**:
- Significantly better accuracy than zero-shot
- No training required
- Easy to update examples

**Cons**:
- Requires careful example selection
- Longer prompts (more processing time)

### Method C: SetFit Fine-Tuning

**Model**: `sentence-transformers/paraphrase-mpnet-base-v2` (~420MB)

**Approach**: Few-shot learning with contrastive training

**Training Config**:
```python
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
trainer = Trainer(model=model, train_dataset=train_dataset)
trainer.train()  # ~30 seconds on CPU
```

**Pros**:
- Best accuracy potential
- Smallest model size (~420MB)
- Fast inference

**Cons**:
- Requires training time
- Needs labeled training data
- Model deployment overhead

---

## 📊 Expected Results

> **Note**: Actual results may vary based on random sampling. Run the pipeline to see your specific results.

### Performance Comparison (Example)

| Method | Accuracy | F1-Score | Latency (ms) | Model Size | API Cost |
|--------|----------|----------|--------------|------------|----------|
| Zero-Shot | ~0.75 | ~0.74 | ~800ms | 1.6GB | FREE 🔓 |
| Few-Shot | ~0.80 | ~0.79 | ~600ms | 850MB | FREE 🔓 |
| SetFit | ~0.85 | ~0.84 | ~50ms | 420MB | FREE 🔓 |

### Key Findings

1. **Accuracy**: SetFit achieves highest accuracy (~85%) with minimal training
   - Few-shot improves ~5% over zero-shot

2. **Latency**: SetFit is **10-15x faster** than LLM-based methods
   - Critical for high-throughput applications

3. **Model Size**: SetFit has smallest footprint (420MB)
   - Easier to deploy and faster to load

4. **Cost**: **All methods are 100% FREE!** 💰
   - No API costs, ever!

---

## 🎨 Visualizations

The comparison script generates four key visualizations:

1. **Accuracy Comparison** (`accuracy_comparison.png`)
   - Side-by-side accuracy and F1-score bars
   - Shows relative performance across methods

2. **Latency Analysis** (`latency_comparison.png`)
   - Average inference latency comparison
   - Highlights speed differences

3. **Model Size** (`model_size_comparison.png`)
   - Model download size comparison
   - Shows resource requirements

4. **Decision Matrix** (`roi_matrix.png`)
   - Accuracy vs. Latency scatter plot
   - Helps choose method based on requirements

---


### What This Demonstrates

**Technical Skills:**
- ✅ ML Engineering & Model Evaluation
- ✅ HuggingFace Transformers & PyTorch
- ✅ NLP & Sentiment Analysis
- ✅ Data Analysis & Visualization

**Business Value:**
- ✅ Cost-conscious engineering (eliminating API costs)
- ✅ ROI analysis & decision-making
- ✅ Trade-off evaluation (accuracy vs latency vs size)

**Software Engineering:**
- ✅ Clean, modular code architecture
- ✅ Complete documentation
- ✅ Reproducible research
- ✅ Professional visualizations

---

## 🔧 Customization

### Use Different Models

**Zero-Shot**: Try other zero-shot classification models
```python
# In method_a_zeroshot.py
classifier = ZeroShotClassifier(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
```

**Few-Shot**: Try other text-generation models
```python
# In method_b_fewshot.py
classifier = FewShotClassifier(model_name="google/flan-t5-large")
```

**SetFit**: Try other sentence-transformers
```python
# In method_c_setfit.py
classifier = SetFitClassifier(model_name="sentence-transformers/all-mpnet-base-v2")
```

### Use Your Own Dataset

Replace `data/financial_tweets.csv` with your data:

**Required format:**
```csv
text,sentiment,label
"Your text here",positive,1
"Another sample",negative,0
"Neutral example",neutral,2
```

### Adjust Test Set Size

Edit any method script:
```python
# Change sample size from 30 to your desired number
test_df = df.sample(n=50, random_state=42)  # Line ~110
```

---

## 🎯 Business Recommendations

### When to Use Each Approach

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Quick prototype/MVP | **Few-Shot** | Good accuracy, no training |
| Best accuracy needed | **SetFit** | Highest accuracy after training |
| Smallest model size | **SetFit** | Only 420MB |
| Zero training data | **Zero-Shot** | Works without examples |
| High-throughput (>1000 req/sec) | **SetFit** | 10-15x faster inference |
| Resource-constrained | **SetFit** | Smallest memory footprint |

### Cost Comparison

**Traditional API-based approach (e.g., OpenAI):**
- Development: $50-200 (testing & prototyping)
- Production (10K predictions/day): ~$100-500/month
- **Annual cost**: $1,200 - $6,000

**This Open-Source Approach:**
- Development: $0
- Production: $0
- **Annual cost**: $0 🎉

**Savings**: $1,200 - $6,000 per year!

---

## 🛠️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 5GB free space (for models)
- **Internet**: Required for initial model download only

### Recommended Setup
- **RAM**: 8GB+ for faster processing
- **CPU**: Multi-core processor for parallel processing
- **GPU**: Optional (CUDA-enabled GPU speeds up inference)

---

## 📝 Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/Niranjan70957/prompt-vs-finetune-roi.git
cd prompt-vs-finetune-roi
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Pipeline
```bash
python run_all.py
```

**First run**: Models download automatically (~3GB)
**Future runs**: Models are cached, instant startup!

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

### Issue: Out of Memory
```python
# In method scripts, reduce batch size
args = TrainingArguments(batch_size=8)  # Was 16
```

### Issue: Slow Performance
```python
# Use GPU if available (automatic detection)
# Or reduce test set size
test_df = df.sample(n=10, random_state=42)  # Was 30
```

---

## 📚 Learn More

**HuggingFace Documentation:**
- [Transformers](https://huggingface.co/docs/transformers)
- [SetFit](https://huggingface.co/docs/setfit)
- [Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification)

**Related Projects:**
- [SetFit GitHub](https://github.com/huggingface/setfit)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Ideas for contributions:
- Add more classification methods (LoRA, Adapters, etc.)
- Expand to other tasks (NER, summarization, etc.)
- Add more datasets
- Improve visualizations

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Niranjan**

- GitHub: [@Niranjan70957](https://github.com/Niranjan70957)
- LinkedIn: [Your LinkedIn Profile]

---

## ⭐ Acknowledgments

- **HuggingFace** for amazing open-source models
- **SetFit** team for efficient few-shot learning
- **PyTorch** for deep learning framework

---

## 🎉 Why Choose This Project?

### 1. Zero Barriers to Entry
- No credit card required
- No API signup
- No usage limits

### 2. Learn Real ML Engineering
- Production-grade code
- Complete pipeline
- Best practices

### 3. Portfolio-Ready
- Professional documentation
- Beautiful visualizations
- Demonstrates multiple skills

### 4. Actually Useful
- Answers real business questions
- Provides actionable insights
- Can be adapted to your needs

---

**🚀 Get Started Now - It's 100% FREE!**

```bash
git clone https://github.com/Niranjan70957/prompt-vs-finetune-roi.git
cd prompt-vs-finetune-roi
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

---

⭐ **Star this repo** if you find it useful!
📧 **Questions?** Open an issue!
🔄 **Contribute** and make it better!

---

*Built with ❤️ using 100% free, open-source tools*
