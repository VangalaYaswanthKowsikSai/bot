# 🚀 Quick Start Guide

> **🔓 100% FREE - No API Keys Required!**

## Installation (2 Minutes)

### 1. Install Python Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python run_all.py
```

That's it! **No API keys, no signup, nothing else needed.**

---

## What Happens on First Run?

### Models Download Automatically (~3GB)
- **Zero-Shot**: BART-large-mnli (~1.6GB)
- **Few-Shot**: FLAN-T5-base (~850MB)
- **SetFit**: paraphrase-mpnet (~420MB)

**Future runs**: Models are cached, instant startup!

---

## Expected Output

After 5-10 minutes, you'll have:

### 1. Results Directory
```
results/
├── REPORT.md                     ← Comprehensive analysis
├── comparison_table.csv          ← Metrics comparison
├── accuracy_comparison.png       ← Accuracy chart
├── latency_comparison.png        ← Speed chart
├── model_size_comparison.png     ← Size chart
└── roi_matrix.png                ← Decision matrix
```

### 2. Key Findings
- **Zero-Shot**: ~75% accuracy, 800ms latency, 1.6GB
- **Few-Shot**: ~80% accuracy, 600ms latency, 850MB
- **SetFit**: ~85% accuracy, 50ms latency, 420MB

### 3. Main Insight
**All methods are 100% FREE!** Choose based on accuracy vs latency trade-offs.

---

## Run Individual Methods

```bash
# Zero-Shot only
python src/method_a_zeroshot.py

# Few-Shot only
python src/method_b_fewshot.py

# SetFit only
python src/method_c_setfit.py

# Generate comparison
python src/compare_methods.py
```

---

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 5GB disk space

**Recommended:**
- 8GB RAM
- GPU optional (faster, but not required)

---

## Troubleshooting

### Models downloading too slow?
```python
# Use smaller models (edit the scripts)
# Zero-Shot: Use DeBERTa-v3-base-mnli instead
# Few-Shot: Use flan-t5-small instead
```

### Out of memory?
```python
# Reduce test set size in scripts
test_df = df.sample(n=10, random_state=42)  # Was 30
```

### Want faster results?
```python
# Use GPU (automatic if available)
# Or just run SetFit method (fastest)
python src/method_c_setfit.py
```

---

## Next Steps

1. ✅ Review `results/REPORT.md`
2. ✅ Check out the visualizations
3. ✅ Customize for your own dataset
4. ✅ Push to GitHub for your portfolio!

---

## Why This Matters

### Traditional Approach (API-based)
- Sign up for API keys
- Pay per request
- $100-500/month for production
- Data sent to external servers

### This Approach (Open-Source)
- **No signup** required
- **$0 costs** forever
- **Full privacy** (runs locally)
- **Complete control** over models

**Annual savings: $1,200 - $6,000!**

---

## Questions?

- 📖 Full documentation: See [README.md](README.md)
- 🛠️ Detailed setup: See [SETUP.md](SETUP.md)
- 🐛 Issues: Check [README.md#troubleshooting](README.md#troubleshooting)

---

**Made it this far?** You're ready to run! 🎉

```bash
python run_all.py
```

---

*100% free, 100% open-source, 100% yours!* 🔓
