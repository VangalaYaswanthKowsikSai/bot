# Detailed Setup Guide

This guide provides step-by-step instructions for setting up and running the Prompt Engineering vs. Fine-Tuning ROI analysis.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [API Key Setup](#api-key-setup)
4. [Running the Project](#running-the-project)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for SetFit)
- **Disk Space**: 2GB free space
- **Internet**: Required for API calls (Zero-Shot and Few-Shot methods)

### Recommended Setup
- **RAM**: 8GB+ for faster SetFit training
- **CPU**: Multi-core processor for parallel processing
- **GPU**: Optional (CUDA-enabled GPU will speed up SetFit training)

## Installation Steps

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer and **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### Step 2: Clone or Download the Repository

**Option A: Using Git**
```bash
git clone https://github.com/Niranjan70957/prompt-vs-finetune-roi.git
cd prompt-vs-finetune-roi
```

**Option B: Download ZIP**
1. Download the project ZIP file
2. Extract to your desired location
3. Open terminal/command prompt in the extracted folder

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time:** 2-5 minutes depending on internet speed.

**Common installation issues:**
- If you get SSL errors, try: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`
- If you get permission errors on Linux/Mac, use: `pip install --user -r requirements.txt`

## API Key Setup

### Groq API Key (Required for Zero-Shot and Few-Shot)

1. **Create Groq Account**
   - Visit: https://console.groq.com
   - Sign up for a free account

2. **Generate API Key**
   - Go to: https://console.groq.com/keys
   - Click "Create API Key"
   - Copy the key (you won't see it again!)

3. **Add Key to Project**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env file and add your key
   # Windows: notepad .env
   # macOS/Linux: nano .env
   ```

   In the `.env` file, replace:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   with your actual key:
   ```
   GROQ_API_KEY=gsk_abc123xyz...
   ```

### Optional: Other API Keys

**OpenAI (for testing with GPT models):**
```bash
OPENAI_API_KEY=sk-...
```

**Anthropic (for testing with Claude):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

## Running the Project

### Quick Start: Run Everything

```bash
python run_all.py
```

This will:
1. Run Zero-Shot classification
2. Run Few-Shot prompting
3. Run SetFit fine-tuning
4. Generate comparison report and visualizations

**Expected runtime:** 5-10 minutes (depending on API response times)

### Run Individual Methods

**Method A: Zero-Shot**
```bash
python src/method_a_zeroshot.py
```
Output: `results/zeroshot_results.json`

**Method B: Few-Shot**
```bash
python src/method_b_fewshot.py
```
Output: `results/fewshot_results.json`

**Method C: SetFit**
```bash
python src/method_c_setfit.py
```
Output: `results/setfit_results.json` and `results/setfit_model/`

**Generate Comparison**
```bash
python src/compare_methods.py
```
Output: `results/REPORT.md` and visualization PNGs

### View Results

**Comprehensive Report:**
```bash
# Windows
notepad results/REPORT.md

# macOS
open results/REPORT.md

# Linux
cat results/REPORT.md
```

**Visualizations:**
Navigate to the `results/` folder and open the PNG files:
- `accuracy_comparison.png`
- `latency_comparison.png`
- `cost_comparison.png`
- `roi_matrix.png`

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Error:**
```
ModuleNotFoundError: No module named 'groq'
```

**Solution:**
```bash
# Ensure virtual environment is activated
# You should see (venv) in your prompt

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "GROQ_API_KEY not found"

**Error:**
```
Error: GROQ_API_KEY environment variable not set
```

**Solution:**
1. Check that `.env` file exists in project root
2. Verify the file contains: `GROQ_API_KEY=your_actual_key`
3. Restart your terminal/IDE to reload environment variables

### Issue: API Rate Limiting

**Error:**
```
RateLimitError: You have exceeded your rate limit
```

**Solution:**
- Groq free tier has rate limits
- Wait a few minutes between runs
- Consider reducing test set size in scripts (change `n=30` to `n=10`)

### Issue: Slow SetFit Training

**Problem:** SetFit training takes too long

**Solution:**
1. Reduce epochs in `method_c_setfit.py`:
   ```python
   classifier.train(train_df, num_epochs=1)  # Was 3
   ```

2. Reduce training samples:
   ```python
   train_df = remaining_df.sample(n=10, random_state=42)  # Was 20
   ```

3. Use a smaller model:
   ```python
   SetFitClassifier("sentence-transformers/all-MiniLM-L6-v2")
   ```

### Issue: Import Errors on Windows

**Error:**
```
ImportError: DLL load failed
```

**Solution:**
1. Install Microsoft Visual C++ Redistributable:
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Reinstall PyTorch:
   ```bash
   pip uninstall torch
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Advanced Configuration

### Customize Test Set Size

Edit any method script:
```python
# Change sample size from 30 to your desired number
test_df = df.sample(n=30, random_state=42)  # Line ~60
```

### Use Different Models

**For Zero-Shot/Few-Shot:**
Edit `method_a_zeroshot.py` or `method_b_fewshot.py`:
```python
# Available models on Groq:
# - llama3-8b-8192 (default)
# - llama3-70b-8192
# - mixtral-8x7b-32768
# - gemma-7b-it

classifier = ZeroShotClassifier(model_name="mixtral-8x7b-32768")
```

**For SetFit:**
Edit `method_c_setfit.py`:
```python
# Try different sentence-transformers:
# - sentence-transformers/all-MiniLM-L6-v2 (faster, smaller)
# - sentence-transformers/all-mpnet-base-v2 (balanced)
# - sentence-transformers/paraphrase-mpnet-base-v2 (default)

classifier = SetFitClassifier("sentence-transformers/all-MiniLM-L6-v2")
```

### Adjust Training Parameters

In `method_c_setfit.py`:
```python
args = TrainingArguments(
    batch_size=16,        # Increase for faster training (needs more RAM)
    num_epochs=1,         # More epochs = better accuracy, longer training
    evaluation_strategy="no",
    save_strategy="no",
    logging_steps=10,
)
```

### Use Your Own Dataset

Replace `data/financial_tweets.csv` with your data.

**Required format:**
```csv
text,sentiment,label
"Your text here",positive,1
"Another sample",negative,0
"Neutral example",neutral,2
```

**Requirements:**
- `text`: String column with your input text
- `label`: Numeric labels (0, 1, 2, etc.)
- `sentiment`: Optional, human-readable labels

## Performance Optimization

### For Faster API Calls
- Use Groq (fastest inference)
- Reduce test set size during development
- Use caching for repeated experiments

### For Faster SetFit Training
- Use GPU if available (automatically detected)
- Reduce training samples
- Use smaller sentence-transformer model
- Reduce epochs to 1

### For Lower Costs
- Use Zero-Shot for simple tasks
- Cache API responses during development
- Use SetFit for production (free inference)

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: https://github.com/Niranjan70957/prompt-vs-finetune-roi/issues
2. **Create new issue**: Include error message, OS, Python version
3. **Email support**: [your-email@example.com]

## Next Steps

After successful setup:

1. ✅ Run the complete pipeline: `python run_all.py`
2. ✅ Review the results in `results/REPORT.md`
3. ✅ Explore visualizations in `results/*.png`
4. ✅ Experiment with different models and parameters
5. ✅ Try your own dataset

---

**Happy Experimenting!** 🚀
