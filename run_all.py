"""
Run All Methods - Complete Pipeline
Executes all three methods sequentially and generates comparison report.
100% FREE - No API Keys Required!
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {str(e)}")
        return False


def main():
    """Execute all methods and generate comparison."""
    print("="*70)
    print("PROMPT ENGINEERING vs. FINE-TUNING ROI ANALYSIS")
    print("🔓 100% FREE - No API Keys Required!")
    print("All models run locally using HuggingFace Transformers")
    print("="*70)

    print("\n📝 First-time setup:")
    print("  - Models will download automatically (~3GB total)")
    print("  - Models are cached for future runs")
    print("  - No signup or API keys needed!")

    response = input("\nReady to start? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Run this script again when ready!")
        return

    scripts = [
        ("src/method_a_zeroshot.py", "Method A: Zero-Shot Classification"),
        ("src/method_b_fewshot.py", "Method B: Few-Shot Prompting"),
        ("src/method_c_setfit.py", "Method C: SetFit Fine-Tuning"),
        ("src/compare_methods.py", "ROI Comparison & Visualization")
    ]

    results = []
    for script_path, description in scripts:
        success = run_script(script_path, description)
        results.append((description, success))

    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)

    all_success = True
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
        if not success:
            all_success = False

    if all_success:
        print("\n🎉 All methods completed successfully!")
        print("\nResults are available in the 'results/' directory:")
        print("  - results/REPORT.md - Comprehensive analysis report")
        print("  - results/*.png - Visualization charts")
        print("  - results/*.json - Detailed results for each method")
        print("\n💰 Total API Cost: $0.00 (All models are FREE!)")
    else:
        print("\n⚠ Some methods failed. Check the output above for details.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
