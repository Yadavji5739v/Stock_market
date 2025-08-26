# Windows Setup Guide for Stock Prediction App

## Problem
You're experiencing a TensorFlow DLL loading error on Windows:
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal: A dynamic link library (DLL) initialization routine failed.
```

## Solutions

### Option 1: Use Windows-Compatible Requirements (Recommended)

1. **Uninstall current TensorFlow:**
   ```bash
   pip uninstall tensorflow
   ```

2. **Install Windows-compatible packages:**
   ```bash
   pip install -r requirements_windows.txt
   ```

3. **Test your setup:**
   ```bash
   python test_setup.py
   ```

### Option 2: Fix TensorFlow Installation

If you want to keep TensorFlow, try these steps:

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install Microsoft Visual C++ Redistributable:**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart your computer

3. **Install TensorFlow CPU-only version:**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu==2.13.0
   ```

4. **Alternative: Use PyTorch instead:**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2
   ```

### Option 3: Use Conda Environment (Most Reliable)

1. **Install Miniconda:**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Install with default settings

2. **Create new environment:**
   ```bash
   conda create -n stock_prediction python=3.9
   conda activate stock_prediction
   ```

3. **Install packages:**
   ```bash
   conda install pandas numpy scikit-learn matplotlib seaborn
   conda install -c conda-forge streamlit yfinance plotly
   conda install -c conda-forge xgboost lightgbm
   conda install tensorflow-cpu
   ```

## What I've Fixed

1. **Updated `models.py`** with error handling for TensorFlow imports
2. **Created `requirements_windows.txt`** with Windows-compatible packages
3. **Added `test_setup.py`** to verify your installation works
4. **Made LSTM models optional** - app works without TensorFlow

## Current Status

- ✅ **Linear Regression** - Works without TensorFlow
- ✅ **Random Forest** - Works without TensorFlow  
- ⚠️ **LSTM Models** - Disabled if TensorFlow unavailable
- ✅ **All other features** - Fully functional

## Testing Your Setup

Run the test script to verify everything works:
```bash
python test_setup.py
```

## Running Your App

After successful setup:
```bash
python app.py
```

The app will automatically detect TensorFlow availability and adjust functionality accordingly.

## Troubleshooting

### If you still get errors:

1. **Check Python version:** Use Python 3.9 or 3.10 (most stable with ML libraries)
2. **Use virtual environment:** Isolate dependencies
3. **Try CPU-only versions:** Often more stable on Windows
4. **Check antivirus:** Some antivirus software blocks DLL loading

### Common Windows-specific issues:

- **Missing Visual C++ Redistributable**
- **Antivirus blocking DLLs**
- **Python path issues**
- **32-bit vs 64-bit mismatch**

## Need Help?

If you continue to have issues:
1. Run `python test_setup.py` and share the output
2. Check your Python version: `python --version`
3. Verify pip is working: `pip --version`
4. Try the conda approach for most reliable results

## Alternative: Cloud-Based Solution

If Windows continues to cause issues, consider:
- **Google Colab** (free, includes TensorFlow)
- **Kaggle Notebooks** (free, includes TensorFlow)
- **AWS SageMaker** (paid, enterprise-grade)
