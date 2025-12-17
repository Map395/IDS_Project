# Setup Guide - Intrusion Detection System

This guide provides detailed instructions for setting up and running the IDS project on your system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.7 or higher
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: At least 2 GB free space
- **CPU**: Multi-core processor recommended for faster training

### Software Requirements

- Python 3.7+
- pip (Python package manager)
- Git (optional, for version control)

## Installation

### Step 1: Verify Python Installation

Open a terminal/command prompt and check your Python version:

```bash
python --version
```

or

```bash
python3 --version
```

You should see Python 3.7 or higher. If not, download and install Python from [python.org](https://www.python.org/downloads/).

### Step 2: Clone or Download the Project

If using Git:

```bash
git clone <repository-url>
cd IDS_Project
```

Or download and extract the project ZIP file, then navigate to the project directory.

### Step 3: Create a Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### Step 4: Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install -r requirements.txt
```

This will install:
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations
- **scikit-learn**: For machine learning algorithms and tools

### Step 5: Verify Installation

Check that all packages are installed correctly:

```bash
pip list
```

You should see pandas, numpy, and scikit-learn in the list.

## Dataset Setup

### Option 1: Dataset Already Included

If the `MachineLearningCVE/` directory already contains all 8 CSV files, you can skip to [Running the Project](#running-the-project).

Verify the dataset files:

```bash
# On Windows
dir MachineLearningCVE

# On macOS/Linux
ls -lh MachineLearningCVE/
```

You should see:
- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

### Option 2: Download the Dataset

If the dataset is not included:

1. **Download CICIDS2017 Dataset**:
   - Visit: https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the CSV files (MachineLearningCVE version)

2. **Create the Dataset Directory**:
   ```bash
   mkdir MachineLearningCVE
   ```

3. **Extract and Place Files**:
   - Extract all downloaded CSV files
   - Move them to the `MachineLearningCVE/` directory
   - Ensure file names match exactly as listed above

4. **Verify File Sizes**:
   The total dataset should be approximately 884 MB.

## Running the Project

### Basic Execution

With the virtual environment activated and dataset in place:

```bash
python Main.py
```

### Expected Output

The script will display:

1. **Loading Phase**:
   ```
   ==================================================
   INTRUSION DETECTION SYSTEM
   Using MachineLearningCVE Dataset
   ==================================================
   Loading MachineLearningCVE dataset...
   Loading Monday-WorkingHours.pcap_ISCX.csv...
     Loaded XXXXX records
   ...
   ```

2. **Preprocessing Phase**:
   - Column information
   - Label distribution
   - Duplicate removal statistics
   - Missing value handling

3. **Training Phase**:
   - Data split information
   - Feature scaling
   - Random Forest training progress

4. **Evaluation Phase**:
   - Accuracy score
   - Classification report
   - Confusion matrix
   - Feature importance

### Execution Time

- **Loading**: 1-3 minutes
- **Preprocessing**: 2-5 minutes
- **Training**: 5-15 minutes (depends on CPU)
- **Evaluation**: 1-2 minutes

**Total**: Approximately 10-25 minutes for complete execution

## Verification

### Check for Successful Completion

The script should end with:

```
==================================================
TRAINING COMPLETED SUCCESSFULLY!
==================================================
Final Model Accuracy: XX.XX%
```

### Verify Output Quality

1. **Accuracy**: Should be above 90% (typically 95%+)
2. **No Errors**: No Python exceptions or errors
3. **All Files Loaded**: All 8 CSV files should be loaded successfully
4. **Label Distribution**: Should show multiple attack types

## Troubleshooting

### Issue: "No module named 'pandas'" (or numpy, sklearn)

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: MachineLearningCVE/..."

**Solution**: 
1. Verify the `MachineLearningCVE/` directory exists
2. Check that all CSV files are present
3. Verify file names match exactly (case-sensitive)

### Issue: "MemoryError" or System Freezes

**Solution**: 
1. Close other applications to free RAM
2. Reduce dataset size by commenting out some files in `Main.py` (lines 30-38)
3. Reduce `n_estimators` in the RandomForestClassifier (line 141)

Example modification:
```python
# In Main.py, line 141
model = RandomForestClassifier(
    n_estimators=50,  # Reduced from 100
    max_depth=15,     # Reduced from 20
    ...
)
```

### Issue: Encoding Errors on Windows

**Solution**: The script handles this automatically. If issues persist:
1. Ensure you're using Python 3.7+
2. Run in PowerShell instead of CMD
3. Check that CSV files are UTF-8 encoded

### Issue: Very Slow Training

**Solution**:
1. **Use fewer trees**: Reduce `n_estimators` to 50
2. **Limit tree depth**: Reduce `max_depth` to 10
3. **Sample the data**: Add sampling in the preprocessing step

Example sampling:
```python
# After line 53 in Main.py
combined_df = combined_df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

### Issue: Low Accuracy (<80%)

**Possible Causes**:
1. Dataset corruption or incomplete files
2. Incorrect preprocessing
3. Data imbalance

**Solution**:
1. Re-download the dataset
2. Check label distribution in output
3. Verify all 8 files loaded successfully

## Advanced Configuration

### Modify Model Parameters

Edit `Main.py` to customize the Random Forest classifier:

```python
# Line 140-146
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees (50-200)
    max_depth=20,            # Tree depth (10-30)
    min_samples_split=2,     # Min samples to split (2-10)
    min_samples_leaf=1,      # Min samples per leaf (1-5)
    random_state=RANDOM_STATE,
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)
```

### Adjust Train/Test Split

Modify the test size ratio:

```python
# Line 23
TEST_SIZE = 0.2  # Change to 0.3 for 70/30 split
```

### Change Random State

For different data splits:

```python
# Line 22
RANDOM_STATE = 42  # Change to any integer
```

### Save the Trained Model

Add model persistence (after line 237):

```python
import joblib

# Save model
joblib.dump(model, 'ids_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Load model later
# model = joblib.load('ids_model.pkl')
```

### Use a Subset of Files

For quick testing, comment out files in the `csv_files` list (lines 30-38):

```python
csv_files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    # 'Wednesday-workingHours.pcap_ISCX.csv',  # Commented out
    # ... comment out other files
]
```

## Running in Different Environments

### Jupyter Notebook

To run in Jupyter:

1. Install Jupyter:
   ```bash
   pip install jupyter
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Create a new notebook and copy code from `Main.py`

### Google Colab

1. Upload `Main.py` and dataset to Google Drive
2. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update `DATA_DIR` path to point to your Drive location

### Docker (Advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "Main.py"]
```

Build and run:
```bash
docker build -t ids-project .
docker run ids-project
```

## Next Steps

After successful setup:

1. **Experiment with parameters**: Try different model configurations
2. **Analyze results**: Study the classification report and feature importance
3. **Extend functionality**: Add model saving, visualization, or real-time detection
4. **Optimize performance**: Profile code and optimize bottlenecks

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all prerequisites are met
3. Ensure dataset files are complete and uncorrupted
4. Review the [README.md](README.md) for additional context
5. Search for similar issues online
6. Open an issue in the repository

## Additional Resources

- **Python Documentation**: https://docs.python.org/3/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/
- **CICIDS2017 Dataset Info**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Random Forest Guide**: https://scikit-learn.org/stable/modules/ensemble.html#forest

---

**Happy Intrusion Detection!** 🛡️
