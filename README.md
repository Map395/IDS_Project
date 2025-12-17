# Intrusion Detection System (IDS)

A machine learning-based Intrusion Detection System using the CICIDS2017 dataset (MachineLearningCVE) and Random Forest classifier to detect network intrusions and attacks.

## Overview

This project implements an IDS that analyzes network traffic data to identify various types of cyber attacks including:
- DDoS attacks
- Port scans
- Web attacks
- Infiltration attempts
- Benign traffic

The system uses a Random Forest classifier trained on the CICIDS2017 dataset, which contains labeled network traffic data collected over five days of network activity.

## Features

- **Multi-day Dataset Processing**: Loads and processes 8 CSV files from the CICIDS2017 dataset
- **Comprehensive Preprocessing**: Handles missing values, duplicates, and infinite values
- **Feature Engineering**: Automatic selection of numeric features and standardization
- **Random Forest Classification**: Robust ensemble learning with 100 estimators
- **Detailed Evaluation**: Provides accuracy metrics, classification reports, confusion matrices, and feature importance analysis
- **Multi-class Detection**: Identifies multiple attack types and benign traffic

## Dataset

The project uses the **CICIDS2017 dataset** (also known as MachineLearningCVE), which includes:

| File | Description | Size |
|------|-------------|------|
| Monday-WorkingHours.pcap_ISCX.csv | Benign traffic | ~177 MB |
| Tuesday-WorkingHours.pcap_ISCX.csv | Brute Force attacks | ~135 MB |
| Wednesday-workingHours.pcap_ISCX.csv | DoS/DDoS attacks | ~225 MB |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | Web attacks | ~52 MB |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | Infiltration | ~83 MB |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | Botnet attacks | ~58 MB |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | Port scans | ~77 MB |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | DDoS attacks | ~77 MB |

**Total Dataset Size**: ~884 MB

## Project Structure

```
IDS_Project/
├── Main.py                    # Main application script
├── requirements.txt           # Python dependencies
├── MachineLearningCVE/       # Dataset directory
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── README.md                  # This file
└── SETUP.md                   # Setup instructions
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure dataset is in place**:
   - All 8 CSV files should be in the `MachineLearningCVE/` directory

3. **Run the IDS**:
   ```bash
   python Main.py
   ```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## How It Works

### 1. Data Loading
The system loads all 8 CSV files from the MachineLearningCVE directory and combines them into a single dataset.

### 2. Preprocessing
- Removes duplicate records
- Handles missing values (drops columns with >50% missing, fills rest with median)
- Replaces infinite values with median
- Encodes categorical labels
- Selects only numeric features
- Standardizes features using StandardScaler

### 3. Model Training
- Splits data into 80% training and 20% testing
- Trains a Random Forest classifier with:
  - 100 decision trees
  - Maximum depth of 20
  - Parallel processing enabled

### 4. Evaluation
Provides comprehensive metrics including:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix
- Top 10 most important features

## Model Configuration

The Random Forest classifier is configured with the following parameters:

```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    random_state=42,       # For reproducibility
    n_jobs=-1,             # Use all CPU cores
    verbose=1              # Show training progress
)
```

## Output

The system provides detailed output including:

1. **Dataset Loading Progress**: Shows which files are being loaded and record counts
2. **Preprocessing Statistics**: Missing values, duplicates removed, feature counts
3. **Label Distribution**: Shows the distribution of attack types in the dataset
4. **Training Progress**: Real-time updates during model training
5. **Evaluation Metrics**: 
   - Accuracy score
   - Classification report (precision, recall, F1-score for each class)
   - Confusion matrix
   - Feature importance rankings

## Performance

Expected performance metrics (may vary based on dataset):
- **Accuracy**: Typically 95%+ on the test set
- **Training Time**: 5-15 minutes depending on hardware
- **Memory Usage**: ~4-8 GB RAM during training

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure all CSV files are in the `MachineLearningCVE/` directory
2. **MemoryError**: Reduce dataset size or increase available RAM
3. **Encoding errors**: The script handles UTF-8 encoding automatically for Windows

### Performance Optimization

- **Reduce n_estimators**: Lower from 100 to 50 for faster training
- **Limit max_depth**: Reduce from 20 to 10 for faster training
- **Sample dataset**: Use a subset of the data for quick testing

## Future Enhancements

- [ ] Add model persistence (save/load trained models)
- [ ] Implement real-time network traffic analysis
- [ ] Add support for additional ML algorithms (SVM, Neural Networks)
- [ ] Create a web-based dashboard for visualization
- [ ] Implement cross-validation for better model evaluation
- [ ] Add hyperparameter tuning

## License

This project is for educational purposes. Please ensure you have the rights to use the CICIDS2017 dataset.

## References

- **CICIDS2017 Dataset**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Random Forest**: Scikit-learn documentation
- **Intrusion Detection**: Network security and machine learning research

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Contact

For questions or suggestions, please open an issue in the repository.
