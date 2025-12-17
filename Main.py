import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import sys
warnings.filterwarnings('ignore')

# Set encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For older python versions or weird environments
        pass

# Configuration
DATA_DIR = 'MachineLearningCVE'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_dataset():
    """Load all CSV files from the MachineLearningCVE directory"""
    print("Loading MachineLearningCVE dataset...")
    
    # List all CSV files in the directory
    csv_files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]
    
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            print(f"Loading {file}...")
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            dataframes.append(df)
            print(f"  Loaded {len(df)} records")
        else:
            print(f"Warning: {file} not found!")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Total features: {len(combined_df.columns)}")
    
    return combined_df

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "="*50)
    print("Preprocessing data...")
    print("="*50)
    
    # Display initial info
    print(f"\nInitial shape: {df.shape}")
    print(f"\nColumn names:")
    print(df.columns.tolist())
    
    # Check for label column (common names in CICIDS2017)
    label_column = None
    possible_labels = ['Label', 'label', ' Label', 'Attack', 'Class']
    for col in possible_labels:
        if col in df.columns:
            label_column = col
            break
    
    if label_column is None:
        # Use the last column as label
        label_column = df.columns[-1]
        print(f"\nUsing last column as label: {label_column}")
    else:
        print(f"\nFound label column: {label_column}")
    
    # Display label distribution
    print(f"\nLabel distribution:")
    print(df[label_column].value_counts())
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        # Drop columns with too many missing values (>50%)
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)
        # Fill remaining missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    else:
        print("No missing values found")
    
    # Handle infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nEncoded labels:")
    for i, label in enumerate(le.classes_):
        print(f"  {i}: {label}")
    
    # Select only numeric features
    X_numeric = X.select_dtypes(include=[np.number])
    
    print(f"\nFeatures after preprocessing: {X_numeric.shape[1]}")
    print(f"Samples: {X_numeric.shape[0]}")
    
    return X_numeric, y_encoded, le

def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    
    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("\nModel training completed!")
    
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the trained model"""
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_,
                                zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("="*50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    return accuracy, y_pred

def main():
    """Main execution function"""
    print("="*50)
    print("INTRUSION DETECTION SYSTEM")
    print("Using MachineLearningCVE Dataset")
    print("="*50)
    
    # Load dataset
    df = load_dataset()
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Split data
    print("\n" + "="*50)
    print("Splitting data into train and test sets...")
    print("="*50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    accuracy, predictions = evaluate_model(model, X_test_scaled, y_test, label_encoder)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Model Accuracy: {accuracy*100:.2f}%")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, label_encoder = main()