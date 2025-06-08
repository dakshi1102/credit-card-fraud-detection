import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Apply SMOTE to balance the dataset and split it into training and test sets.
    Assumes the target column is 'Class'.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    """Train and save Random Forest and XGBoost models."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    joblib.dump(rf, 'rf_model.pkl')
    joblib.dump(xgb, 'xgb_model.pkl')

    return rf, xgb

def evaluate_model(model, X_test, y_test):
    """Generate a classification report from the test data."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)
