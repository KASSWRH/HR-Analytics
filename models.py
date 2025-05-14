import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, model_type="XGBoost"):
    """
    Train a machine learning model for turnover prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
    
    Returns:
        Trained model
    """
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Various metrics for model evaluation
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, auc, conf_matrix

def predict_turnover(data, model, preprocessor, feature_names):
    """
    Generate turnover predictions for the given data.
    
    Args:
        data: DataFrame with employee data
        model: Trained prediction model
        preprocessor: Fitted data preprocessor
        feature_names: Feature names used during training
    
    Returns:
        DataFrame with original data and predictions
    """
    # Create a copy of the data
    df = data.copy()
    
    # Get employee IDs
    employee_ids = df['Employee_ID']
    
    # Prepare data for prediction
    X = preprocessor.transform(df.drop('Resigned', axis=1))
    
    # Make predictions
    turnover_proba = model.predict_proba(X)[:, 1]
    
    # Create output DataFrame
    predictions = df.copy()
    predictions['Turnover_Probability'] = turnover_proba
    
    return predictions
