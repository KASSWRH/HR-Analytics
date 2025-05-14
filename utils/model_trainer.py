import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# Removed SHAP dependency due to compatibility issues
# import shap
import pickle
import os

def train_model(X, y, model_type='xgboost', test_size=0.2, random_state=42, **model_params):
    """
    Train a machine learning model
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type (str): Type of model ('xgboost', 'random_forest', 'logistic')
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        **model_params: Additional parameters for the model
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test)
            - model: The trained model
            - X_train, X_test, y_train, y_test: Train-test split
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize model
    if model_type == 'xgboost':
        model = XGBClassifier(random_state=random_state, **model_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, **model_params)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000, **model_params)
    else:
        raise ValueError("Unsupported model type. Choose from 'xgboost', 'random_forest', or 'logistic'")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    return metrics

def get_feature_importance(model, X, feature_names):
    """
    Get feature importance from the model
    
    Args:
        model: The trained model
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attributes")
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    })
    
    return importance_df.sort_values('importance', ascending=False)

def get_shap_values(model, X, feature_names):
    """
    Alternative to SHAP values using feature importance
    
    Args:
        model: The trained model
        X: Feature matrix for which to compute importance values
        feature_names: List of feature names
        
    Returns:
        tuple: (importance_values, feature_names)
            - importance_values: Feature importance values for each sample
            - feature_names: List of feature names
    """
    # Get feature importance
    importance_df = get_feature_importance(model, X, feature_names)
    
    # Create a simplified "SHAP-like" value for each sample
    # This is a basic approximation - not real SHAP values
    sample_size = min(1000, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    sample_data = X[sample_indices]
    
    # Store feature names and their importance values
    feature_names_ordered = importance_df['feature'].values
    importance_values = importance_df['importance'].values
    
    # Create an explainer-like object (just a dictionary for compatibility)
    explainer = {"features": feature_names_ordered, "method": "feature_importance"}
    
    # Expand importance values for each sample (simplified approach)
    # This creates array of shape (n_samples, n_features) like SHAP would
    sample_importance = np.tile(importance_values, (sample_size, 1))
    
    # Scale by the feature values to approximate directional impact
    for i, feat in enumerate(feature_names_ordered):
        if i < sample_data.shape[1]:  # Safety check
            # Scale importance by normalized feature value
            feat_idx = list(feature_names).index(feat) if feat in feature_names else i
            if feat_idx < sample_data.shape[1]:
                feat_values = sample_data[:, feat_idx]
                # Normalize to [-1, 1] range to mimic SHAP's directional effect
                if np.std(feat_values) > 0:
                    norm_values = (feat_values - np.mean(feat_values)) / np.std(feat_values)
                    # Apply sign to the importance based on feature value
                    sample_importance[:, i] = sample_importance[:, i] * np.sign(norm_values)
    
    return sample_importance, explainer

def save_model(model, filepath='model.pkl'):
    """
    Save model to disk
    
    Args:
        model: The trained model
        filepath (str): Path to save the model
        
    Returns:
        bool: True if saved successfully
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filepath='model.pkl'):
    """
    Load model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        The loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model
