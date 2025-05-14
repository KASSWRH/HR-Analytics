import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
import re

def calculate_years_at_company(hire_date):
    """
    Calculate years at company based on hire date.
    
    Args:
        hire_date: Hire date string
    
    Returns:
        Years at company as float
    """
    try:
        # Convert string to datetime
        if isinstance(hire_date, str):
            hire_date = pd.to_datetime(hire_date)
        
        # Calculate years at company
        today = datetime.now()
        years = (today - hire_date).days / 365.25
        
        return round(years, 1)
    except:
        # Return NaN if calculation fails
        return np.nan

def preprocess_data(df, target_column, id_column):
    """
    Preprocess the input data for machine learning model.
    
    Args:
        df: Pandas DataFrame with the input data
        target_column: Name of the target column
        id_column: Name of the ID column
    
    Returns:
        X: Features matrix
        y: Target vector
        preprocessor: Fitted preprocessor object
        feature_names: List of feature names after preprocessing
    """
    # Make a copy to avoid modifying the original data
    data = df.copy()
    
    # Create features and target
    y = data[target_column].astype(int)
    
    # Drop unnecessary columns for modeling
    features = data.drop([target_column, id_column], axis=1)
    
    # Calculate years at company if not present and hire date is available
    if 'Years_At_Company' not in features.columns and 'Hire_Date' in features.columns:
        features['Years_At_Company'] = features['Hire_Date'].apply(
            lambda x: calculate_years_at_company(x)
        )
    
    # Identify column types
    categorical_cols = [col for col in features.columns if 
                        features[col].dtype == 'object' or
                        features[col].nunique() < 10]
    
    numerical_cols = [col for col in features.columns if 
                     features[col].dtype in ['int64', 'float64'] and 
                     col not in categorical_cols]
    
    # Define preprocessing for categorical and numerical features
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Fit and transform the data
    X = preprocessor.fit_transform(features)
    
    # Get feature names
    onehot_cols = []
    if categorical_cols:
        onehot_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    
    feature_names = numerical_cols + list(onehot_cols)
    
    return X, y, preprocessor, feature_names

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X: Features matrix
        y: Target vector
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def feature_importance(model, feature_names, model_type):
    """
    Extract feature importance from the trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model (XGBoost, Random Forest, etc.)
    
    Returns:
        DataFrame with feature names and importance scores
    """
    if model_type == "XGBoost":
        importance_scores = model.feature_importances_
    elif model_type == "Random Forest":
        importance_scores = model.feature_importances_
    elif model_type == "Logistic Regression":
        importance_scores = np.abs(model.coef_[0])
    else:
        # Default fallback
        importance_scores = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(len(feature_names))
    
    # Create a DataFrame of feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return feature_importance_df
