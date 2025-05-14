import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file):
    """
    Load data from a CSV or Excel file
    
    Args:
        file: The uploaded file object
        
    Returns:
        pd.DataFrame: The loaded dataframe
    """
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

def preprocess_data(df, target_column='Resigned'):
    """
    Preprocess the data for model training
    
    Args:
        df (pd.DataFrame): The input dataframe
        target_column (str): The name of the target column
        
    Returns:
        tuple: (X, y, preprocessor, feature_names)
            - X: The preprocessed feature matrix
            - y: The target variable
            - preprocessor: The preprocessing pipeline
            - feature_names: List of feature names after preprocessing
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Extract target variable
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data")
    
    y = data[target_column].astype(int)
    data = data.drop(columns=[target_column])
    
    # Handle date columns by converting to datetime and extracting useful features
    date_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    date_columns += [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    for col in date_columns:
        if col in data.columns:
            try:
                # Convert to datetime if not already
                if data[col].dtype != 'datetime64[ns]':
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                
                # Calculate relative time (in days) from the present
                data[f'{col}_days'] = (datetime.now() - data[col]).dt.days
                
                # Extract useful components
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_year'] = data[col].dt.year
                
                # Drop the original date column
                data = data.drop(columns=[col])
            except:
                # Skip if conversion fails
                pass
    
    # Identify column types
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='drop'  # Drop columns that are not specified
    )
    
    # Fit and transform the data
    X = preprocessor.fit_transform(data)
    
    # Get feature names
    onehot_columns = []
    if categorical_columns:
        categorical_transformer_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
        onehot_encoder = preprocessor.transformers_[categorical_transformer_idx][1].named_steps['onehot']
        onehot_categories = onehot_encoder.categories_
        for i, cats in enumerate(onehot_categories):
            for cat in cats:
                onehot_columns.append(f"{categorical_columns[i]}_{cat}")
    
    feature_names = numerical_columns + onehot_columns
    
    return X, y, preprocessor, feature_names

def calculate_data_statistics(df):
    """
    Calculate basic statistics for the DataFrame
    
    Args:
        df (pd.DataFrame): The dataframe
        
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {}
    
    # Basic statistics
    stats['rows'] = df.shape[0]
    stats['columns'] = df.shape[1]
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    # Column types
    stats['numerical_columns'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    stats['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    stats['datetime_columns'] = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Data distribution
    if 'Resigned' in df.columns:
        stats['resigned_count'] = df['Resigned'].sum()
        stats['resigned_percentage'] = (df['Resigned'].sum() / df.shape[0]) * 100
    elif 'resigned' in df.columns:
        stats['resigned_count'] = df['resigned'].sum()
        stats['resigned_percentage'] = (df['resigned'].sum() / df.shape[0]) * 100
    
    return stats

def identify_outliers(df, columns=None):
    """
    Identify outliers in numerical columns using IQR method
    
    Args:
        df (pd.DataFrame): The dataframe
        columns (list): List of columns to check (if None, all numerical columns are checked)
        
    Returns:
        dict: Dictionary mapping column names to outlier indices
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    outliers = {}
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        
        if len(outlier_indices) > 0:
            outliers[column] = list(outlier_indices)
    
    return outliers
