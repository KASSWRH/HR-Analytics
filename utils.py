from datetime import datetime
import pandas as pd
import numpy as np

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

def assign_risk_category(probability):
    """
    Assign risk category based on turnover probability.
    
    Args:
        probability: Turnover probability
    
    Returns:
        Risk category as string
    """
    if probability >= 0.6:
        return 'High'
    elif probability >= 0.3:
        return 'Medium'
    else:
        return 'Low'

def calculate_department_metrics(dept_data):
    """
    Calculate department-level metrics.
    
    Args:
        dept_data: DataFrame with department data
    
    Returns:
        Dictionary with department metrics
    """
    metrics = {
        'total_employees': len(dept_data),
        'high_risk_count': len(dept_data[dept_data['Risk_Category'] == 'High']),
        'avg_probability': dept_data['Turnover_Probability'].mean(),
        'avg_years': dept_data['Years_At_Company'].mean()
    }
    
    # Calculate high risk percentage
    metrics['high_risk_percentage'] = metrics['high_risk_count'] / metrics['total_employees']
    
    return metrics

def format_feature_name(feature_name):
    """
    Format feature names to be more readable.
    
    Args:
        feature_name: Original feature name
    
    Returns:
        Formatted feature name
    """
    # Replace underscores with spaces
    formatted = feature_name.replace('_', ' ')
    
    # Handle one-hot encoded features
    if ' x0 ' in formatted:
        parts = formatted.split(' x0 ')
        formatted = f"{parts[0]} = {parts[1]}"
    
    return formatted.title()
