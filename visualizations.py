import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def plot_feature_importance(feature_importance_df, x_label, y_label, top_n=15):
    """
    Plot feature importance.
    
    Args:
        feature_importance_df: DataFrame with feature names and importance scores
        x_label: Label for x-axis
        y_label: Label for y-axis
        top_n: Number of top features to display
    
    Returns:
        Plotly figure
    """
    # Get top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create plot
    fig = px.bar(
        top_features,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        labels={
            'Feature': x_label,
            'Importance': y_label
        },
        title=f'Top {top_n} Features by Importance'
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def plot_department_turnover(predictions, translation_func):
    """
    Plot department turnover risk.
    
    Args:
        predictions: DataFrame with predictions
        translation_func: Function for text translation
    
    Returns:
        Plotly figure
    """
    # Group by department
    dept_risk = predictions.groupby('Department')['Turnover_Probability'].agg(['mean', 'count']).reset_index()
    dept_risk.columns = ['Department', 'Average_Risk', 'Employee_Count']
    dept_risk = dept_risk.sort_values('Average_Risk', ascending=False)
    
    # Create color scale based on risk
    dept_risk['Color'] = dept_risk['Average_Risk'].apply(
        lambda x: 'red' if x >= 0.6 else ('orange' if x >= 0.3 else 'green')
    )
    
    # Create plot
    fig = px.bar(
        dept_risk,
        x='Department',
        y='Average_Risk',
        color='Average_Risk',
        color_continuous_scale=['green', 'yellow', 'red'],
        labels={
            'Department': translation_func('department'),
            'Average_Risk': translation_func('avg_turnover_probability')
        },
        title=translation_func('department_turnover_risk'),
        hover_data=['Employee_Count']
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def plot_risk_distribution(predictions, translation_func):
    """
    Plot distribution of turnover risk.
    
    Args:
        predictions: DataFrame with predictions
        translation_func: Function for text translation
    
    Returns:
        Plotly figure
    """
    # Create histogram
    fig = px.histogram(
        predictions, 
        x='Turnover_Probability',
        color='Risk_Category',
        color_discrete_map={
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        },
        labels={
            'Turnover_Probability': translation_func('turnover_probability'),
            'count': translation_func('employee_count'),
            'Risk_Category': translation_func('risk_category')
        },
        title=translation_func('turnover_probability_distribution'),
        nbins=30
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def plot_employee_analysis(values, metrics, translation_func):
    """
    Create radar chart for employee analysis.
    
    Args:
        values: List of values for each metric
        metrics: List of metric names
        translation_func: Function for text translation
    
    Returns:
        Plotly figure
    """
    # Normalize values to a 0-100 scale for better visualization
    # For each metric, we assume certain min and max values
    min_values = {
        'Performance_Score': 1,
        'Work_Hours_Per_Week': 20,
        'Projects_Handled': 0,
        'Employee_Satisfaction_Score': 1,
        'Training_Hours': 0,
        'Overtime_Hours': 0,
        'Sick_Days': 0
    }
    
    max_values = {
        'Performance_Score': 5,
        'Work_Hours_Per_Week': 60,
        'Projects_Handled': 50,
        'Employee_Satisfaction_Score': 5,
        'Training_Hours': 100,
        'Overtime_Hours': 30,
        'Sick_Days': 15
    }
    
    # Normalize values
    normalized_values = []
    for i, metric in enumerate(metrics):
        min_val = min_values.get(metric, 0)
        max_val = max_values.get(metric, 100)
        
        # Scale to 0-100
        normalized_value = 100 * (values[i] - min_val) / (max_val - min_val)
        
        # Clip to 0-100 range
        normalized_value = max(0, min(100, normalized_value))
        normalized_values.append(normalized_value)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=[translation_func(metric.lower()) for metric in metrics],
        fill='toself',
        name=translation_func('employee_metrics')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=translation_func('employee_performance_metrics')
    )
    
    return fig

def plot_shap_values(model, preprocessor, employee_data, feature_names, model_type, translation_func):
    """
    Create feature importance visualization for an employee.
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor
        employee_data: DataFrame with a single employee's data
        feature_names: Feature names
        model_type: Type of model
        translation_func: Function for text translation
    
    Returns:
        Matplotlib figure
    """
    # Transform employee data
    X_emp = preprocessor.transform(employee_data.drop('Resigned', axis=1) if 'Resigned' in employee_data.columns else employee_data)
    
    # Get feature importance from the model
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        # Fallback - equal importance
        importances = np.ones(len(feature_names))
    
    # Predict probability for this employee
    pred_proba = model.predict_proba(X_emp)[0, 1] if hasattr(model, 'predict_proba') else model.predict(X_emp)[0]
    
    # Scale the importances by the prediction probability to show impact on this prediction
    scaled_importances = importances * pred_proba
    
    # Create a DataFrame with feature names and importance values
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': scaled_importances
    })
    
    # Sort by absolute importance
    imp_df['Abs_Importance'] = np.abs(imp_df['Importance'])
    imp_df = imp_df.sort_values('Abs_Importance', ascending=False).head(10)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot horizontal bar chart
    bars = ax.barh(
        y=imp_df['Feature'],
        width=imp_df['Importance'],
        color=imp_df['Importance'].apply(lambda x: 'red' if x > 0 else 'blue')
    )
    
    # Add title and labels
    ax.set_title(translation_func('factors_affecting_prediction'))
    ax.set_xlabel(translation_func('feature_impact'))
    
    # Set y-axis labels with readable feature names
    readable_names = []
    for feature in imp_df['Feature']:
        # Clean up one-hot encoded feature names
        if '_' in feature and any(x in feature for x in ['Department', 'Job_Title', 'Education_Level']):
            parts = feature.split('_')
            category = parts[0]
            value = '_'.join(parts[1:])
            readable_names.append(f"{category}={value}")
        else:
            readable_names.append(feature)
    
    ax.set_yticklabels(readable_names)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    return fig
