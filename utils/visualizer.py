import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# Removed SHAP due to compatibility issues
# import shap 
from io import BytesIO
import streamlit as st

def plot_distribution(df, column, title=None, kde=True):
    """
    Plot the distribution of a column
    
    Args:
        df (pd.DataFrame): The dataframe
        column (str): The column to plot
        title (str): Plot title
        kde (bool): Whether to include KDE curve
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=kde, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Distribution of {column}')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    
    return fig

def plot_risk_distribution(predictions, t=None):
    """
    Plot histogram of risk predictions
    
    Args:
        predictions (np.array): Model predictions (probabilities)
        t: Translation function
        
    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # Define risk categories
    risk_categories = []
    colors = []
    for p in predictions:
        if p >= 0.7:
            risk_categories.append(t('high'))
            colors.append('red')
        elif p >= 0.3:
            risk_categories.append(t('medium'))
            colors.append('orange')
        else:
            risk_categories.append(t('low'))
            colors.append('green')
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'probability': predictions,
        'risk_category': risk_categories
    })
    
    # Create plot
    fig = px.histogram(
        plot_df, 
        x='probability', 
        color='risk_category',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
        title=t('turnover_risk_distribution'),
        labels={'probability': t('risk_level'), 'risk_category': t('risk_level')},
        nbins=20
    )
    
    fig.update_layout(
        xaxis_title=t('turnover_probability'),
        yaxis_title=t('employee_count'),
        legend_title=t('risk_level'),
        bargap=0.1
    )
    
    return fig

def plot_risk_by_category(df, category_column, risk_column, t=None):
    """
    Plot risk levels by category (e.g., department)
    
    Args:
        df (pd.DataFrame): The dataframe
        category_column (str): Column with categories
        risk_column (str): Column with risk probabilities
        t: Translation function
        
    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # Group by category and calculate metrics
    category_stats = df.groupby(category_column).agg({
        risk_column: ['mean', 'count']
    }).reset_index()
    
    category_stats.columns = [category_column, 'avg_risk', 'count']
    category_stats = category_stats.sort_values('avg_risk', ascending=False)
    
    # Calculate high risk percentage
    category_stats['high_risk_count'] = df[df[risk_column] >= 0.7].groupby(category_column).size().reindex(category_stats[category_column]).fillna(0)
    category_stats['high_risk_pct'] = (category_stats['high_risk_count'] / category_stats['count'] * 100).round(1)
    
    # Create plot
    fig = px.bar(
        category_stats,
        x=category_column,
        y='avg_risk',
        color='high_risk_pct',
        color_continuous_scale='Reds',
        text='high_risk_pct',
        hover_data=['count', 'high_risk_count'],
        title=f"{t('risk_by')} {category_column}"
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    fig.update_layout(
        xaxis_title=category_column,
        yaxis_title=t('average_risk_score'),
        coloraxis_colorbar_title=f"% {t('high_risk')}",
        height=500
    )
    
    return fig

def plot_correlation_heatmap(df, columns=None):
    """
    Plot correlation heatmap
    
    Args:
        df (pd.DataFrame): The dataframe
        columns (list): List of columns to include (defaults to all numerical)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if columns is None:
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Limit to 15 columns to avoid too large heatmaps
        if len(numerical_cols) > 15:
            numerical_cols = numerical_cols[:15]
        
        columns = numerical_cols
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Correlation Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout to ensure labels fit
    fig.tight_layout()
    
    return fig

def plot_feature_importance(importance_df, top_n=10, t=None):
    """
    Plot feature importance
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importance
        top_n (int): Number of top features to show
        t: Translation function
        
    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # Get top N features
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance')
    
    # Create plot
    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        title=t('feature_importance'),
        labels={'importance': t('importance_score'), 'feature': t('feature')},
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis_title='',
        height=500
    )
    
    return fig

def plot_shap_summary(shap_values, feature_names, t=None):
    """
    Create feature importance plot as an alternative to SHAP summary plot
    
    Args:
        shap_values: Feature importance values
        feature_names: List of feature names
        t: Translation function
        
    Returns:
        BytesIO: PNG image as bytes
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # Calculate average absolute impact for each feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Get top features by importance
    indices = np.argsort(feature_importance)[-15:]  # Top 15 features
    top_features = [feature_names[i] for i in indices]
    top_importance = feature_importance[indices]
    
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(top_features, top_importance, color='steelblue')
    plt.xlabel(t('feature_impact'))
    plt.ylabel('')
    plt.title(t('feature_importance_summary'))
    plt.gca().invert_yaxis()  # Display highest importance at the top
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

def plot_shap_force(explainer, shap_values, X, index, feature_names, t=None):
    """
    Create a bar chart showing feature impacts for a specific sample
    
    Args:
        explainer: Dictionary with feature importance info
        shap_values: Feature importance values 
        X: Feature matrix
        index: Index of the sample to explain
        feature_names: List of feature names
        t: Translation function
        
    Returns:
        BytesIO: PNG image as bytes
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # Get the importance values for the specific sample
    sample_importance = shap_values[index, :]
    
    # Sort by absolute importance
    indices = np.argsort(np.abs(sample_importance))[-10:]  # Top 10 features
    top_features = [feature_names[i] for i in indices]
    top_importance = sample_importance[indices]
    
    # Create a horizontal bar chart with colors based on impact direction
    plt.figure(figsize=(12, 6))
    colors = ['red' if x > 0 else 'blue' for x in top_importance]
    plt.barh(top_features, top_importance, color=colors)
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel(t('impact_on_prediction'))
    plt.ylabel('')
    plt.title(t('factors_influencing_prediction'))
    plt.gca().invert_yaxis()  # Display highest importance at the top
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

def create_department_dashboard(df, risk_column, t=None):
    """
    Create department dashboard visualizations
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        risk_column (str): Column containing risk probabilities
        t: Translation function
        
    Returns:
        tuple: (fig1, fig2, fig3)
            - fig1: Risk by department
            - fig2: Risk distribution by department
            - fig3: Department size comparison
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # 1. Risk by department
    fig1 = plot_risk_by_category(df, 'Department', risk_column, t)
    
    # 2. Risk distribution by department
    dept_counts = df.groupby(['Department', 'Risk_Level']).size().unstack(fill_value=0)
    if 'High' not in dept_counts.columns:
        dept_counts['High'] = 0
    if 'Medium' not in dept_counts.columns:
        dept_counts['Medium'] = 0
    if 'Low' not in dept_counts.columns:
        dept_counts['Low'] = 0
    
    fig2 = px.bar(
        dept_counts.reset_index(), 
        x='Department', 
        y=['High', 'Medium', 'Low'],
        title=t('risk_distribution_by_department'),
        labels={'value': t('employee_count'), 'Department': t('department'), 'variable': t('risk_level')},
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    )
    
    # 3. Department size comparison
    dept_size = df.groupby('Department').size().reset_index(name='count')
    fig3 = px.pie(
        dept_size, 
        values='count', 
        names='Department',
        title=t('department_size_comparison'),
        hole=0.4
    )
    
    return fig1, fig2, fig3

def create_employee_dashboard(employee_data, shap_values, feature_names, t=None):
    """
    Create individual employee dashboard visualizations
    
    Args:
        employee_data (pd.Series): Employee data
        shap_values: SHAP values for the employee
        feature_names: List of feature names
        t: Translation function
        
    Returns:
        tuple: (fig1, fig2)
            - fig1: Key factors chart
            - fig2: Employee metrics comparison
    """
    if t is None:
        # Default translation function
        t = lambda x: x
    
    # 1. Key factors influencing risk (top 5 SHAP values)
    top_features_idx = np.argsort(np.abs(shap_values))[-5:]
    top_features = [feature_names[i] for i in top_features_idx]
    feature_values = shap_values[top_features_idx]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=top_features,
        x=feature_values,
        orientation='h',
        marker_color=['red' if x > 0 else 'green' for x in feature_values]
    ))
    
    fig1.update_layout(
        title=t('key_factors_influencing_risk'),
        xaxis_title=t('impact_on_risk'),
        yaxis_title='',
        height=400
    )
    
    # 2. Radar chart comparing employee metrics to department average
    metrics = ['Performance_Score', 'Work_Hours_Per_Week', 'Overtime_Hours', 
               'Sick_Days', 'Training_Hours', 'Employee_Satisfaction_Score']
    available_metrics = [m for m in metrics if m in employee_data.index]
    
    if available_metrics:
        employee_values = [employee_data[m] for m in available_metrics]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatterpolar(
            r=employee_values,
            theta=available_metrics,
            fill='toself',
            name=t('employee')
        ))
        
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title=t('employee_metrics_comparison'),
            height=500
        )
    else:
        # If metrics not available, create a simple placeholder
        fig2 = go.Figure()
        fig2.add_annotation(
            text=t('metrics_not_available'),
            showarrow=False,
            font=dict(size=16)
        )
    
    return fig1, fig2
