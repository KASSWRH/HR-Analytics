import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def generate_pdf_report(predictions, translation_func):
    """
    Generate a PDF report of turnover predictions.
    
    Args:
        predictions: DataFrame with predictions
        translation_func: Function for text translation
    
    Returns:
        PDF file as bytes
    """
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    
    # Create a PDF document
    with PdfPages(buffer) as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.9, translation_func('hr_analytics_report'), fontsize=24, ha='center')
        plt.text(0.5, 0.85, translation_func('turnover_prediction_analysis'), fontsize=18, ha='center')
        plt.text(0.5, 0.8, datetime.now().strftime('%Y-%m-%d'), fontsize=14, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Executive summary
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.95, translation_func('executive_summary'), fontsize=18, ha='center')
        
        # Calculate summary statistics
        total_employees = len(predictions)
        high_risk = len(predictions[predictions['Risk_Category'] == 'High'])
        medium_risk = len(predictions[predictions['Risk_Category'] == 'Medium'])
        low_risk = len(predictions[predictions['Risk_Category'] == 'Low'])
        
        high_risk_pct = high_risk / total_employees * 100
        medium_risk_pct = medium_risk / total_employees * 100
        low_risk_pct = low_risk / total_employees * 100
        
        # Add summary text
        summary_text = f"""
        {translation_func('total_employees_analyzed')}: {total_employees}

        {translation_func('employee_risk_breakdown')}:
        - {translation_func('high_risk')}: {high_risk} ({high_risk_pct:.1f}%)
        - {translation_func('medium_risk')}: {medium_risk} ({medium_risk_pct:.1f}%)
        - {translation_func('low_risk')}: {low_risk} ({low_risk_pct:.1f}%)

        {translation_func('avg_turnover_probability')}: {predictions['Turnover_Probability'].mean():.2f}
        
        {translation_func('departments_highest_risk')}:
        """
        
        # Add top 3 departments by risk
        dept_risk = predictions.groupby('Department')['Turnover_Probability'].mean().sort_values(ascending=False)
        for i, (dept, risk) in enumerate(dept_risk.head(3).items()):
            summary_text += f"  {i+1}. {dept}: {risk:.2f}\n"
        
        plt.text(0.1, 0.85, summary_text, fontsize=12, va='top')
        
        # Add a pie chart of risk categories
        plt.axes([0.2, 0.35, 0.6, 0.3])
        risk_counts = [high_risk, medium_risk, low_risk]
        risk_labels = [
            f"{translation_func('high_risk')} ({high_risk_pct:.1f}%)",
            f"{translation_func('medium_risk')} ({medium_risk_pct:.1f}%)",
            f"{translation_func('low_risk')} ({low_risk_pct:.1f}%)"
        ]
        plt.pie(risk_counts, labels=risk_labels, colors=['#ff6666', '#ffcc66', '#66cc66'], autopct='%1.1f%%')
        plt.title(translation_func('risk_distribution'))
        
        # Footer
        plt.text(0.5, 0.02, translation_func('report_generated_on') + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=8, ha='center')
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Department analysis
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.95, translation_func('department_analysis'), fontsize=18, ha='center')
        
        # Calculate department statistics
        dept_stats = predictions.groupby('Department').agg({
            'Turnover_Probability': ['mean', 'count'],
            'Employee_ID': 'count'
        }).reset_index()
        
        dept_stats.columns = ['Department', 'Avg_Risk', 'Risk_Count', 'Employee_Count']
        dept_stats['High_Risk_Count'] = predictions[predictions['Risk_Category'] == 'High'].groupby('Department').size().reindex(dept_stats['Department']).fillna(0).astype(int)
        dept_stats['High_Risk_Pct'] = dept_stats['High_Risk_Count'] / dept_stats['Employee_Count'] * 100
        dept_stats = dept_stats.sort_values('Avg_Risk', ascending=False)
        
        # Plot department risk
        plt.axes([0.1, 0.6, 0.8, 0.25])
        bars = plt.barh(dept_stats['Department'], dept_stats['Avg_Risk'], color='#3B6EA5')
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f"{dept_stats['Avg_Risk'].iloc[i]:.2f}", 
                     va='center')
        
        plt.xlim(0, max(dept_stats['Avg_Risk']) * 1.2)
        plt.title(translation_func('avg_turnover_risk_by_department'))
        plt.xlabel(translation_func('avg_risk'))
        
        # Create a table with department statistics
        plt.axes([0.1, 0.1, 0.8, 0.4])
        plt.axis('off')
        
        table_data = []
        table_data.append([translation_func('department'), 
                          translation_func('employees'), 
                          translation_func('avg_risk'), 
                          translation_func('high_risk_count'),
                          translation_func('high_risk_percentage')])
        
        for _, row in dept_stats.iterrows():
            table_data.append([
                row['Department'],
                f"{row['Employee_Count']}",
                f"{row['Avg_Risk']:.2f}",
                f"{row['High_Risk_Count']}",
                f"{row['High_Risk_Pct']:.1f}%"
            ])
        
        table = plt.table(cellText=table_data, loc='center', cellLoc='center', edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Footer
        plt.text(0.5, -0.1, translation_func('report_generated_on') + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=8, ha='center')
        
        pdf.savefig()
        plt.close()
        
        # High risk employees list
        high_risk_employees = predictions[predictions['Risk_Category'] == 'High'].sort_values('Turnover_Probability', ascending=False)
        
        if len(high_risk_employees) > 0:
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.95, translation_func('high_risk_employees'), fontsize=18, ha='center')
            
            # Add explanation
            plt.text(0.1, 0.9, translation_func('high_risk_explanation'), fontsize=10)
            
            # Split high risk employees into chunks of 20 for readability
            max_per_page = 20
            for i in range(0, len(high_risk_employees), max_per_page):
                if i > 0:
                    # Create a new page for each chunk after the first
                    plt.figure(figsize=(8.5, 11))
                    plt.text(0.5, 0.95, translation_func('high_risk_employees') + f" ({i+1}-{min(i+max_per_page, len(high_risk_employees))})", 
                             fontsize=18, ha='center')
                
                chunk = high_risk_employees.iloc[i:i+max_per_page]
                
                # Create a table of high risk employees
                plt.axes([0.05, 0.1, 0.9, 0.75])
                plt.axis('off')
                
                table_data = []
                table_data.append([
                    translation_func('employee_id'),
                    translation_func('department'),
                    translation_func('job_title'),
                    translation_func('risk_probability'),
                    translation_func('years_at_company')
                ])
                
                for _, row in chunk.iterrows():
                    table_data.append([
                        str(row['Employee_ID']),
                        row['Department'],
                        row['Job_Title'],
                        f"{row['Turnover_Probability']:.2f}",
                        f"{row['Years_At_Company']:.1f}"
                    ])
                
                table = plt.table(cellText=table_data, loc='center', cellLoc='center', edges='horizontal')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                # Footer
                plt.text(0.5, 0.02, translation_func('report_generated_on') + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                         fontsize=8, ha='center')
                
                pdf.savefig()
                plt.close()
            
        # Job title analysis
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.95, translation_func('job_title_analysis'), fontsize=18, ha='center')
        
        # Calculate job title statistics (top 10 by count)
        job_stats = predictions.groupby('Job_Title').agg({
            'Turnover_Probability': 'mean',
            'Employee_ID': 'count'
        }).reset_index()
        
        job_stats.columns = ['Job_Title', 'Avg_Risk', 'Employee_Count']
        job_stats = job_stats.sort_values('Employee_Count', ascending=False).head(10)
        
        # Plot job title risk
        plt.axes([0.1, 0.6, 0.8, 0.25])
        bars = plt.barh(job_stats['Job_Title'], job_stats['Avg_Risk'], color='#5A8F29')
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f"{job_stats['Avg_Risk'].iloc[i]:.2f}", 
                     va='center')
        
        plt.xlim(0, max(job_stats['Avg_Risk']) * 1.2)
        plt.title(translation_func('avg_turnover_risk_by_job_title'))
        plt.xlabel(translation_func('avg_risk'))
        
        # Create a table with job title statistics
        plt.axes([0.1, 0.1, 0.8, 0.4])
        plt.axis('off')
        
        table_data = []
        table_data.append([translation_func('job_title'), 
                          translation_func('employees'), 
                          translation_func('avg_risk')])
        
        for _, row in job_stats.iterrows():
            table_data.append([
                row['Job_Title'],
                f"{row['Employee_Count']}",
                f"{row['Avg_Risk']:.2f}"
            ])
        
        table = plt.table(cellText=table_data, loc='center', cellLoc='center', edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Footer
        plt.text(0.5, -0.1, translation_func('report_generated_on') + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=8, ha='center')
        
        pdf.savefig()
        plt.close()
        
        # Recommendations page
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.95, translation_func('recommendations'), fontsize=18, ha='center')
        
        recommendations_text = f"""
        1. {translation_func('focus_high_risk')}:
           {translation_func('high_risk_recommendation')}

        2. {translation_func('address_department_issues')}:
           {translation_func('department_recommendation')}

        3. {translation_func('improve_satisfaction')}:
           {translation_func('satisfaction_recommendation')}

        4. {translation_func('develop_talent')}:
           {translation_func('talent_recommendation')}

        5. {translation_func('monitor_changes')}:
           {translation_func('monitoring_recommendation')}
        """
        
        plt.text(0.1, 0.85, recommendations_text, fontsize=12, va='top')
        
        # Footer
        plt.text(0.5, 0.02, translation_func('report_generated_on') + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=8, ha='center')
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    # Reset buffer position to the beginning
    buffer.seek(0)
    
    # Return the PDF file
    return buffer.getvalue()
