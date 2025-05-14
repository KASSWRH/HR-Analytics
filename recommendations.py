import pandas as pd
import numpy as np
import random

def generate_recommendations(employee_data, all_employees_data, translation_func):
    """
    Generate personalized recommendations for employee retention.
    
    Args:
        employee_data: Series or dict with a single employee's data
        all_employees_data: DataFrame with all employees' data
        translation_func: Function for text translation
    
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Extract employee attributes
    risk_level = employee_data.get('Risk_Category', 'Medium')
    turnover_probability = employee_data.get('Turnover_Probability', 0.5)
    performance_score = employee_data.get('Performance_Score', 3)
    years_at_company = employee_data.get('Years_At_Company', 1)
    dept = employee_data.get('Department', '')
    job_title = employee_data.get('Job_Title', '')
    
    # Extract additional attributes if available
    satisfaction_score = employee_data.get('Employee_Satisfaction_Score', None)
    training_hours = employee_data.get('Training_Hours', None)
    salary = employee_data.get('Monthly_Salary', None)
    work_hours = employee_data.get('Work_Hours_Per_Week', None)
    projects = employee_data.get('Projects_Handled', None)
    remote_work = employee_data.get('Remote_Work_Frequency', None)
    overtime = employee_data.get('Overtime_Hours', None)
    
    # Generate recommendations based on risk level and employee attributes
    if risk_level == 'High':
        # Always include a general high-risk recommendation
        recommendations.append({
            'title': translation_func('high_risk_recommendation_title'),
            'description': translation_func('high_risk_recommendation_desc'),
            'action': translation_func('schedule_retention_interview')
        })
        
        # Check for low satisfaction
        if satisfaction_score is not None and satisfaction_score < 3:
            recommendations.append({
                'title': translation_func('low_satisfaction_title'),
                'description': translation_func('low_satisfaction_desc').format(score=satisfaction_score),
                'action': translation_func('conduct_satisfaction_survey')
            })
        
        # Check for salary issues
        if salary is not None:
            # Compare with department and job title average
            dept_job_avg = all_employees_data[
                (all_employees_data['Department'] == dept) & 
                (all_employees_data['Job_Title'] == job_title)
            ]['Monthly_Salary'].mean()
            
            if salary < dept_job_avg * 0.9:  # 10% below average
                recommendations.append({
                    'title': translation_func('compensation_review_title'),
                    'description': translation_func('compensation_review_desc'),
                    'action': translation_func('salary_adjustment_action')
                })
        
        # Check for overwork
        if overtime is not None and overtime > 15:
            recommendations.append({
                'title': translation_func('work_life_balance_title'),
                'description': translation_func('work_life_balance_desc'),
                'action': translation_func('reduce_overtime_action')
            })
        
        # Check if high performer with no recent promotion
        if performance_score >= 4 and 'Promotions' in employee_data:
            if employee_data['Promotions'] == 0 and years_at_company > 2:
                recommendations.append({
                    'title': translation_func('career_path_title'),
                    'description': translation_func('career_path_high_performer_desc'),
                    'action': translation_func('promotion_consideration_action')
                })
    
    elif risk_level == 'Medium':
        # General medium risk recommendation
        recommendations.append({
            'title': translation_func('medium_risk_recommendation_title'),
            'description': translation_func('medium_risk_recommendation_desc'),
            'action': translation_func('preventive_measures_action')
        })
        
        # Check for training opportunities
        if training_hours is not None and training_hours < 20:
            recommendations.append({
                'title': translation_func('training_opportunities_title'),
                'description': translation_func('training_opportunities_desc'),
                'action': translation_func('increase_training_action')
            })
        
        # Check for remote work opportunities
        if remote_work is not None and remote_work < 50:
            recommendations.append({
                'title': translation_func('flexible_work_title'),
                'description': translation_func('flexible_work_desc'),
                'action': translation_func('increase_remote_work_action')
            })
    
    else:  # Low risk
        # General low risk recommendation
        recommendations.append({
            'title': translation_func('low_risk_recommendation_title'),
            'description': translation_func('low_risk_recommendation_desc')
        })
        
        # For high performers, suggest development opportunities
        if performance_score >= 4:
            recommendations.append({
                'title': translation_func('talent_development_title'),
                'description': translation_func('talent_development_desc'),
                'action': translation_func('leadership_program_action')
            })
    
    # Add department-specific recommendations
    if dept == 'IT' or dept == 'Engineering':
        recommendations.append({
            'title': translation_func('tech_engagement_title'),
            'description': translation_func('tech_engagement_desc'),
            'action': translation_func('tech_engagement_action')
        })
    elif dept == 'Sales' or dept == 'Marketing':
        recommendations.append({
            'title': translation_func('sales_incentives_title'),
            'description': translation_func('sales_incentives_desc'),
            'action': translation_func('sales_incentives_action')
        })
    
    # Ensure we have at least 3 recommendations
    generic_recommendations = [
        {
            'title': translation_func('recognition_program_title'),
            'description': translation_func('recognition_program_desc'),
            'action': translation_func('implement_recognition_action')
        },
        {
            'title': translation_func('mentorship_title'),
            'description': translation_func('mentorship_desc'),
            'action': translation_func('assign_mentor_action')
        },
        {
            'title': translation_func('skill_development_title'),
            'description': translation_func('skill_development_desc'),
            'action': translation_func('create_development_plan_action')
        },
        {
            'title': translation_func('team_building_title'),
            'description': translation_func('team_building_desc'),
            'action': translation_func('organize_team_activity_action')
        }
    ]
    
    # Add generic recommendations if needed
    while len(recommendations) < 3:
        if not generic_recommendations:
            break
        
        # Get a random recommendation that's not already included
        rec = random.choice(generic_recommendations)
        generic_recommendations.remove(rec)
        recommendations.append(rec)
    
    return recommendations
