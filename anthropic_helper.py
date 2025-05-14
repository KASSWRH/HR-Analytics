import os
import json
import anthropic
from anthropic import Anthropic
import pandas as pd

# the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

def get_anthropic_client():
    """Get an Anthropic client if API key is available."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    
    return Anthropic(api_key=api_key)

def generate_ai_recommendations(employee_data, risk_level):
    """
    Generate personalized recommendations for employee retention using Anthropic Claude.
    
    Args:
        employee_data: DataFrame or Series with a single employee's data
        risk_level: Risk level (High, Medium, Low)
    
    Returns:
        List of recommendation dictionaries or None if API is not available
    """
    client = get_anthropic_client()
    if not client:
        return None
    
    # Convert employee data to a readable format
    if isinstance(employee_data, pd.DataFrame):
        employee_dict = employee_data.iloc[0].to_dict()
    else:
        employee_dict = employee_data.to_dict()
    
    # Create the prompt
    prompt = f"""
    You are an expert HR consultant specializing in employee retention. Based on the following employee data, 
    provide 3 specific, actionable recommendations to reduce turnover risk. 
    This employee has been identified as having a {risk_level} risk level of leaving the company.
    
    Employee Data:
    {json.dumps(employee_dict, indent=2)}
    
    For each recommendation, provide:
    1. A clear title
    2. A detailed explanation
    3. The expected impact on retention
    
    Format your response as a JSON list with fields: title, explanation, impact
    """
    
    try:
        # Make the API call
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            temperature=0.7,
            system="You are an expert HR consultant. Provide recommendations in the requested JSON format only. Do not include any other text in your response.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the recommendations from the response
        response_text = response.content[0].text
        
        # Try to parse as JSON
        try:
            recommendations = json.loads(response_text)
            return recommendations
        except json.JSONDecodeError:
            # If not valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    recommendations = json.loads(json_match.group(0))
                    return recommendations
                except:
                    pass
            
            # If all else fails, return a formatted error
            return [{"title": "Error processing recommendations", 
                    "explanation": "Unable to generate AI recommendations. Please try again later.",
                    "impact": "None"}]
    
    except Exception as e:
        print(f"Error generating AI recommendations: {str(e)}")
        return None

def analyze_department_trends(department_data):
    """
    Analyze department data for trends and insights using Anthropic Claude.
    
    Args:
        department_data: DataFrame with department data
    
    Returns:
        Dictionary with insights or None if API is not available
    """
    client = get_anthropic_client()
    if not client:
        return None
    
    # Calculate some basic statistics
    dept_stats = {
        "employee_count": len(department_data),
        "avg_turnover_risk": department_data['Turnover_Probability'].mean(),
        "high_risk_count": len(department_data[department_data['Risk_Category'] == 'High']),
        "avg_performance": department_data['Performance_Score'].mean() if 'Performance_Score' in department_data.columns else None,
        "avg_tenure": department_data['Years_At_Company'].mean() if 'Years_At_Company' in department_data.columns else None,
        "department": department_data['Department'].iloc[0] if 'Department' in department_data.columns else "Unknown"
    }
    
    # Create the prompt
    prompt = f"""
    You are an expert HR data analyst. Based on the following department statistics, 
    provide insights and recommendations for improving retention in this department.
    
    Department: {dept_stats['department']}
    Number of Employees: {dept_stats['employee_count']}
    Average Turnover Risk: {dept_stats['avg_turnover_risk']:.2f}
    High Risk Employees: {dept_stats['high_risk_count']}
    Average Performance Score: {dept_stats['avg_performance']:.2f if dept_stats['avg_performance'] else 'N/A'}
    Average Tenure (Years): {dept_stats['avg_tenure']:.2f if dept_stats['avg_tenure'] else 'N/A'}
    
    Provide:
    1. Key insights about this department's retention situation
    2. Top 3 recommendations for improving retention
    3. Potential root causes of turnover issues
    
    Format your response as a JSON object with keys: insights, recommendations, root_causes
    Each should contain an array of strings.
    """
    
    try:
        # Make the API call
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            temperature=0.7,
            system="You are an expert HR analyst. Provide insights in the requested JSON format only. Do not include any other text in your response.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the insights from the response
        response_text = response.content[0].text
        
        # Try to parse as JSON
        try:
            insights = json.loads(response_text)
            return insights
        except json.JSONDecodeError:
            # If not valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    insights = json.loads(json_match.group(0))
                    return insights
                except:
                    pass
            
            # If all else fails, return basic insights
            return {
                "insights": ["Unable to generate AI insights for this department."],
                "recommendations": ["Consider manual analysis of department data."],
                "root_causes": ["Data analytics system requires troubleshooting."]
            }
    
    except Exception as e:
        print(f"Error generating department insights: {str(e)}")
        return None