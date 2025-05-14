import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from data_processing import preprocess_data, split_data, feature_importance, calculate_years_at_company
from models import train_model, evaluate_model, predict_turnover
from visualizations import (plot_department_turnover, plot_feature_importance, 
                            plot_employee_analysis, plot_risk_distribution, plot_shap_values)
from recommendations import generate_recommendations
from utils.utils import assign_risk_category, calculate_department_metrics, format_feature_name

# Utility functions now moved to utils/utils.py
from database import (save_session, load_sessions, load_session_data, create_tables,
                     delete_session, save_trained_model, load_trained_models, 
                     load_trained_model, delete_trained_model, get_latest_model_by_type)
from translations import translations
# Import Anthropic helper for AI-powered recommendations
from anthropic_helper import generate_ai_recommendations, analyze_department_trends
from pdf_generator import generate_pdf_report
from translations import translations

# Set page config
st.set_page_config(
    page_title="HR Analytics - Employee Turnover Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to generate printable HTML report
def generate_printable_report(predictions, is_individual=False, employee_id=None, department=None, lang='ar'):
    t = lambda key: translations.get(key, {}).get(lang, key)
    
    # Enhanced CSS for better printing experience
    css = """
    <style>
        body {
            font-family: 'Arial', 'Helvetica', sans-serif;
            line-height: 1.6;
            margin: 20px;
            direction: rtl;
            background-color: #ffffff;
            color: #333333;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            position: relative;
        }
        .header::before {
            content: "";
            position: absolute;
            bottom: -2px;
            right: 0;
            left: 0;
            height: 2px;
            background: linear-gradient(to left, #3498db, #2ecc71);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        h2 {
            color: #3498db;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-top: 30px;
            page-break-after: avoid;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }
        h4 {
            color: #2980b9;
            margin-top: 15px;
            page-break-after: avoid;
        }
        p {
            margin-bottom: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: right;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr {
            page-break-inside: avoid;
        }
        .risk-high {
            background-color: #ffcccc;
            color: #cc0000;
            font-weight: bold;
        }
        .risk-medium {
            background-color: #fff4cc;
            color: #cc7a00;
        }
        .risk-low {
            background-color: #ccffcc;
            color: #006600;
        }
        .metrics {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 20px;
            page-break-inside: avoid;
        }
        .metric-box {
            width: 22%;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
            color: #2980b9;
        }
        .reason {
            background-color: #f9f9f9;
            border-right: 4px solid #e74c3c;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            page-break-inside: avoid;
        }
        ul, ol {
            padding-right: 20px;
            margin-bottom: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        .print-button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .print-button:hover {
            background-color: #2980b9;
        }
        .print-only {
            display: none;
        }
        
        /* Page break controls */
        .page-break {
            page-break-after: always;
            height: 0;
            margin: 0;
            padding: 0;
        }
        
        /* Print-specific styles */
        @media print {
            @page {
                size: A4;
                margin: 1.5cm;
            }
            html, body {
                width: 210mm;
                height: 297mm;
            }
            .no-print {
                display: none !important;
            }
            .print-only {
                display: block;
            }
            body {
                margin: 0;
                padding: 15px;
                font-size: 12pt;
            }
            .header {
                position: running(header);
            }
            .metric-box {
                box-shadow: none;
                border: 1px solid #ddd;
                break-inside: avoid;
            }
            h1 { font-size: 22pt; }
            h2 { font-size: 18pt; }
            h3 { font-size: 15pt; }
            h4 { font-size: 13pt; }
            
            /* Guarantee that certain elements stay together */
            h1, h2, h3, h4, h5, h6 {
                page-break-after: avoid;
            }
            h1 + *, h2 + *, h3 + * {
                page-break-before: avoid;
            }
            table, figure, .metrics, .reason {
                page-break-inside: avoid;
            }
            
            /* Display URLs after links in printed version */
            a::after {
                content: " (" attr(href) ")";
                font-size: 90%;
                color: #333;
            }
        }
    </style>
    """
    
    # Improved print button with better styling and multiple browser support
    print_button = """
    <button class="print-button no-print" 
            style="background: #4CAF50; 
                   color: white; 
                   font-size: 18px; 
                   padding: 10px 20px; 
                   cursor: pointer; 
                   border: none; 
                   border-radius: 4px; 
                   margin: 20px 0; 
                   display: block;">طباعة التقرير</button>
    <script>
        // Function to handle print button click with better cross-browser support
        function printReport() {
            // For most modern browsers
            if (window.print) {
                // Give the browser a moment to render everything properly
                setTimeout(function() {
                    window.print();
                }, 300);
            } else {
                // Fallback message for very old browsers
                alert("عفواً، متصفحك لا يدعم وظيفة الطباعة. يرجى تحديث المتصفح أو استخدام متصفح آخر.");
            }
        }
        
        // Immediately attach event handlers when script loads
        (function() {
            // Multiple approaches to ensure button works
            var printButton = document.querySelector('.print-button');
            if (printButton) {
                // Modern event listener
                printButton.addEventListener('click', printReport);
                // Also set the onclick property for older browsers
                printButton.onclick = printReport;
            }
            
            // Add keyboard shortcut (Ctrl+P) in case button fails
            document.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
                    // Let browser handle the print dialog
                    console.log('Print shortcut detected');
                }
            });
            
            // Auto-trigger print dialog after 2 seconds for convenience
            setTimeout(function() {
                var printButton = document.querySelector('.print-button');
                if (printButton) {
                    // Make button pulse to attract attention
                    printButton.style.animation = 'pulse 1.5s infinite';
                    printButton.style.webkitAnimation = 'pulse 1.5s infinite';
                }
            }, 1000);
        })();
    </script>
    <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @-webkit-keyframes pulse {
            0% { -webkit-transform: scale(1); }
            50% { -webkit-transform: scale(1.05); }
            100% { -webkit-transform: scale(1); }
        }
    </style>
    """
    
    # Generate report content based on type
    if is_individual and employee_id is not None:
        # Individual employee report
        employee = predictions[predictions['Employee_ID'] == employee_id].iloc[0]
        
        # Header
        header = f"""
        <div class="header">
            <h1>تقرير مخاطر ترك العمل للموظف</h1>
            <p><strong>تاريخ التقرير:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """
        
        # Employee details
        employee_details = f"""
        <h2>معلومات الموظف</h2>
        <div class="metrics">
            <div class="metric-box">
                <p>رقم الموظف</p>
                <div class="metric-value">{employee['Employee_ID']}</div>
            </div>
            <div class="metric-box">
                <p>المسمى الوظيفي</p>
                <div class="metric-value">{employee['Job_Title']}</div>
            </div>
            <div class="metric-box">
                <p>القسم</p>
                <div class="metric-value">{employee['Department']}</div>
            </div>
            <div class="metric-box">
                <p>سنوات الخدمة</p>
                <div class="metric-value">{employee['Years_At_Company']:.1f}</div>
            </div>
        </div>
        """
        
        # Risk assessment
        risk_color = {
            'High': '#cc0000',
            'Medium': '#cc7a00',
            'Low': '#006600'
        }
        
        risk_assessment = f"""
        <h2>تقييم مخاطر ترك العمل</h2>
        <div style="text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px; margin-bottom: 30px;">
            <h3>احتمالية ترك العمل</h3>
            <div style="font-size: 36px; font-weight: bold; margin: 20px 0; color: {risk_color[employee['Risk_Category']]};">
                {employee['Turnover_Probability']:.1%}
            </div>
            <h3>مستوى المخاطرة</h3>
            <div style="font-size: 24px; font-weight: bold; color: {risk_color[employee['Risk_Category']]};">
                {employee['Risk_Category']}
            </div>
        </div>
        """
        
        # Potential resignation reasons based on employee data
        reasons_section = """
        <h2>الأسباب المحتملة للاستقالة</h2>
        <div style="padding: 15px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 20px;">
        """
        
        # Analyze employee data to determine potential reasons
        reasons = []
        
        # Salary-related reasons
        if 'Monthly_Salary' in employee and employee['Performance_Score'] > 3 and employee['Monthly_Salary'] < 10000:
            reasons.append("""
            <div class="reason">
                <h4>العوامل المالية والمرتبات</h4>
                <p>راتب الموظف أقل من المستوى المتوقع مقارنة بأدائه العالي، مما قد يؤدي إلى شعوره بعدم التقدير المالي لمساهماته.</p>
            </div>
            """)
            
        # Work hours related reasons
        if 'Work_Hours_Per_Week' in employee and employee['Work_Hours_Per_Week'] > 45:
            reasons.append("""
            <div class="reason">
                <h4>عبء العمل وساعات العمل</h4>
                <p>يعمل الموظف ساعات إضافية بشكل منتظم، مما قد يؤثر على توازن حياته المهنية والشخصية ويزيد من مستوى الإجهاد.</p>
            </div>
            """)
            
        # Career growth concerns
        if 'Years_At_Company' in employee and employee['Years_At_Company'] > 3 and employee['Performance_Score'] > 3:
            reasons.append("""
            <div class="reason">
                <h4>فرص التطور المهني</h4>
                <p>الموظف لديه أداء مرتفع وخدمة طويلة في الشركة ولكن قد يشعر بتوقف مساره الوظيفي أو محدودية فرص الترقية.</p>
            </div>
            """)
            
        # Department specific concerns
        if 'Department' in employee:
            if employee['Department'] == 'Sales':
                reasons.append("""
                <div class="reason">
                    <h4>ضغوط العمل في قسم المبيعات</h4>
                    <p>الموظفون في قسم المبيعات يعانون من ضغوط مستمرة لتحقيق الأهداف، مما قد يؤدي إلى الإرهاق وانخفاض الرضا الوظيفي.</p>
                </div>
                """)
            elif employee['Department'] == 'IT' or employee['Department'] == 'Technology':
                reasons.append("""
                <div class="reason">
                    <h4>تنافسية سوق العمل التقني</h4>
                    <p>خبراء التكنولوجيا مطلوبون بشدة في سوق العمل، مما يعرضهم لفرص خارجية أفضل من حيث الراتب والمزايا.</p>
                </div>
                """)
            elif employee['Department'] == 'HR' or employee['Department'] == 'Human Resources':
                reasons.append("""
                <div class="reason">
                    <h4>التحديات المتعلقة بإدارة الموارد البشرية</h4>
                    <p>قد يواجه موظفو الموارد البشرية تحديات في التعامل مع ضغوط ومتطلبات مختلف الإدارات، مما يزيد العبء عليهم.</p>
                </div>
                """)
        
        # Add more generic reasons if we haven't found specific ones
        if len(reasons) < 2:
            if employee['Risk_Category'] == 'High':
                reasons.append("""
                <div class="reason">
                    <h4>عدم الرضا الوظيفي</h4>
                    <p>قد يعاني الموظف من عدم الرضا عن بيئة العمل أو ثقافة الشركة أو أسلوب الإدارة.</p>
                </div>
                """)
                reasons.append("""
                <div class="reason">
                    <h4>فرص سوق العمل</h4>
                    <p>توفر فرص مهنية أفضل في سوق العمل من حيث التعويضات أو المسار المهني أو بيئة العمل.</p>
                </div>
                """)
        
        # Complete reasons section
        for reason in reasons:
            reasons_section += reason
            
        reasons_section += "</div>"
        
        # Personalized recommendations
        recommendations = """
        <h2>التوصيات المخصصة للاحتفاظ بالموظف</h2>
        """
        
        # Add personalized recommendations section
        if employee['Risk_Category'] == 'High':
            recommendations += """
            <div style="padding: 15px; background-color: #ffeeee; border-radius: 5px; margin-bottom: 15px;">
                <h3>خطة احتفاظ عاجلة</h3>
                <p>هذا الموظف معرض بدرجة كبيرة لخطر الاستقالة ويتطلب اهتمامًا فوريًا واستراتيجية احتفاظ مخصصة.</p>
            </div>
            """
        elif employee['Risk_Category'] == 'Medium':
            recommendations += """
            <div style="padding: 15px; background-color: #fff8ee; border-radius: 5px; margin-bottom: 15px;">
                <h3>خطة احتفاظ متوسطة الأولوية</h3>
                <p>هذا الموظف يحتاج إلى مراقبة وخطة تطوير مخصصة لتعزيز رضاه الوظيفي وتقليل احتمالية تركه للعمل.</p>
            </div>
            """
        else:
            recommendations += """
            <div style="padding: 15px; background-color: #eeffee; border-radius: 5px; margin-bottom: 15px;">
                <h3>خطة تطوير مستمرة</h3>
                <p>مخاطر استقالة هذا الموظف منخفضة، ولكن يجب الاستمرار في خطط التطوير والتحفيز الروتينية.</p>
            </div>
            """
            
        recommendations += "<ul>"
        
        # Generate tailored recommendations based on employee data
        rec_list = []
        
        # Add specific recommendations based on identified reasons
        for reason in reasons:
            if "العوامل المالية" in reason:
                rec_list.append("إجراء مراجعة فورية للراتب وتعديله بما يتناسب مع أداء الموظف وقيمته في السوق")
                rec_list.append("تقديم مكافآت مالية مرتبطة بالأداء ومزايا إضافية لتحسين التعويض الإجمالي")
            
            if "عبء العمل" in reason:
                rec_list.append("مراجعة عبء العمل وتوزيع المهام بشكل أكثر توازناً")
                rec_list.append("النظر في برامج العمل المرنة أو العمل عن بعد لتحسين التوازن بين الحياة المهنية والشخصية")
            
            if "فرص التطور" in reason:
                rec_list.append("تطوير خطة مسار وظيفي واضحة مع خطوات الترقية المحتملة والمهارات المطلوبة")
                rec_list.append("توفير فرص للتدريب وتطوير المهارات في مجالات جديدة لتوسيع آفاق التطور المهني")
            
            if "قسم المبيعات" in reason:
                rec_list.append("مراجعة أهداف المبيعات لضمان واقعيتها وتحقيق التوازن بين التحدي وإمكانية الإنجاز")
                rec_list.append("تطوير نظام دعم أفضل لفريق المبيعات وتحسين أدوات العمل")
            
            if "سوق العمل التقني" in reason:
                rec_list.append("تحديث حزمة التعويضات والمزايا لتكون منافسة في سوق تكنولوجيا المعلومات")
                rec_list.append("توفير فرص العمل على أحدث التقنيات والمشاريع المبتكرة للحفاظ على الاهتمام المهني")
            
            if "الموارد البشرية" in reason:
                rec_list.append("تقديم الدعم الإضافي لفريق الموارد البشرية وتبسيط العمليات الإدارية")
                rec_list.append("توفير فرص التدريب المتخصص في مجالات متقدمة من إدارة الموارد البشرية")
        
        # Add general recommendations if we don't have enough specific ones
        if len(rec_list) < 3:
            if employee['Risk_Category'] == 'High':
                rec_list.append("إجراء مقابلة احتفاظ عاجلة مع الموظف للاستماع إلى مخاوفه واحتياجاته")
                rec_list.append("تطوير حزمة تعويضات مخصصة تشمل مكافآت مالية ومزايا إضافية")
                rec_list.append("تقديم فرص للعمل على مشاريع مهمة ومرئية تعزز من مكانة الموظف في المؤسسة")
            elif employee['Risk_Category'] == 'Medium':
                rec_list.append("جدولة مقابلات دورية للتطوير المهني ومتابعة رضا الموظف")
                rec_list.append("تقديم فرص تدريبية وتطويرية تتماشى مع اهتمامات الموظف")
            else:
                rec_list.append("الحفاظ على التواصل المنتظم واستمرار برامج التطوير الحالية")
        
        # Add recommendations to HTML
        for r in rec_list:
            recommendations += f"<li>{r}</li>\n"
        
        recommendations += "</ul>"
        
        # Add action plan section
        recommendations += """
        <h3>خطة العمل المقترحة</h3>
        <div style="padding: 15px; background-color: #f0f8ff; border-radius: 5px;">
            <p><strong>الخطوات التالية:</strong></p>
            <ol>
                <li>جدولة اجتماع مباشر مع الموظف خلال الأسبوع القادم</li>
                <li>مناقشة مسار التطور المهني وتوثيق أهداف الموظف</li>
                <li>مراجعة حزمة التعويضات والمزايا مع الإدارة</li>
                <li>تطوير خطة تطوير مهني مخصصة بالتعاون مع الموظف</li>
                <li>جدولة متابعة دورية كل ثلاثة أشهر لقياس فعالية الخطة</li>
            </ol>
        </div>
        """
        
        # Complete the report
        report_html = f"{css}{header}{print_button}{employee_details}{risk_assessment}{reasons_section}{recommendations}"
        
    elif department is not None:
        # Department level report
        dept_data = predictions[predictions['Department'] == department]
        dept_metrics = calculate_department_metrics(dept_data)
        
        # Header
        header = f"""
        <div class="header">
            <h1>تقرير مخاطر ترك العمل للقسم</h1>
            <p><strong>القسم:</strong> {department}</p>
            <p><strong>تاريخ التقرير:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """
        
        # Department metrics
        dept_summary = f"""
        <h2>ملخص القسم</h2>
        <div class="metrics">
            <div class="metric-box">
                <p>عدد الموظفين</p>
                <div class="metric-value">{dept_metrics['total_employees']}</div>
            </div>
            <div class="metric-box">
                <p>نسبة المخاطر العالية</p>
                <div class="metric-value">{dept_metrics['high_risk_percentage']:.1%}</div>
            </div>
            <div class="metric-box">
                <p>متوسط احتمالية ترك العمل</p>
                <div class="metric-value">{dept_metrics['avg_probability']:.2f}</div>
            </div>
            <div class="metric-box">
                <p>متوسط سنوات الخدمة</p>
                <div class="metric-value">{dept_metrics['avg_years']:.1f}</div>
            </div>
        </div>
        """
        
        # High risk employees table
        high_risk = dept_data[dept_data['Risk_Category'] == 'High'].sort_values(
            'Turnover_Probability', ascending=False
        )
        
        high_risk_table = """
        <h2>الموظفون ذوو المخاطر العالية</h2>
        """
        
        if len(high_risk) > 0:
            table = """
            <table>
                <tr>
                    <th>رقم الموظف</th>
                    <th>المسمى الوظيفي</th>
                    <th>احتمالية ترك العمل</th>
                    <th>درجة الأداء</th>
                    <th>سنوات الخدمة</th>
                </tr>
            """
            
            for _, row in high_risk.iterrows():
                table += f"""
                <tr class="risk-high">
                    <td>{row['Employee_ID']}</td>
                    <td>{row['Job_Title']}</td>
                    <td>{row['Turnover_Probability']:.1%}</td>
                    <td>{row['Performance_Score']}</td>
                    <td>{row['Years_At_Company']:.1f}</td>
                </tr>
                """
                
            table += "</table>"
            high_risk_table += table
        else:
            high_risk_table += "<p>لا يوجد موظفون ذوو مخاطر عالية في هذا القسم.</p>"
        
        # Job title risk section
        job_risk = dept_data.groupby('Job_Title')['Turnover_Probability'].mean().reset_index()
        job_risk = job_risk.sort_values('Turnover_Probability', ascending=False)
        
        job_risk_table = """
        <h2>مخاطر ترك العمل حسب المسمى الوظيفي</h2>
        <table>
            <tr>
                <th>المسمى الوظيفي</th>
                <th>متوسط احتمالية ترك العمل</th>
                <th>مستوى المخاطرة</th>
            </tr>
        """
        
        for _, row in job_risk.iterrows():
            risk_level = assign_risk_category(row['Turnover_Probability'])
            risk_class = f"risk-{risk_level.lower()}"
            
            job_risk_table += f"""
            <tr class="{risk_class}">
                <td>{row['Job_Title']}</td>
                <td>{row['Turnover_Probability']:.1%}</td>
                <td>{risk_level}</td>
            </tr>
            """
        
        job_risk_table += "</table>"
        
        # Recommendations
        recommendations = """
        <h2>توصيات للقسم</h2>
        """
        
        if dept_metrics['high_risk_percentage'] > 0.3:
            recommendations += """
            <div style="padding: 15px; background-color: #ffcccc; border-radius: 5px;">
                <h3>مخاطر عالية للقسم</h3>
                <p>هذا القسم يواجه مخاطر عالية لترك الموظفين. يجب اتخاذ إجراءات فورية لتحسين الاحتفاظ بالموظفين.</p>
            </div>
            <ul>
                <li>إجراء مراجعة شاملة لسياسات الرواتب والتعويضات في القسم</li>
                <li>تقييم عبء العمل وتوازن الحياة المهنية للموظفين</li>
                <li>تحسين برامج التطوير المهني وفرص الترقية</li>
                <li>معالجة قضايا الثقافة التنظيمية والقيادة</li>
                <li>تنفيذ برامج احتفاظ خاصة للموظفين ذوي المخاطر العالية</li>
            </ul>
            """
        elif dept_metrics['high_risk_percentage'] > 0.15:
            recommendations += """
            <div style="padding: 15px; background-color: #fff4cc; border-radius: 5px;">
                <h3>مخاطر متوسطة للقسم</h3>
                <p>هذا القسم يواجه بعض المخاطر المتعلقة بترك الموظفين. هناك حاجة إلى تحسينات محددة.</p>
            </div>
            <ul>
                <li>تحليل أسباب مخاطر ترك العمل بين المسميات الوظيفية المختلفة</li>
                <li>تحسين برامج التقدير والمكافآت</li>
                <li>تقديم فرص تدريبية إضافية وبرامج تطوير المهارات</li>
                <li>تعزيز التواصل وجمع التغذية الراجعة من الموظفين</li>
            </ul>
            """
        else:
            recommendations += """
            <div style="padding: 15px; background-color: #ccffcc; border-radius: 5px;">
                <h3>مخاطر منخفضة للقسم</h3>
                <p>هذا القسم يتمتع بمعدل احتفاظ جيد بالموظفين. استمر في الممارسات الحالية مع التحسين المستمر.</p>
            </div>
            <ul>
                <li>الحفاظ على التواصل المنتظم مع الموظفين</li>
                <li>الاستمرار في تقديم فرص النمو والتطوير</li>
                <li>مشاركة أفضل الممارسات مع الأقسام الأخرى</li>
            </ul>
            """
        
        # Complete the report
        report_html = f"{css}{header}{print_button}{dept_summary}{high_risk_table}{job_risk_table}{recommendations}"
        
    else:
        # Overall report for all data
        # Header
        header = f"""
        <div class="header">
            <h1>تقرير تحليل مخاطر ترك العمل</h1>
            <p><strong>تاريخ التقرير:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """
        
        # Overall metrics
        total_employees = len(predictions)
        high_risk = len(predictions[predictions['Risk_Category'] == 'High'])
        medium_risk = len(predictions[predictions['Risk_Category'] == 'Medium'])
        low_risk = len(predictions[predictions['Risk_Category'] == 'Low'])
        
        overall_metrics = f"""
        <h2>الملخص العام</h2>
        <div class="metrics">
            <div class="metric-box">
                <p>إجمالي الموظفين</p>
                <div class="metric-value">{total_employees}</div>
            </div>
            <div class="metric-box">
                <p>موظفون بمخاطر عالية</p>
                <div class="metric-value">{high_risk} ({high_risk/total_employees:.1%})</div>
            </div>
            <div class="metric-box">
                <p>موظفون بمخاطر متوسطة</p>
                <div class="metric-value">{medium_risk} ({medium_risk/total_employees:.1%})</div>
            </div>
            <div class="metric-box">
                <p>موظفون بمخاطر منخفضة</p>
                <div class="metric-value">{low_risk} ({low_risk/total_employees:.1%})</div>
            </div>
        </div>
        """
        
        # Department breakdown
        dept_breakdown = """
        <h2>تحليل القسم</h2>
        <table>
            <tr>
                <th>القسم</th>
                <th>عدد الموظفين</th>
                <th>نسبة المخاطر العالية</th>
                <th>متوسط احتمالية ترك العمل</th>
            </tr>
        """
        
        for dept in predictions['Department'].unique():
            dept_data = predictions[predictions['Department'] == dept]
            total_dept = len(dept_data)
            dept_high_risk = len(dept_data[dept_data['Risk_Category'] == 'High'])
            dept_high_pct = dept_high_risk / total_dept if total_dept > 0 else 0
            dept_avg_prob = dept_data['Turnover_Probability'].mean()
            
            risk_class = ""
            if dept_high_pct > 0.3:
                risk_class = "risk-high"
            elif dept_high_pct > 0.15:
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            dept_breakdown += f"""
            <tr class="{risk_class}">
                <td>{dept}</td>
                <td>{total_dept}</td>
                <td>{dept_high_pct:.1%}</td>
                <td>{dept_avg_prob:.2f}</td>
            </tr>
            """
        
        dept_breakdown += "</table>"
        
        # Top high-risk employees
        top_risk = predictions.sort_values('Turnover_Probability', ascending=False).head(10)
        
        top_risk_table = """
        <h2>أعلى 10 موظفين من حيث مخاطر ترك العمل</h2>
        <table>
            <tr>
                <th>رقم الموظف</th>
                <th>القسم</th>
                <th>المسمى الوظيفي</th>
                <th>احتمالية ترك العمل</th>
                <th>درجة الأداء</th>
                <th>سنوات الخدمة</th>
            </tr>
        """
        
        for _, row in top_risk.iterrows():
            risk_class = "risk-high" if row['Risk_Category'] == 'High' else "risk-medium"
            top_risk_table += f"""
            <tr class="{risk_class}">
                <td>{row['Employee_ID']}</td>
                <td>{row['Department']}</td>
                <td>{row['Job_Title']}</td>
                <td>{row['Turnover_Probability']:.1%}</td>
                <td>{row['Performance_Score']}</td>
                <td>{row['Years_At_Company']:.1f}</td>
            </tr>
            """
            
        top_risk_table += "</table>"
        
        # Summary
        summary = """
        <h2>ملخص وتوصيات</h2>
        <p>
            يقدم هذا التقرير تحليلاً شاملاً لمخاطر ترك العمل داخل المنظمة. 
            الإجراءات الموصى بها تشمل:
        </p>
        <ul>
            <li>إعطاء الأولوية للتدخلات في الأقسام ذات نسب المخاطر العالية</li>
            <li>وضع خطط احتفاظ مخصصة للموظفين ذوي القيمة العالية والمخاطر العالية</li>
            <li>معالجة العوامل الرئيسية المؤثرة على ترك العمل وفقًا لتحليل النظام</li>
            <li>تنفيذ برامج تحسين مستمرة وجمع التغذية الراجعة من الموظفين</li>
            <li>متابعة مقاييس الاحتفاظ بالموظفين بشكل دوري ومراجعة التقدم</li>
        </ul>
        """
        
        # Complete the report
        report_html = f"{css}{header}{print_button}{overall_metrics}{dept_breakdown}{top_risk_table}{summary}"
    
    return report_html

# Function to load the latest trained model automatically
def load_latest_model_if_available():
    """Attempt to load the latest trained model from database if no model is already loaded"""
    try:
        # Check for trained models
        trained_models = load_trained_models()
        if trained_models:
            # Load the latest model
            latest_model = trained_models[0]  # First model is the latest
            model_id = latest_model[0]
            try:
                model, preprocessor, feature_names, metrics, model_type = load_trained_model(model_id)
                
                # Save model in session state
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.feature_names = feature_names
                st.session_state.model_type = model_type
                st.session_state.loaded_model_id = model_id
                
                # Save metrics if available
                if metrics:
                    st.session_state.model_metrics = metrics
                
                print(f"Model automatically loaded: {latest_model[1]}")
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                # Try fallback to default model if available
                if len(trained_models) > 1:
                    try:
                        fallback_model = trained_models[1]
                        model_id = fallback_model[0]
                        model, preprocessor, feature_names, _, model_type = load_trained_model(model_id)
                        
                        # Save fallback model in session state
                        st.session_state.model = model
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                        st.session_state.model_type = model_type
                        st.session_state.loaded_model_id = model_id
                        
                        print(f"Fallback model loaded: {fallback_model[1]}")
                        return True
                    except Exception as e2:
                        print(f"Error loading fallback model: {str(e2)}")
    except Exception as e:
        print(f"Error checking for trained models: {str(e)}")
    
    return False

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'print_predictions' not in st.session_state:
    st.session_state.print_predictions = None
    
# Try to load the latest model automatically at startup
load_latest_model_if_available()
    
# Check for printable report view
query_params = st.query_params
view_mode = query_params.get("view", None)

if view_mode == "print_report":
    if 'print_predictions' in st.session_state and st.session_state.print_predictions is not None:
        predictions = st.session_state.print_predictions
        
        # Check for individual employee or department report
        employee_id = query_params.get("employee_id", None)
        department = query_params.get("department", None)
        
        if employee_id is not None:
            # Individual employee report
            employee_id = int(employee_id) if employee_id.isdigit() else employee_id
            report_html = generate_printable_report(
                predictions, 
                is_individual=True, 
                employee_id=employee_id, 
                lang=st.session_state.language
            )
        elif department is not None:
            # Department report
            report_html = generate_printable_report(
                predictions, 
                department=department, 
                lang=st.session_state.language
            )
        else:
            # General report
            report_html = generate_printable_report(
                predictions, 
                lang=st.session_state.language
            )
        
        # Display the report
        st.components.v1.html(report_html, height=800, scrolling=True)
        st.stop()
    else:
        st.error("لا توجد بيانات متاحة للتقرير. يرجى العودة وتحميل البيانات أولاً.")
        
        # Back button
        if st.button("العودة إلى التطبيق الرئيسي"):
            js = """
            <script>
                window.location.href = "/";
            </script>
            """
            st.components.v1.html(js, height=0)
        st.stop()
if 'session_name' not in st.session_state:
    st.session_state.session_name = ""
if 'previous_sessions' not in st.session_state:
    st.session_state.previous_sessions = None
if 'comparison_session' not in st.session_state:
    st.session_state.comparison_session = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
    
# Settings state
if 'custom_recommendations' not in st.session_state:
    st.session_state.custom_recommendations = []
if 'external_model' not in st.session_state:
    st.session_state.external_model = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_endpoint' not in st.session_state:
    st.session_state.api_endpoint = ""
if 'enable_notifications' not in st.session_state:
    st.session_state.enable_notifications = True
if 'notification_threshold' not in st.session_state:
    st.session_state.notification_threshold = 0.6
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'unread_notifications' not in st.session_state:
    st.session_state.unread_notifications = 0

# Initialize database
create_tables()

# Translation function
def t(key):
    return translations.get(key, {}).get(st.session_state.language, key)

# Sidebar for language selection and session management
with st.sidebar:
    # Logo and title
    st.image("assets/logo.svg", width=100)
    st.title(t("app_title"))
    
    # Language selector
    lang_option = st.selectbox(
        t("select_language"),
        options=["English", "العربية"],
        index=0 if st.session_state.language == 'en' else 1
    )
    
    if lang_option == "English" and st.session_state.language != 'en':
        st.session_state.language = 'en'
        st.rerun()
    elif lang_option == "العربية" and st.session_state.language != 'ar':
        st.session_state.language = 'ar'
        st.rerun()
    
    st.divider()
    
    # Session management
    st.subheader(t("session_management"))
    session_name = st.text_input(t("session_name"), value=st.session_state.session_name)
    
    if session_name != st.session_state.session_name:
        st.session_state.session_name = session_name
    
    save_button = st.button(t("save_session"))
    if save_button and st.session_state.data is not None and st.session_state.predictions is not None:
        if not session_name:
            st.error(t("session_name_required"))
        else:
            # Get the model type if available
            model_type = getattr(st.session_state, 'model_type', None)
            
            save_session(
                session_name, 
                st.session_state.data, 
                st.session_state.predictions, 
                st.session_state.model,
                st.session_state.preprocessor,
                st.session_state.feature_names,
                model_type,
                True,  # This is a training session by default
                None,  # No reference to a pre-trained model
                None   # No additional notes
            )
            st.success(t("session_saved"))
    
    st.divider()
    
    # Load previous sessions
    st.subheader(t("previous_sessions"))
    previous_sessions = load_sessions()
    st.session_state.previous_sessions = previous_sessions
    
    if previous_sessions and len(previous_sessions) > 0:
        selected_session = st.selectbox(
            t("select_session"),
            options=[session[1] for session in previous_sessions],
            index=0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button(t("load_session"))
            if load_btn:
                session_id = [s[0] for s in previous_sessions if s[1] == selected_session][0]
                result = load_session_data(session_id)
                
                if len(result) == 9:
                    # New format with additional fields
                    (data, predictions, model, preprocessor, feature_names, 
                     model_type, is_training_session, used_model_id, notes) = result
                else:
                    # Old format for backwards compatibility
                    (data, predictions, model, preprocessor, feature_names) = result
                    model_type = None
                    is_training_session = True
                    used_model_id = None
                    notes = None
                
                # Update session state
                st.session_state.data = data
                st.session_state.predictions = predictions
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.feature_names = feature_names
                st.session_state.session_name = selected_session
                
                # Store additional metadata if available
                if model_type:
                    st.session_state.model_type = model_type
                if used_model_id:
                    st.session_state.loaded_model_id = used_model_id
                
                # Show success message with more details
                if is_training_session:
                    success_msg = t("session_loaded")
                else:
                    success_msg = f"Prediction session loaded using '{model_type}' model"
                st.success(success_msg)
                st.rerun()
        
        with col2:
            compare_btn = st.button(t("compare_session"))
            if compare_btn:
                session_id = [s[0] for s in previous_sessions if s[1] == selected_session][0]
                result = load_session_data(session_id)
                
                if len(result) == 9:
                    # New format with additional fields
                    (data, predictions, _, _, _, _, _, _, _) = result
                else:
                    # Old format for backwards compatibility
                    (data, predictions, _, _, _) = result
                
                st.session_state.comparison_session = selected_session
                st.session_state.comparison_data = (data, predictions)
                st.success(t("comparison_ready"))

# Main content
st.title(t("hr_analytics_title"))
st.write(t("app_description"))

# Show info about automatically loaded model if it exists
if st.session_state.model is not None:
    model_type_str = getattr(st.session_state, 'model_type', 'Unknown')
    model_name = ""
    
    # Try to get the model name if loaded_model_id exists
    if hasattr(st.session_state, 'loaded_model_id') and st.session_state.loaded_model_id is not None:
        try:
            model_id = st.session_state.loaded_model_id
            trained_models = load_trained_models()
            for model in trained_models:
                if model[0] == model_id:
                    model_name = model[1]
                    break
        except:
            pass
    
    st.success(f"نموذج مدرب جاهز للاستخدام: {model_name} (نوع: {model_type_str}). يمكنك الانتقال مباشرة إلى قسم 'التنبؤات' لتحميل بيانات جديدة واستخدام هذا النموذج.")

# Tabs for different sections
# Create a badge for unread notifications
notification_label = t("notifications")
if st.session_state.unread_notifications > 0:
    notification_label = f"{t('notifications')} 🔴"

# Determine which tab should be selected by default
# If we have a loaded model, select the predictions tab (index 2) by default
default_tab_index = 2 if st.session_state.model is not None else 0

# Create tabs with the determined default tab
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    t("data_upload"), 
    t("model_training"), 
    t("predictions"), 
    t("individual_analysis"), 
    t("department_analysis"),
    t("visual_analytics"),
    t("settings"),
    notification_label
])

# Tab 1: Data Upload and Preprocessing
with tab1:
    st.header("تحميل بيانات للتدريب")
    
    st.write(
        """
        #### استخدم هذا القسم لتحميل البيانات التاريخية للتدريب
        حمّل ملف البيانات الخاص بالموظفين (التاريخي) ليتم استخدامه في تدريب النموذج التنبؤي.
        يمكنك استخدام ملف CSV أو Excel يحتوي على بيانات الموظفين.
        
        **ملاحظة مهمة**: هذه البيانات ستستخدم فقط لتدريب النموذج. لإجراء تنبؤات على بيانات جديدة، 
        يرجى الانتقال إلى قسم "التنبؤات" بعد تدريب النموذج.
        """
    )
    
    uploaded_file = st.file_uploader("تحميل ملف بيانات التدريب (CSV أو Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display sample of the data
            st.subheader("معاينة البيانات")
            st.dataframe(df.head())
            
            # Data information
            st.subheader("معلومات البيانات")
            cols = st.columns(4)
            cols[0].metric("عدد السجلات", df.shape[0])
            cols[1].metric("عدد الخصائص", df.shape[1])
            cols[2].metric("الموظفون المستقيلون", df['Resigned'].sum() if 'Resigned' in df.columns else "غير متوفر")
            
            # Check for required columns
            required_columns = [
                'Employee_ID', 'Department', 'Performance_Score', 
                'Monthly_Salary', 'Work_Hours_Per_Week', 'Resigned'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"الأعمدة المطلوبة المفقودة: {', '.join(missing_columns)}")
            else:
                # Save the data to session state
                st.session_state.data = df
                st.success("تم تحميل بيانات التدريب بنجاح! انتقل إلى تبويب 'تدريب النموذج' للمتابعة.")
                
                # Data preprocessing info
                with st.expander("تفاصيل معالجة البيانات"):
                    st.write(
                        """
                        سيتم معالجة البيانات قبل التدريب بالطرق التالية:
                        - معالجة القيم المفقودة
                        - تحويل البيانات النصية إلى شكل رقمي
                        - تطبيع البيانات الرقمية
                        - حساب الميزات المشتقة (مثل سنوات العمل في الشركة)
                        """
                    )
                    
                    # Display data types and missing values
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("أنواع البيانات")
                        # Convert dtypes to strings to avoid Arrow conversion issues
                        dtypes_df = pd.DataFrame({
                            "العمود": df.dtypes.index,
                            "نوع البيانات": df.dtypes.astype(str)
                        })
                        st.dataframe(dtypes_df)
                    
                    with col2:
                        st.write("القيم المفقودة")
                        # Format missing values data to avoid Arrow conversion issues
                        missing_vals = df.isnull().sum()
                        missing_data = pd.DataFrame({
                            "العمود": missing_vals.index,
                            "القيم المفقودة": missing_vals.values,
                            "النسبة المئوية": 100 * missing_vals.values / len(df)
                        })
                        missing_data["النسبة المئوية"] = missing_data["النسبة المئوية"].round(2)
                        st.dataframe(missing_data)
                
                # Show button to go to training tab
                if st.button("الانتقال إلى تدريب النموذج", type="primary"):
                    # Since Streamlit doesn't support direct tab switching, we'll add this note
                    st.info("يرجى النقر على تبويب 'تدريب النموذج' أعلاه للمتابعة")
                
        except Exception as e:
            st.error(t("error_loading_data") + f": {str(e)}")

# Tab 2: Model Training
with tab2:
    st.header(t("model_training"))
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Training options: use existing model or train new one
        st.subheader("Training Options")
        
        # Get list of trained models from database
        trained_models = load_trained_models()
        
        training_option = st.radio(
            "Select an option",
            ["Use a pre-trained model", "Train a new model"],
            index=0 if trained_models and len(trained_models) > 0 else 1
        )
        
        if training_option == "Use a pre-trained model":
            if trained_models and len(trained_models) > 0:
                # Display available models
                st.subheader("Available Pre-trained Models")
                
                # Create a dataframe for better display
                models_df = pd.DataFrame({
                    "ID": [m[0] for m in trained_models],
                    "Name": [m[1] for m in trained_models],
                    "Type": [m[2] for m in trained_models],
                    "Created": [m[3] for m in trained_models],
                    "Training Data Size": [m[4] if m[4] else "Unknown" for m in trained_models]
                })
                
                st.dataframe(models_df)
                
                # Select model to use
                selected_model_id = st.selectbox(
                    "Select model to use",
                    options=models_df["ID"].tolist(),
                    format_func=lambda x: f"{models_df[models_df['ID']==x]['Name'].iloc[0]} ({models_df[models_df['ID']==x]['Type'].iloc[0]})"
                )
                
                # Load model button
                load_model_btn = st.button("Load Selected Model", type="primary")
                
                if load_model_btn:
                    with st.spinner("Loading pre-trained model..."):
                        model, preprocessor, feature_names, metrics, model_type = load_trained_model(selected_model_id)
                        
                        # Save to session state
                        st.session_state.model = model
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                        st.session_state.model_type = model_type
                        
                        st.success(f"Model '{models_df[models_df['ID']==selected_model_id]['Name'].iloc[0]}' loaded successfully!")
                        
                        # Display metrics if available
                        if metrics:
                            st.subheader("Model Performance Metrics")
                            cols = st.columns(5)
                            metrics_keys = ["accuracy", "precision", "recall", "f1", "auc"]
                            
                            for i, key in enumerate(metrics_keys):
                                if key in metrics:
                                    cols[i].metric(t(key), f"{metrics[key]:.2f}")
            else:
                st.info("No pre-trained models available. Please train a new model first.")
                training_option = "Train a new model"
        
        if training_option == "Train a new model":
            # Model configuration
            st.subheader(t("model_configuration"))
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider(t("test_size"), 0.1, 0.5, 0.3, 0.05)
                model_type = st.selectbox(
                    t("model_type"),
                    options=["XGBoost", "Random Forest", "Logistic Regression"]
                )
                
                # Model name for saving
                model_name = st.text_input("Model Name", value=f"{model_type} - {datetime.now().strftime('%Y-%m-%d')}")
            
            with col2:
                target_col = st.selectbox(
                    t("target_column"),
                    options=[col for col in data.columns if col.lower() in ['resigned', 'attrition', 'turnover', 'left']],
                    index=0
                )
                
                id_col = st.selectbox(
                    t("id_column"),
                    options=[col for col in data.columns if 'id' in col.lower()],
                    index=0
                )
                
                # Option to save the trained model
                save_model_option = st.checkbox("Save model after training", value=True)
            
            # Train button
            train_btn = st.button(t("train_model"), type="primary")
            
            if train_btn:
                try:
                    with st.spinner(t("training_model")):
                        # Preprocess data
                        X, y, preprocessor, feature_names = preprocess_data(data, target_col, id_col)
                        
                        # Save preprocessor and feature names to session state
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                        
                        # Split data
                        X_train, X_test, y_train, y_test = split_data(X, y, test_size)
                        
                        # Train model
                        model = train_model(X_train, y_train, model_type)
                        
                        # Evaluate model
                        accuracy, precision, recall, f1, auc, conf_matrix = evaluate_model(model, X_test, y_test)
                        
                        # Create metrics dictionary
                        metrics = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "auc": auc,
                            "confusion_matrix": conf_matrix.tolist()
                        }
                        
                        # Save model to session state
                        st.session_state.model = model
                        st.session_state.model_type = model_type
                        
                        # Save trained model to database if option selected
                        if save_model_option:
                            model_id = save_trained_model(
                                model_name, 
                                model_type, 
                                model, 
                                preprocessor, 
                                feature_names, 
                                metrics, 
                                len(data),
                                f"Trained on {len(data)} records. Test size: {test_size}"
                            )
                            st.success(f"Model saved with ID: {model_id}")
                        
                        # Display metrics
                        st.subheader(t("model_performance"))
                        
                        cols = st.columns(5)
                        cols[0].metric(t("accuracy"), f"{accuracy:.2f}")
                        cols[1].metric(t("precision"), f"{precision:.2f}")
                        cols[2].metric(t("recall"), f"{recall:.2f}")
                        cols[3].metric(t("f1_score"), f"{f1:.2f}")
                        cols[4].metric(t("auc"), f"{auc:.2f}")
                        
                        # Display confusion matrix
                        st.subheader(t("confusion_matrix"))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.matshow(conf_matrix, cmap='Blues')
                        for (i, j), val in np.ndenumerate(conf_matrix):
                            ax.text(j, i, f"{val}", ha='center', va='center')
                        ax.set_xlabel(t("predicted_label"))
                        ax.set_ylabel(t("true_label"))
                        ax.set_title(t("confusion_matrix"))
                        ax.xaxis.set_ticks([0, 1])
                        ax.yaxis.set_ticks([0, 1])
                        ax.xaxis.set_ticklabels([t("stayed"), t("resigned")])
                        ax.yaxis.set_ticklabels([t("stayed"), t("resigned")])
                        st.pyplot(fig)
                        
                        # Feature importance
                        st.subheader(t("feature_importance"))
                        feature_imp = feature_importance(model, feature_names, model_type)
                        fig = plot_feature_importance(feature_imp, t("feature"), t("importance_score"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(t("model_trained_successfully"))
                
                except Exception as e:
                    st.error(t("error_training_model") + f": {str(e)}")
    else:
        st.warning(t("upload_data_first"))

# Tab 3: Predictions and Analysis
with tab3:
    st.header(t("predictions_analysis"))
    
    # Try to load model automatically if not already loaded
    if st.session_state.model is None:
        load_latest_model_if_available()
    
    # Check if we have a model loaded
    if st.session_state.model is not None:
        model_type_str = getattr(st.session_state, 'model_type', 'Unknown')
        model_name = ""
        
        # Try to get the model name if loaded_model_id exists
        loaded_model_id = getattr(st.session_state, 'loaded_model_id', None)
        trained_models = load_trained_models()
        
        if loaded_model_id is not None and trained_models:
            for model in trained_models:
                if model[0] == loaded_model_id:
                    model_name = model[1]
                    break
        
        # محاولة الحصول على دقة النموذج
        model_accuracy = None
        if loaded_model_id is not None and trained_models:
            try:
                # تحميل معلومات النموذج للحصول على الدقة
                _, _, _, metrics, _ = load_trained_model(loaded_model_id)
                if metrics and 'accuracy' in metrics:
                    model_accuracy = metrics['accuracy']
            except:
                pass
        
        # عرض معلومات النموذج مع الدقة إذا كانت متوفرة
        if model_accuracy:
            st.success(f"النموذج الحالي جاهز للاستخدام: {model_name} (نوع: {model_type_str}، الدقة: {model_accuracy:.2f})")
        else:
            st.success(f"النموذج الحالي جاهز للاستخدام: {model_name} (نوع: {model_type_str})")
        
        # Create a 2-column layout for model selection and data upload
        col_model, col_data = st.columns(2)
        
        with col_model:
            # Model selection section
            st.subheader("اختيار النموذج")
            
            if trained_models and len(trained_models) > 0:
                # Create a dataframe for better display of available models
                models_df = pd.DataFrame({
                    "ID": [m[0] for m in trained_models],
                    "Name": [m[1] for m in trained_models],
                    "Type": [m[2] for m in trained_models],
                    "Created": [m[3] for m in trained_models],
                })
                
                # Display the model selection dropdown
                selected_model_id = st.selectbox(
                    "اختر النموذج المراد استخدامه للتنبؤ",
                    options=models_df["ID"].tolist(),
                    format_func=lambda x: f"{models_df[models_df['ID']==x]['Name'].iloc[0]} ({models_df[models_df['ID']==x]['Type'].iloc[0]})"
                )
                
                # Load model button
                load_model_btn = st.button("تحميل النموذج للتنبؤ", type="primary")
                
                if load_model_btn:
                    with st.spinner("جاري تحميل النموذج المدرب..."):
                        model, preprocessor, feature_names, _, model_type = load_trained_model(selected_model_id)
                        
                        # Save to session state
                        st.session_state.model = model
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                        st.session_state.model_type = model_type
                        st.session_state.loaded_model_id = selected_model_id
                        
                        # Reset predictions to allow new predictions with this model
                        if 'predictions' in st.session_state:
                            st.session_state.predictions = None
                        
                        st.success(f"تم تحميل النموذج '{models_df[models_df['ID']==selected_model_id]['Name'].iloc[0]}' بنجاح!")
                        st.rerun()
            else:
                st.warning("لا توجد نماذج مدربة. يرجى تدريب نموذج أولاً في تبويب 'تدريب النموذج'.")
        
        with col_data:
            # Data loading section
            st.subheader("تحميل بيانات للتنبؤ")
            
            if 'data' not in st.session_state or st.session_state.data is None:
                st.info("يرجى تحميل ملف بيانات جديد للتنبؤ باستخدام النموذج المدرب.")
            
            # DATA LOADING SECTION - Allow user to load new data for prediction
            prediction_file = st.file_uploader(
                "تحميل ملف بيانات الموظفين (CSV أو Excel)", 
                type=["csv", "xlsx", "xls"],
                key="prediction_file_uploader_main"
            )
            
            if prediction_file:
                try:
                    # Load the data
                    if prediction_file.name.endswith('.csv'):
                        prediction_data = pd.read_csv(prediction_file)
                    else:
                        prediction_data = pd.read_excel(prediction_file)
                    
                    # Update session state with the new prediction data
                    st.session_state.data = prediction_data
                    
                    # Reset predictions if new data is loaded
                    st.session_state.predictions = None
                    
                    # Show preview
                    st.subheader("معاينة البيانات")
                    st.dataframe(prediction_data.head())
                    
                    st.success(f"تم تحميل البيانات بنجاح! عدد السجلات: {len(prediction_data)}")
                    
                    # Option to save the prediction session
                    prediction_name = st.text_input(
                        "اسم جلسة التنبؤ", 
                        value=f"تنبؤ {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        key="prediction_name_main"
                    )
                except Exception as e:
                    st.error(f"خطأ في تحميل البيانات: {str(e)}")
        
        # Add large button to generate predictions below both columns
        if st.session_state.data is not None and ('predictions' not in st.session_state or st.session_state.predictions is None):
            st.write("")  # Add some spacing
            
            # إضافة حقل لاسم جلسة التنبؤ
            prediction_name = st.text_input(
                "اسم جلسة التنبؤ",
                value=f"تنبؤ {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                key="pred_name_input"
            )
            
            # Add large button to generate predictions
            st.markdown("<br>", unsafe_allow_html=True)  # إضافة مسافة
            generate_btn = st.button("إنشاء التنبؤات", type="primary", key="gen_preds_main", use_container_width=True)
            if generate_btn:
                with st.spinner(t("generating_predictions")):
                    # Auto-load model if not already loaded
                    if st.session_state.model is None:
                        load_latest_model_if_available()
                        
                    # Check if model is loaded now
                    if st.session_state.model is not None:
                        model = st.session_state.model
                        preprocessor = st.session_state.preprocessor
                        feature_names = st.session_state.feature_names
                        model_type = getattr(st.session_state, 'model_type', 'Unknown')
                        data = st.session_state.data
                        
                        # Make predictions
                        predictions = predict_turnover(data, model, preprocessor, feature_names)
                        
                        # Add risk category
                        predictions['Risk_Category'] = predictions['Turnover_Probability'].apply(assign_risk_category)
                        
                        # Save predictions to session state
                        st.session_state.predictions = predictions
                        
                        # Display success message
                        st.success("تم إنشاء التنبؤات بنجاح!")
                    else:
                        st.error("لم يتم العثور على نموذج مدرب. يرجى الانتقال إلى قسم 'تدريب النموذج' أولاً.")
                    
                    # Save session to database
                    if st.session_state.model is not None and prediction_name:
                        # Get updated values from session state to ensure we have the most current data
                        model = st.session_state.model
                        data = st.session_state.data
                        predictions = st.session_state.predictions
                        preprocessor = st.session_state.preprocessor
                        feature_names = st.session_state.feature_names
                        model_type = getattr(st.session_state, 'model_type', 'Unknown')
                        used_model_id = getattr(st.session_state, 'loaded_model_id', None)
                            
                        try:
                            save_session(
                                prediction_name,
                                data,
                                predictions,
                                None,  # Don't save model in prediction session
                                preprocessor,
                                feature_names,
                                model_type,
                                False,  # Not a training session
                                used_model_id,
                                f"التنبؤ على {len(data)} سجل باستخدام نموذج {model_type}"
                            )
                            st.success("تم حفظ جلسة التنبؤ بنجاح!")
                        except Exception as e:
                            st.error(f"حدث خطأ عند حفظ الجلسة: {str(e)}")
                    
                    st.success("تم إنشاء التنبؤات بنجاح!")
                    
                    # Quick metrics overview
                    st.subheader("ملخص النتائج")
                    total_employees = len(predictions)
                    high_risk = len(predictions[predictions['Risk_Category'] == 'High'])
                    high_risk_percent = high_risk / total_employees * 100
                    
                    st.info(f"عدد الموظفين ذوي المخاطر العالية: {high_risk} من أصل {total_employees} ({high_risk_percent:.1f}%)")
                    
                    # Recommend viewing departments
                    st.markdown("**يمكنك الآن:**")
                    st.markdown("- تحليل النتائج حسب القسم في تبويب 'تحليل الأقسام'")
                    st.markdown("- تحليل الموظفين الفرديين في تبويب 'تحليل فردي للموظفين'")
                    st.markdown("- الاطلاع على التحليلات المرئية في تبويب 'التحليلات المرئية'")
                    
                    # Refresh page to update state
                    st.rerun()
    else:
        # Allow selecting a trained model for predictions first
        trained_models = load_trained_models()
        
        if not trained_models or len(trained_models) == 0:
            st.warning("لا توجد نماذج مدربة. يرجى تدريب نموذج أولاً في تبويب 'تدريب النموذج'.")
            
            # Add button to go to training tab
            if st.button("الانتقال إلى تدريب النموذج", type="primary"):
                st.info("يرجى النقر على تبويب 'تدريب النموذج' أعلاه للمتابعة")
            
            st.stop()
        # Show currently loaded model information
        model_type_str = getattr(st.session_state, 'model_type', 'Unknown')
        model_name = ""
        
        # Try to get the model name if loaded_model_id exists
        if hasattr(st.session_state, 'loaded_model_id') and st.session_state.loaded_model_id is not None:
            try:
                model_id = st.session_state.loaded_model_id
                for model in trained_models:
                    if model[0] == model_id:
                        model_name = model[1]
                        break
            except:
                pass
        
        # محاولة الحصول على دقة النموذج
        model_accuracy = None
        if hasattr(st.session_state, 'loaded_model_id') and st.session_state.loaded_model_id is not None:
            try:
                # تحميل معلومات النموذج للحصول على الدقة
                _, _, _, metrics, _ = load_trained_model(st.session_state.loaded_model_id)
                if metrics and 'accuracy' in metrics:
                    model_accuracy = metrics['accuracy']
            except:
                pass
        
        # عرض معلومات النموذج مع الدقة إذا كانت متوفرة
        if model_accuracy:
            st.success(f"النموذج الحالي المستخدم للتنبؤ: {model_name} (نوع: {model_type_str}، الدقة: {model_accuracy:.2f})")
        else:
            st.success(f"النموذج الحالي المستخدم للتنبؤ: {model_name} (نوع: {model_type_str})")
    
    # جزء عرض النتائج - سيتم الاحتفاظ به
    # لاحظ: هذا لإظهار مساحة بين عناصر الواجهة فقط
    if st.session_state.model is not None and st.session_state.data is not None and st.session_state.predictions is None:
        # المساحة بين عناصر واجهة المستخدم
        st.write("")
        
    # الحصول على التنبؤات من حالة الجلسة
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
    else:
        predictions = None
    
    # Check if predictions are available
    if predictions is not None:
        # Display overall metrics
        st.subheader(t("overall_metrics"))
        
        cols = st.columns(4)
        total_employees = len(predictions)
        high_risk = len(predictions[predictions['Risk_Category'] == 'High'])
        medium_risk = len(predictions[predictions['Risk_Category'] == 'Medium'])
        low_risk = len(predictions[predictions['Risk_Category'] == 'Low'])
        
        cols[0].metric(t("total_employees"), total_employees)
        cols[1].metric(t("high_risk_employees"), high_risk, f"{high_risk/total_employees:.1%}")
        cols[2].metric(t("medium_risk_employees"), medium_risk, f"{medium_risk/total_employees:.1%}")
        cols[3].metric(t("low_risk_employees"), low_risk, f"{low_risk/total_employees:.1%}")
        
        # Risk distribution plot
        st.subheader(t("risk_distribution"))
        fig = plot_risk_distribution(predictions, t)
        st.plotly_chart(fig, use_container_width=True)
        
        # Department turnover risk
        st.subheader(t("department_turnover_risk"))
        dept_fig = plot_department_turnover(predictions, t)
        st.plotly_chart(dept_fig, use_container_width=True)
        
        # Display predictions table
        st.subheader(t("employee_predictions"))
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            dept_filter = st.multiselect(
                t("filter_by_department"),
                options=sorted(predictions['Department'].unique()),
                default=[]
            )
            
        with col2:
            risk_filter = st.multiselect(
                t("filter_by_risk"),
                options=['High', 'Medium', 'Low'],
                default=[]
            )
            
        with col3:
            prob_threshold = st.slider(
                t("probability_threshold"),
                0.0, 1.0, 0.0, 0.05
            )
            
        # Apply filters
        filtered_predictions = predictions.copy()
        if dept_filter:
            filtered_predictions = filtered_predictions[filtered_predictions['Department'].isin(dept_filter)]
        if risk_filter:
            filtered_predictions = filtered_predictions[filtered_predictions['Risk_Category'].isin(risk_filter)]
        if prob_threshold > 0:
            filtered_predictions = filtered_predictions[filtered_predictions['Turnover_Probability'] >= prob_threshold]
            
        # Sort by probability
        filtered_predictions = filtered_predictions.sort_values('Turnover_Probability', ascending=False)
            
        # Display table
        st.dataframe(
            filtered_predictions[[
                'Employee_ID', 'Department', 'Job_Title', 'Turnover_Probability', 
                'Risk_Category', 'Performance_Score', 'Years_At_Company'
            ]],
            use_container_width=True
        )
    else:
        st.info("قم بتحميل بيانات التنبؤ أولاً من خلال زر 'تحميل بيانات للتنبؤ' أعلاه.")
        
    # If we have predictions, show export options and comparison
    if predictions is not None:
        # Define filtered_predictions variable for use in exports
        filtered_predictions = predictions.copy()
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            export_csv = st.button(t("export_to_csv"), key="export_csv_btn")
            if export_csv:
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label=t("download_csv"),
                    data=csv,
                    file_name=f"turnover_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_csv_all"
                )
            
            with col2:
                export_pdf = st.button(t("generate_pdf_report"), key="export_pdf_btn")
                if export_pdf:
                    with st.spinner(t("generating_pdf")):
                        pdf_file = generate_pdf_report(predictions, t)
                        st.download_button(
                            label=t("download_pdf"),
                            data=pdf_file,
                            file_name=f"turnover_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            key="download_pdf_btn"
                        )
            
            with col3:
                # Printable web report button
                print_report = st.button("عرض تقرير قابل للطباعة", key="print_report_btn")
                if print_report:
                    # Store the current predictions in session state for the printable report
                    if 'filtered_predictions' in locals():
                        st.session_state.print_predictions = filtered_predictions
                    else:
                        st.session_state.print_predictions = predictions
                    
                    # Open in a new tab using JavaScript
                    js = f"""
                    <script>
                        window.open("/?view=print_report", "_blank");
                    </script>
                    """
                    st.components.v1.html(js, height=0)
            
            # Comparison with previous session
            if st.session_state.comparison_data is not None:
                st.subheader(t("comparison_with_previous"))
                st.write(t("comparing_with") + f" '{st.session_state.comparison_session}'")
                
                comparison_data, comparison_predictions = st.session_state.comparison_data
                
                # Calculate metrics for both
                current_high_risk = len(predictions[predictions['Risk_Category'] == 'High']) / len(predictions)
                previous_high_risk = len(comparison_predictions[comparison_predictions['Risk_Category'] == 'High']) / len(comparison_predictions)
                
                cols = st.columns(3)
                cols[0].metric(
                    t("high_risk_percentage"),
                    f"{current_high_risk:.1%}",
                    f"{(current_high_risk - previous_high_risk) * 100:.1f}%",
                    delta_color="inverse"
                )
                
                cols[1].metric(
                    t("avg_turnover_probability"),
                    f"{predictions['Turnover_Probability'].mean():.2f}",
                    f"{predictions['Turnover_Probability'].mean() - comparison_predictions['Turnover_Probability'].mean():.2f}",
                    delta_color="inverse"
                )
                
                # Department comparison
                st.subheader(t("department_comparison"))
                
                # Prepare comparison data
                dept_current = predictions.groupby('Department')['Turnover_Probability'].mean().reset_index()
                dept_previous = comparison_predictions.groupby('Department')['Turnover_Probability'].mean().reset_index()
                
                dept_comparison = pd.merge(
                    dept_current, dept_previous,
                    on='Department',
                    suffixes=('_current', '_previous')
                )
                
                dept_comparison['Difference'] = dept_comparison['Turnover_Probability_current'] - dept_comparison['Turnover_Probability_previous']
                
                # Plot comparison
                fig = px.bar(
                    dept_comparison,
                    x='Department',
                    y='Difference',
                    color='Difference',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    labels={'Difference': t("probability_difference")},
                    title=t("department_turnover_change")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Clear comparison button
                if st.button(t("clear_comparison"), key="clear_comp_btn"):
                    st.session_state.comparison_session = None
                    st.session_state.comparison_data = None
                    st.rerun()
    
    else:
        st.warning(t("train_model_first"))

# Tab 4: Individual Employee Analysis
with tab4:
    st.header(t("individual_employee_analysis"))
    
    # Try to load model automatically if not already loaded
    if st.session_state.model is None:
        load_latest_model_if_available()
    
    if st.session_state.predictions is None:
        st.warning("يجب إجراء التنبؤات أولاً قبل تحليل الموظفين بشكل فردي.")
        
        # If we have a model but no predictions, guide the user
        if st.session_state.model is not None:
            st.info("لديك نموذج جاهز للاستخدام. يرجى الانتقال إلى قسم 'التنبؤات' لتحميل بيانات الموظفين وإجراء التنبؤات أولاً.")
            if st.button("الانتقال إلى قسم التنبؤات", key="goto_predictions_from_individual"):
                st.info("يرجى النقر على تبويب 'التنبؤات' أعلاه للمتابعة")
    
    elif st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Employee selector
        st.subheader(t("select_employee"))
        
        col1, col2 = st.columns(2)
        with col1:
            employee_id = st.selectbox(
                t("employee_id"),
                options=sorted(predictions['Employee_ID'].unique()),
                index=0
            )
        
        # Get employee data
        employee_data = predictions[predictions['Employee_ID'] == employee_id].iloc[0]
        employee_df = st.session_state.data[st.session_state.data['Employee_ID'] == employee_id]
        
        # Display employee information
        st.subheader(t("employee_information"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(t("department"), employee_data['Department'])
        
        with col2:
            st.metric(t("job_title"), employee_data['Job_Title'])
        
        with col3:
            st.metric(t("years_at_company"), f"{employee_data['Years_At_Company']:.1f}")
        
        with col4:
            st.metric(t("performance_score"), f"{employee_data['Performance_Score']}")
            
        # Print report button
        if st.button("طباعة تقرير الموظف", key="print_employee_report"):
            st.session_state.print_predictions = predictions
            js = f"""
            <script>
                window.open("/?view=print_report&employee_id={employee_id}", "_blank");
            </script>
            """
            st.components.v1.html(js, height=0)
        
        # Turnover risk
        st.subheader(t("turnover_risk_analysis"))
        
        risk_color = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="background-color: {risk_color[employee_data['Risk_Category']]};
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;">
                    <h2>{t("turnover_probability")}</h2>
                    <h1>{employee_data['Turnover_Probability']:.1%}</h1>
                    <h3>{t("risk_level")}: {t(employee_data['Risk_Category'].lower())}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            # Get SHAP values for this employee
            if 'model_type' in locals():
                model_type_value = model_type
            else:
                model_type_value = "XGBoost"  # default
                
            fig = plot_shap_values(
                st.session_state.model, 
                st.session_state.preprocessor, 
                employee_df, 
                st.session_state.feature_names,
                model_type_value,
                t
            )
            st.pyplot(fig)
        
        # Recommendations
        st.subheader(t("retention_recommendations"))
        
        # Choose between standard and AI-powered recommendations
        recommendation_tabs = st.tabs(["Standard Recommendations", "AI-Powered Insights"])
        
        with recommendation_tabs[0]:
            # Standard recommendations
            recommendations = generate_recommendations(employee_data, st.session_state.data, t)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec['title']}**")
                st.markdown(f"{rec['description']}")
        
        with recommendation_tabs[1]:
            # Check if we have access to Anthropic Claude
            ai_recommendations = None
            if 'ANTHROPIC_API_KEY' in os.environ or st.session_state.external_model == "Anthropic":
                with st.spinner("Generating AI-powered recommendations..."):
                    ai_recommendations = generate_ai_recommendations(
                        employee_data, 
                        employee_data['Risk_Category']
                    )
                    
                    if ai_recommendations:
                        for i, rec in enumerate(ai_recommendations, 1):
                            with st.expander(f"{i}. {rec.get('title', 'Recommendation')}"):
                                st.markdown(f"**Explanation:** {rec.get('explanation', rec.get('description', 'No details available'))}")
                                st.markdown(f"**Expected Impact:** {rec.get('impact', 'Impact not specified')}")
                    else:
                        st.info("AI-powered recommendations could not be generated. Please check your Anthropic API key.")
            else:
                st.info("To enable AI-powered recommendations, please set up Anthropic API access in the Settings tab.")
                if st.button("Go to Settings"):
                    # This is a workaround since streamlit doesn't support direct tab switching
                    st.session_state.active_tab = "settings"
                    st.rerun()
            
            # Display suggested action if available in the recommendations
            if ai_recommendations and len(ai_recommendations) > 0:
                for rec in ai_recommendations:
                    if 'action' in rec:
                        st.info(f"**{t('suggested_action')}:** {rec['action']}")
                        break
            
            st.divider()
        
        # Employee performance metrics
        st.subheader(t("performance_metrics"))
        
        metrics = ['Performance_Score', 'Work_Hours_Per_Week', 'Projects_Handled', 
                 'Employee_Satisfaction_Score', 'Training_Hours']
        
        try:
            radar_values = [employee_df[metric].iloc[0] for metric in metrics if metric in employee_df.columns]
            radar_metrics = [metric for metric in metrics if metric in employee_df.columns]
            
            if len(radar_values) > 2:  # Need at least 3 metrics for radar chart
                radar_fig = plot_employee_analysis(radar_values, radar_metrics, t)
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning(t("insufficient_metrics"))
        except:
            st.warning(t("metrics_not_available"))
    
    else:
        st.warning(t("generate_predictions_first"))

# Tab 5: Department Analysis
with tab5:
    st.header(t("department_analysis"))
    
    # Try to load model automatically if not already loaded
    if st.session_state.model is None:
        load_latest_model_if_available()
    
    if st.session_state.predictions is None:
        st.warning("يجب إجراء التنبؤات أولاً قبل تحليل الأقسام.")
        
        # If we have a model but no predictions, guide the user
        if st.session_state.model is not None:
            st.info("لديك نموذج جاهز للاستخدام. يرجى الانتقال إلى قسم 'التنبؤات' لتحميل بيانات الموظفين وإجراء التنبؤات أولاً.")
            if st.button("الانتقال إلى قسم التنبؤات", key="goto_predictions_from_department"):
                st.info("يرجى النقر على تبويب 'التنبؤات' أعلاه للمتابعة")
    
    elif st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Department selector
        st.subheader(t("select_department"))
        
        department = st.selectbox(
            t("department"),
            options=sorted(predictions['Department'].unique()),
            index=0
        )
        
        # Get department data
        dept_data = predictions[predictions['Department'] == department]
        
        # Calculate department metrics
        dept_metrics = calculate_department_metrics(dept_data)
        
        # Display department metrics
        st.subheader(t("department_metrics"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(t("total_employees"), dept_metrics['total_employees'])
        
        with col2:
            st.metric(t("high_risk_percentage"), f"{dept_metrics['high_risk_percentage']:.1%}")
        
        with col3:
            st.metric(t("avg_turnover_probability"), f"{dept_metrics['avg_probability']:.2f}")
        
        with col4:
            st.metric(t("avg_years_at_company"), f"{dept_metrics['avg_years']:.1f}")
            
        # Print report button
        if st.button("طباعة تقرير القسم", key="print_dept_report"):
            st.session_state.print_predictions = predictions
            js = f"""
            <script>
                window.open("/?view=print_report&department={department}", "_blank");
            </script>
            """
            st.components.v1.html(js, height=0)
        
        # Risk distribution by job title
        st.subheader(t("risk_by_job_title"))
        
        # Group by job title
        job_risk = dept_data.groupby('Job_Title')['Turnover_Probability'].mean().reset_index()
        job_risk = job_risk.sort_values('Turnover_Probability', ascending=False)
        
        fig = px.bar(
            job_risk,
            x='Job_Title',
            y='Turnover_Probability',
            color='Turnover_Probability',
            color_continuous_scale=['green', 'yellow', 'red'],
            labels={
                'Turnover_Probability': t("avg_turnover_probability"),
                'Job_Title': t("job_title")
            },
            title=t("turnover_probability_by_job_title")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # High risk employees in department
        st.subheader(t("high_risk_employees"))
        
        high_risk_employees = dept_data[dept_data['Risk_Category'] == 'High'].sort_values('Turnover_Probability', ascending=False)
        
        if len(high_risk_employees) > 0:
            st.dataframe(
                high_risk_employees[[
                    'Employee_ID', 'Job_Title', 'Turnover_Probability', 
                    'Performance_Score', 'Years_At_Company'
                ]],
                use_container_width=True
            )
        else:
            st.info(t("no_high_risk_employees"))
        
        # Department recommendations
        st.subheader(t("department_recommendations"))
        
        # Tabs for standard and AI recommendations
        dept_rec_tabs = st.tabs(["Standard Recommendations", "AI-Powered Analysis"])
        
        with dept_rec_tabs[0]:
            # Generate department-level recommendations
            if dept_metrics['high_risk_percentage'] > 0.3:
                st.error(t("critical_turnover_risk"))
                st.markdown(t("critical_risk_recommendations"))
            elif dept_metrics['high_risk_percentage'] > 0.15:
                st.warning(t("moderate_turnover_risk"))
                st.markdown(t("moderate_risk_recommendations"))
            else:
                st.success(t("low_turnover_risk"))
                st.markdown(t("low_risk_recommendations"))
                
        with dept_rec_tabs[1]:
            # Check if we have access to Anthropic Claude
            if 'ANTHROPIC_API_KEY' in os.environ or st.session_state.external_model == "Anthropic":
                with st.spinner("Analyzing department data with AI..."):
                    insights = analyze_department_trends(dept_data)
                    
                    if insights:
                        # Key insights
                        st.subheader("Key Insights")
                        for insight in insights.get('insights', []):
                            st.markdown(f"• {insight}")
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        for rec in insights.get('recommendations', []):
                            st.markdown(f"• {rec}")
                        
                        # Root causes
                        with st.expander("Potential Root Causes"):
                            for cause in insights.get('root_causes', []):
                                st.markdown(f"• {cause}")
                    else:
                        st.info("AI-powered department analysis could not be generated. Please check your Anthropic API key.")
            else:
                st.info("To enable AI-powered department analysis, please set up Anthropic API access in the Settings tab.")
                if st.button("Go to Settings", key="dept_settings_btn"):
                    # This is a workaround since streamlit doesn't support direct tab switching
                    st.session_state.active_tab = "settings"
                    st.rerun()
    
    else:
        st.warning(t("generate_predictions_first"))

# Tab 6: Visual Analytics
with tab6:
    st.header(t("visual_analytics_title"))
    
    # Try to load model automatically if not already loaded
    if st.session_state.model is None:
        load_latest_model_if_available()
    
    if st.session_state.predictions is None or st.session_state.data is None:
        st.warning("يجب إجراء التنبؤات أولاً قبل عرض التحليلات المرئية.")
        
        # If we have a model but no predictions, guide the user
        if st.session_state.model is not None:
            st.info("لديك نموذج جاهز للاستخدام. يرجى الانتقال إلى قسم 'التنبؤات' لتحميل بيانات الموظفين وإجراء التنبؤات أولاً.")
            if st.button("الانتقال إلى قسم التنبؤات", key="goto_predictions_from_visuals"):
                st.info("يرجى النقر على تبويب 'التنبؤات' أعلاه للمتابعة")
    
    elif st.session_state.predictions is not None and st.session_state.data is not None:
        data = st.session_state.data
        predictions = st.session_state.predictions
        
        # Visualization selector
        viz_type = st.selectbox(
            t("select_visualization"),
            options=[
                t("correlation_heatmap"),
                t("risk_factors_chart"),
                t("department_comparison_chart"),
                t("turnover_trends"),
                t("performance_vs_risk"),
                t("employee_clusters")
            ]
        )
        
        # Correlation Heatmap
        if viz_type == t("correlation_heatmap"):
            st.subheader(t("correlation_heatmap"))
            
            # Select only numeric columns for correlation analysis
            numeric_data = data.select_dtypes(include=['number'])
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Plot heatmap using plotly
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                title=t("correlation_heatmap"),
                labels=dict(color=t("correlation"))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Description and insights
            with st.expander("Insights and Interpretation"):
                st.write("""
                The correlation heatmap shows the relationship between different numeric variables:
                - Values close to 1 indicate strong positive correlation
                - Values close to -1 indicate strong negative correlation
                - Values close to 0 indicate little or no correlation
                
                Look for strong correlations with turnover-related metrics to identify potential risk factors.
                """)
        
        # Risk Factors Chart
        elif viz_type == t("risk_factors_chart"):
            st.subheader(t("risk_factors_chart"))
            
            # Get feature importance data if available
            if st.session_state.model is not None and st.session_state.feature_names is not None:
                feature_imp = feature_importance(
                    st.session_state.model, 
                    st.session_state.feature_names, 
                    st.session_state.model.__class__.__name__
                )
                
                # Plot feature importance
                fig = plot_feature_importance(
                    feature_imp, 
                    t("feature"), 
                    t("importance_score"), 
                    top_n=10
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add detailed insights for top features
                st.subheader("Key Risk Factor Insights")
                
                top_features = feature_imp.head(5)
                
                for _, row in top_features.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    with st.expander(f"{feature} (Score: {importance:.3f})"):
                        # Calculate average values for high and low risk groups
                        if feature in predictions.columns:
                            high_risk_avg = predictions[predictions['Risk_Category'] == 'High'][feature].mean()
                            low_risk_avg = predictions[predictions['Risk_Category'] == 'Low'][feature].mean()
                            
                            # Display comparison
                            cols = st.columns(2)
                            cols[0].metric("High Risk Avg", f"{high_risk_avg:.2f}")
                            cols[1].metric("Low Risk Avg", f"{low_risk_avg:.2f}")
                            
                            # Simple histogram to compare distributions
                            hist_fig = px.histogram(
                                predictions, 
                                x=feature, 
                                color="Risk_Category",
                                nbins=20,
                                barmode="overlay",
                                opacity=0.7,
                                color_discrete_map={"High": "#EF553B", "Medium": "#FFA15A", "Low": "#636EFA"}
                            )
                            st.plotly_chart(hist_fig, use_container_width=True)
            else:
                st.warning("Model needs to be trained first to view risk factors.")
        
        # Department Comparison Chart
        elif viz_type == t("department_comparison_chart"):
            st.subheader(t("department_comparison_chart"))
            
            # Calculate department-level metrics
            dept_metrics = []
            for dept in predictions['Department'].unique():
                dept_data = predictions[predictions['Department'] == dept]
                metrics = calculate_department_metrics(dept_data)
                metrics['Department'] = dept
                dept_metrics.append(metrics)
            
            dept_df = pd.DataFrame(dept_metrics)
            
            # Select metric to compare
            metric_options = {
                "avg_probability": t("avg_turnover_probability"),
                "high_risk_percentage": t("high_risk_percentage"),
                "avg_performance": t("avg_performance"),
                "avg_salary": t("avg_salary"),
                "avg_work_hours": t("avg_work_hours")
            }
            
            selected_metric = st.selectbox(
                "Select Metric to Compare",
                options=list(metric_options.keys()),
                format_func=lambda x: metric_options[x]
            )
            
            # Create bar chart
            fig = px.bar(
                dept_df.sort_values(selected_metric, ascending=False),
                x="Department",
                y=selected_metric,
                title=metric_options[selected_metric],
                color=selected_metric,
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table
            st.dataframe(
                dept_df[[
                    "Department", "total_employees", "avg_probability", 
                    "high_risk_percentage", "avg_performance"
                ]],
                use_container_width=True
            )
            
        # Performance vs Risk
        elif viz_type == t("performance_vs_risk"):
            st.subheader(t("performance_vs_risk"))
            
            # Create scatter plot of performance vs turnover risk
            fig = px.scatter(
                predictions,
                x="Performance_Score",
                y="Turnover_Probability",
                color="Department",
                size="Years_At_Company",
                hover_name="Employee_ID",
                hover_data=["Job_Title", "Monthly_Salary"],
                title="Performance Score vs Turnover Risk",
                labels={
                    "Performance_Score": t("performance_score"),
                    "Turnover_Probability": t("turnover_probability"),
                    "Years_At_Company": t("years_at_company")
                }
            )
            
            # Add horizontal lines for risk thresholds
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Low Risk")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis of performance vs risk
            with st.expander("Analysis"):
                st.write("""
                This visualization shows the relationship between employee performance and turnover risk:
                
                - **High-performing employees at risk**: Points in the upper-right quadrant represent high-performing employees who are at risk of leaving. These should be prioritized for retention efforts.
                
                - **Low-performing employees at low risk**: Points in the lower-left quadrant represent low-performing employees who are likely to stay. These may be candidates for performance improvement plans.
                
                - **Size of points**: Larger points represent employees with longer tenure at the company.
                """)
            
        # Employee Clusters
        elif viz_type == t("employee_clusters"):
            st.subheader(t("employee_clusters"))
            
            # Use a simplified clustering approach based on key metrics
            if 'Performance_Score' in predictions.columns and 'Turnover_Probability' in predictions.columns:
                # For demo purposes, create simple clusters
                predictions_copy = predictions.copy()
                
                # Define clusters
                def assign_cluster(row):
                    perf = row['Performance_Score']
                    risk = row['Turnover_Probability']
                    
                    if perf >= 4 and risk >= 0.5:
                        return "High Performers at Risk"
                    elif perf >= 4 and risk < 0.5:
                        return "Stable High Performers"
                    elif perf < 4 and risk >= 0.5:
                        return "Low Performers at Risk"
                    else:
                        return "Stable Low Performers"
                
                predictions_copy['Cluster'] = predictions_copy.apply(assign_cluster, axis=1)
                
                # Create visualization
                fig = px.scatter(
                    predictions_copy,
                    x="Performance_Score",
                    y="Turnover_Probability",
                    color="Cluster",
                    hover_name="Employee_ID",
                    hover_data=["Department", "Job_Title"],
                    title="Employee Clusters",
                    labels={
                        "Performance_Score": t("performance_score"),
                        "Turnover_Probability": t("turnover_probability")
                    },
                    color_discrete_map={
                        "High Performers at Risk": "#EF553B",
                        "Stable High Performers": "#636EFA",
                        "Low Performers at Risk": "#FFA15A",
                        "Stable Low Performers": "#FECB52"
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster statistics
                cluster_stats = predictions_copy.groupby('Cluster').agg({
                    'Employee_ID': 'count',
                    'Performance_Score': 'mean',
                    'Turnover_Probability': 'mean',
                    'Monthly_Salary': 'mean',
                    'Years_At_Company': 'mean'
                }).reset_index()
                
                cluster_stats.columns = ['Cluster', 'Count', 'Avg Performance', 'Avg Risk', 'Avg Salary', 'Avg Tenure']
                
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Recommendations for each cluster
                st.subheader("Cluster-Specific Recommendations")
                
                cluster_recs = {
                    "High Performers at Risk": "These are your most valuable employees who are at risk of leaving. Prioritize retention strategies such as competitive compensation packages, career advancement opportunities, and recognition programs.",
                    "Stable High Performers": "These employees are performing well and likely to stay. Focus on continued engagement, development opportunities, and succession planning to prepare them for future leadership roles.",
                    "Low Performers at Risk": "These employees may benefit from performance improvement plans. Evaluate whether to invest in their development or prepare for their potential departure.",
                    "Stable Low Performers": "These employees are not performing well but are likely to stay. Consider performance improvement plans, role reassignments, or evaluate whether they are in positions that match their skills."
                }
                
                for cluster, rec in cluster_recs.items():
                    with st.expander(f"Recommendations for {cluster}"):
                        st.write(rec)
                        
                        # Show example employees from this cluster
                        st.write("#### Example Employees")
                        sample = predictions_copy[predictions_copy['Cluster'] == cluster].head(3)
                        if len(sample) > 0:
                            st.dataframe(
                                sample[['Employee_ID', 'Department', 'Job_Title', 'Performance_Score', 'Turnover_Probability']],
                                use_container_width=True
                            )
                
        # Turnover Trends
        elif viz_type == t("turnover_trends"):
            st.subheader(t("turnover_trends"))
            
            st.info("This visualization would typically show turnover trends over time. For a complete implementation, historical data with timestamps would be required.")
            
            # Create a placeholder visualization using simulated data
            # In a real implementation, this would use actual historical data
            
            # Simulate monthly data for the last 12 months
            np.random.seed(42)  # For reproducibility
            
            months = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')
            turnover_rates = np.random.uniform(0.08, 0.15, 12)
            new_hires = np.random.randint(10, 30, 12)
            departures = np.random.randint(5, 25, 12)
            
            trend_data = pd.DataFrame({
                'Month': months,
                'Turnover_Rate': turnover_rates,
                'New_Hires': new_hires,
                'Departures': departures
            })
            
            # Plot turnover rate trend
            fig1 = px.line(
                trend_data,
                x='Month',
                y='Turnover_Rate',
                title='Monthly Turnover Rate (Simulated Data)',
                labels={'Turnover_Rate': 'Turnover Rate', 'Month': 'Month'},
            )
            
            fig1.update_traces(line=dict(color='#EF553B', width=3))
            fig1.update_layout(yaxis=dict(tickformat='.1%'))
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Plot hires vs departures
            fig2 = px.bar(
                trend_data,
                x='Month',
                y=['New_Hires', 'Departures'],
                title='New Hires vs Departures (Simulated Data)',
                barmode='group',
                labels={'value': 'Number of Employees', 'Month': 'Month', 'variable': 'Type'},
                color_discrete_map={'New_Hires': '#636EFA', 'Departures': '#EF553B'}
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Disclaimer about simulated data
            st.info("Note: The data shown above is simulated for demonstration purposes.")
            
            # Suggested implementation steps
            with st.expander("Implementation Notes"):
                st.write("""
                To implement actual turnover trend tracking:
                
                1. Store historical prediction data with timestamps
                2. Track actual employee departures
                3. Calculate monthly/quarterly turnover rates
                4. Compare predicted turnover with actual results
                """)
    else:
        st.warning(t("train_model_first"))

# Tab 7: Settings
with tab7:
    st.header(t("settings_page_title"))
    
    # Create tabs for different settings sections
    settings_tabs = st.tabs([
        t("ai_model_settings"),
        t("model_management"),
        t("recommendation_settings"),
        t("notification_settings")
    ])
    
    # AI Model Settings
    with settings_tabs[0]:
        st.subheader(t("ai_model_settings"))
        
        # External AI model options
        st.write("### " + t("select_external_model"))
        
        ai_model_option = st.selectbox(
            "AI Model Provider",
            options=["None", "OpenAI", "Anthropic", "Custom API"],
            index=0
        )
        
        if ai_model_option != "None":
            # API Key input
            api_key = st.text_input(
                t("api_key_label"), 
                value=st.session_state.api_key,
                type="password"
            )
            
            # API Endpoint (for custom API)
            if ai_model_option == "Custom API":
                api_endpoint = st.text_input(
                    t("api_endpoint"),
                    value=st.session_state.api_endpoint
                )
            
            # Save button
            if st.button(t("save_settings"), key="save_ai_settings"):
                st.session_state.external_model = ai_model_option
                st.session_state.api_key = api_key
                
                if ai_model_option == "Custom API":
                    st.session_state.api_endpoint = api_endpoint
                
                # Set environment variable for Anthropic API key
                if ai_model_option == "Anthropic" and api_key:
                    os.environ['ANTHROPIC_API_KEY'] = api_key
                    st.success("Anthropic API key set successfully. Advanced AI features are now enabled.")
                    
                st.success(t("settings_saved"))
                
                # If user selected Anthropic and didn't provide API key, ask for it
                if ai_model_option == "Anthropic" and not api_key:
                    st.warning("An Anthropic API key is required to use Claude. Please provide your API key.")
    
    # Model Management Settings
    with settings_tabs[1]:
        st.subheader("Model Management")
        
        # Get list of trained models
        trained_models = load_trained_models()
        
        # Display available models
        if trained_models and len(trained_models) > 0:
            st.write("### Available Trained Models")
            
            # Create a dataframe for better display
            models_df = pd.DataFrame({
                "ID": [m[0] for m in trained_models],
                "Name": [m[1] for m in trained_models],
                "Type": [m[2] for m in trained_models],
                "Created": [m[3] for m in trained_models],
                "Training Data Size": [m[4] if m[4] else "Unknown" for m in trained_models]
            })
            
            st.dataframe(models_df)
            
            # Model selection
            selected_model_id = st.selectbox(
                "Select model to manage",
                options=models_df["ID"].tolist(),
                format_func=lambda x: f"{models_df[models_df['ID']==x]['Name'].iloc[0]} ({models_df[models_df['ID']==x]['Type'].iloc[0]})"
            )
            
            # Model actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # View model details
                if st.button("View Model Details", key="view_model_details"):
                    model, preprocessor, feature_names, metrics, model_type = load_trained_model(selected_model_id)
                    
                    # Display metrics if available
                    if metrics:
                        st.subheader("Model Performance Metrics")
                        metric_cols = st.columns(5)
                        metrics_keys = ["accuracy", "precision", "recall", "f1", "auc"]
                        
                        for i, key in enumerate(metrics_keys):
                            if key in metrics:
                                metric_cols[i].metric(t(key), f"{metrics[key]:.2f}")
                    
                    # Feature importance if available
                    if model is not None and feature_names is not None:
                        st.subheader("Feature Importance")
                        feature_imp = feature_importance(model, feature_names, model_type)
                        fig = plot_feature_importance(feature_imp, "Feature", "Importance", top_n=10)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Load model into current session
                if st.button("Load Model", key="load_model_setting"):
                    model, preprocessor, feature_names, metrics, model_type = load_trained_model(selected_model_id)
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = feature_names
                    st.session_state.model_type = model_type
                    
                    st.success(f"Model '{models_df[models_df['ID']==selected_model_id]['Name'].iloc[0]}' loaded successfully!")
            
            with col3:
                # Delete model
                if st.button("Delete Model", key="delete_model"):
                    if st.session_state.model is not None and getattr(st.session_state, 'loaded_model_id', None) == selected_model_id:
                        st.error("Cannot delete the currently active model. Please load a different model first.")
                    else:
                        delete_trained_model(selected_model_id)
                        st.success(f"Model deleted successfully!")
                        st.rerun()
            
            # Model comparison
            st.write("### Compare Models")
            
            if len(trained_models) > 1:
                comparison_model_id = st.selectbox(
                    "Select model to compare with",
                    options=[m[0] for m in trained_models if m[0] != selected_model_id],
                    format_func=lambda x: f"{models_df[models_df['ID']==x]['Name'].iloc[0]} ({models_df[models_df['ID']==x]['Type'].iloc[0]})"
                )
                
                if st.button("Compare Models"):
                    # Load both models' metrics
                    _, _, _, metrics1, model_type1 = load_trained_model(selected_model_id)
                    _, _, _, metrics2, model_type2 = load_trained_model(comparison_model_id)
                    
                    if metrics1 and metrics2:
                        # Create comparison dataframe
                        model1_name = models_df[models_df['ID']==selected_model_id]['Name'].iloc[0]
                        model2_name = models_df[models_df['ID']==comparison_model_id]['Name'].iloc[0]
                        
                        comparison_data = {
                            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
                            model1_name: [
                                metrics1.get("accuracy", "N/A"),
                                metrics1.get("precision", "N/A"),
                                metrics1.get("recall", "N/A"),
                                metrics1.get("f1", "N/A"),
                                metrics1.get("auc", "N/A")
                            ],
                            model2_name: [
                                metrics2.get("accuracy", "N/A"),
                                metrics2.get("precision", "N/A"),
                                metrics2.get("recall", "N/A"),
                                metrics2.get("f1", "N/A"),
                                metrics2.get("auc", "N/A")
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        
                        # Visual comparison
                        metrics_to_compare = ["accuracy", "precision", "recall", "f1", "auc"]
                        chart_data = []
                        
                        for metric in metrics_to_compare:
                            if metric in metrics1 and metric in metrics2:
                                chart_data.append({
                                    "Metric": metric.capitalize(),
                                    model1_name: metrics1[metric],
                                    model2_name: metrics2[metric]
                                })
                        
                        if chart_data:
                            chart_df = pd.DataFrame(chart_data)
                            
                            # Reshape for plotting
                            chart_df_long = pd.melt(
                                chart_df, 
                                id_vars=["Metric"], 
                                value_vars=[model1_name, model2_name],
                                var_name="Model", 
                                value_name="Score"
                            )
                            
                            fig = px.bar(
                                chart_df_long,
                                x="Metric",
                                y="Score",
                                color="Model",
                                barmode="group",
                                title="Model Performance Comparison"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("One or both models don't have performance metrics stored.")
            else:
                st.info("Need at least two models to perform comparison.")
        else:
            st.info("No trained models available. Go to the Model Training tab to train and save models.")
    
    # Recommendation Settings
    with settings_tabs[2]:
        st.subheader(t("recommendation_settings"))
        
        st.write("### " + t("customize_recommendations"))
        
        # Display existing custom recommendations
        if st.session_state.custom_recommendations:
            for i, rec in enumerate(st.session_state.custom_recommendations):
                with st.expander(f"{rec['title']} ({rec['category']})"):
                    st.write(f"**Description:** {rec['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(t("edit_recommendation"), key=f"edit_rec_{i}"):
                            st.session_state.edit_rec_index = i
                    with col2:
                        if st.button(t("delete_recommendation"), key=f"delete_rec_{i}"):
                            st.session_state.custom_recommendations.pop(i)
                            st.rerun()
        else:
            st.info("No custom recommendations added yet.")
        
        # Add new recommendation
        st.write("### " + t("add_recommendation"))
        
        # Form for adding/editing recommendations
        rec_title = st.text_input("Title", key="rec_title")
        rec_category = st.selectbox(
            "Category",
            options=["Compensation", "Work-Life Balance", "Career Development", "Management", "Culture"],
            key="rec_category"
        )
        rec_description = st.text_area("Description", key="rec_description")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("add_recommendation"), key="add_rec_btn"):
                if rec_title and rec_description:
                    new_rec = {
                        "title": rec_title,
                        "category": rec_category,
                        "description": rec_description
                    }
                    
                    # If editing, replace existing recommendation
                    if hasattr(st.session_state, 'edit_rec_index'):
                        st.session_state.custom_recommendations[st.session_state.edit_rec_index] = new_rec
                        delattr(st.session_state, 'edit_rec_index')
                    else:
                        # Otherwise add new recommendation
                        if 'custom_recommendations' not in st.session_state:
                            st.session_state.custom_recommendations = []
                        st.session_state.custom_recommendations.append(new_rec)
                    
                    st.success("Recommendation saved successfully")
                    st.rerun()
                else:
                    st.error("Please provide a title and description")
    
    # Notification Settings
    with settings_tabs[3]:
        st.subheader(t("notification_settings"))
        
        # Enable/disable notifications
        enable_notifications = st.toggle(
            t("enable_notifications"),
            value=st.session_state.enable_notifications
        )
        
        # Notification threshold
        notification_threshold = st.slider(
            t("notification_threshold"),
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.notification_threshold,
            step=0.05
        )
        
        # Email notifications (UI only, not functional)
        st.write("### " + t("email_notifications"))
        
        email_address = st.text_input("Email Address")
        email_frequency = st.selectbox(
            "Frequency",
            options=["Daily", "Weekly", "Monthly", "Real-time"]
        )
        
        # Save notification settings
        if st.button(t("save_settings"), key="save_notification_settings"):
            st.session_state.enable_notifications = enable_notifications
            st.session_state.notification_threshold = notification_threshold
            st.success(t("settings_saved"))

# Tab 8: Notifications
with tab8:
    st.header(t("notifications_title"))
    
    # Reset unread notifications when visiting this tab
    st.session_state.unread_notifications = 0
    
    # Create notifications if we have predictions but don't have notifications yet
    if st.session_state.predictions is not None and len(st.session_state.notifications) == 0:
        # Create notifications for high-risk employees
        high_risk_employees = st.session_state.predictions[st.session_state.predictions['Risk_Category'] == 'High']
        
        if len(high_risk_employees) > 0:
            for _, employee in high_risk_employees.head(min(5, len(high_risk_employees))).iterrows():
                notification = {
                    "employee_id": employee['Employee_ID'],
                    "message": f"{t('notification_employee_risk')}: {employee['Turnover_Probability']:.1%}",
                    "department": employee['Department'],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "read": False
                }
                st.session_state.notifications.append(notification)
                st.session_state.unread_notifications += 1
    
    # Display notifications
    if st.session_state.notifications:
        # Notification filters
        col1, col2 = st.columns(2)
        with col1:
            notification_type = st.radio(
                "Filter",
                options=[t("all_notifications"), t("new_notifications")],
                horizontal=True
            )
        
        with col2:
            department_filter = st.multiselect(
                t("filter_by_department"),
                options=sorted(set(n['department'] for n in st.session_state.notifications))
            )
        
        # Filter notifications
        filtered_notifications = st.session_state.notifications
        if notification_type == t("new_notifications"):
            filtered_notifications = [n for n in filtered_notifications if not n['read']]
        if department_filter:
            filtered_notifications = [n for n in filtered_notifications if n['department'] in department_filter]
        
        # Display notifications
        for i, notification in enumerate(filtered_notifications):
            with st.container():
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    st.write(notification['date'])
                with cols[1]:
                    st.write(f"**{notification['employee_id']}** ({notification['department']}): {notification['message']}")
                with cols[2]:
                    if not notification['read']:
                        if st.button(t("mark_as_read"), key=f"read_{i}"):
                            notification['read'] = True
                            st.rerun()
                
                st.divider()
        
        # Mark all as read button
        unread = sum(1 for n in st.session_state.notifications if not n['read'])
        if unread > 0:
            if st.button(f"{t('mark_as_read')} ({unread})"):
                for notification in st.session_state.notifications:
                    notification['read'] = True
                st.rerun()
    else:
        st.info(t("no_notifications"))

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center;'>{t('footer_text')} | {datetime.now().year}</div>", 
    unsafe_allow_html=True
)
