import pandas as pd
import numpy as np
import random
from collections import defaultdict

class RecommendationGenerator:
    """
    Class to generate recommendations for employee retention based on risk factors
    """
    
    def __init__(self, language='en'):
        """
        Initialize the recommendation generator
        
        Args:
            language (str): Language for recommendations ('en' or 'ar')
        """
        self.language = language
        self.recommendation_templates = self._initialize_templates()
    
    def _initialize_templates(self):
        """
        Initialize recommendation templates for different risk factors
        
        Returns:
            dict: Dictionary mapping risk factors to recommendation templates
        """
        templates = {
            # Performance-related recommendations
            'low_performance': {
                'en': [
                    "Implement a performance improvement plan for {employee_name}",
                    "Schedule regular check-ins to provide feedback and support",
                    "Identify specific training needs to improve {employee_name}'s performance",
                    "Pair with a high-performing mentor for skill development",
                    "Set clear, achievable goals with regular milestones"
                ],
                'ar': [
                    "تنفيذ خطة تحسين الأداء لـ {employee_name}",
                    "جدولة اجتماعات منتظمة لتقديم التغذية الراجعة والدعم",
                    "تحديد احتياجات التدريب المحددة لتحسين أداء {employee_name}",
                    "الإقران مع مرشد عالي الأداء لتطوير المهارات",
                    "وضع أهداف واضحة وقابلة للتحقيق مع معالم منتظمة"
                ]
            },
            
            # Overtime-related recommendations
            'high_overtime': {
                'en': [
                    "Review workload distribution to reduce excessive overtime",
                    "Consider temporary staff augmentation to balance team workload",
                    "Implement a maximum overtime policy to prevent burnout",
                    "Provide additional compensation or time off for excessive overtime periods",
                    "Conduct workload analysis to identify efficiency improvements"
                ],
                'ar': [
                    "مراجعة توزيع عبء العمل لتقليل العمل الإضافي المفرط",
                    "النظر في تعزيز الموظفين المؤقتين لموازنة عبء عمل الفريق",
                    "تنفيذ سياسة الحد الأقصى للعمل الإضافي لمنع الإرهاق",
                    "تقديم تعويض إضافي أو إجازة لفترات العمل الإضافي المفرطة",
                    "إجراء تحليل لعبء العمل لتحديد تحسينات الكفاءة"
                ]
            },
            
            # Salary-related recommendations
            'low_salary': {
                'en': [
                    "Conduct a compensation review to ensure market competitiveness",
                    "Consider a salary adjustment based on performance and market benchmarks",
                    "Implement a performance-based bonus program",
                    "Develop a clear path for salary progression tied to achievements",
                    "Offer additional non-monetary benefits to improve total compensation package"
                ],
                'ar': [
                    "إجراء مراجعة للتعويضات لضمان القدرة التنافسية في السوق",
                    "النظر في تعديل الراتب بناءً على الأداء ومعايير السوق",
                    "تنفيذ برنامج مكافآت على أساس الأداء",
                    "تطوير مسار واضح لتقدم الراتب مرتبط بالإنجازات",
                    "تقديم مزايا إضافية غير نقدية لتحسين حزمة التعويض الإجمالية"
                ]
            },
            
            # Promotion-related recommendations
            'no_promotions': {
                'en': [
                    "Create a clear career development plan with defined milestones",
                    "Discuss long-term career aspirations and align with organizational opportunities",
                    "Provide opportunities for skill development in preparation for advancement",
                    "Identify specific leadership or technical training needed for promotion",
                    "Consider lateral moves to gain broader experience if vertical promotion isn't immediately available"
                ],
                'ar': [
                    "إنشاء خطة تطوير وظيفي واضحة مع معالم محددة",
                    "مناقشة طموحات المهنة على المدى الطويل ومواءمتها مع الفرص التنظيمية",
                    "توفير فرص لتطوير المهارات استعدادًا للتقدم",
                    "تحديد التدريب القيادي أو التقني المحدد اللازم للترقية",
                    "النظر في التحركات الجانبية لاكتساب خبرة أوسع إذا لم تكن الترقية العمودية متاحة على الفور"
                ]
            },
            
            # Training-related recommendations
            'low_training': {
                'en': [
                    "Develop a personalized training plan addressing specific skill gaps",
                    "Allocate dedicated time for professional development activities",
                    "Provide access to online learning platforms for self-paced development",
                    "Encourage participation in industry conferences or workshops",
                    "Implement a learning budget for courses or certifications of interest"
                ],
                'ar': [
                    "تطوير خطة تدريب شخصية تعالج فجوات المهارات المحددة",
                    "تخصيص وقت مخصص لأنشطة التطوير المهني",
                    "توفير الوصول إلى منصات التعلم عبر الإنترنت للتطوير الذاتي",
                    "تشجيع المشاركة في المؤتمرات أو ورش العمل الصناعية",
                    "تنفيذ ميزانية تعليمية للدورات أو الشهادات ذات الاهتمام"
                ]
            },
            
            # Satisfaction-related recommendations
            'low_satisfaction': {
                'en': [
                    "Conduct a one-on-one meeting to identify specific sources of dissatisfaction",
                    "Implement regular pulse surveys to track satisfaction trends",
                    "Address work-life balance concerns through flexible scheduling options",
                    "Improve workplace environment or resources based on feedback",
                    "Recognize achievements more frequently to improve engagement"
                ],
                'ar': [
                    "إجراء اجتماع فردي لتحديد مصادر محددة لعدم الرضا",
                    "تنفيذ استطلاعات نبض منتظمة لتتبع اتجاهات الرضا",
                    "معالجة مخاوف التوازن بين العمل والحياة من خلال خيارات الجدولة المرنة",
                    "تحسين بيئة العمل أو الموارد بناءً على التعليقات",
                    "الاعتراف بالإنجازات بشكل أكثر تكرارًا لتحسين المشاركة"
                ]
            },
            
            # Work hours-related recommendations
            'excessive_hours': {
                'en': [
                    "Review workload and consider redistribution among team members",
                    "Implement time management training to improve efficiency",
                    "Encourage use of vacation time to prevent burnout",
                    "Consider flexible working hours or partial remote work options",
                    "Automate routine tasks to reduce time spent on administrative work"
                ],
                'ar': [
                    "مراجعة عبء العمل والنظر في إعادة التوزيع بين أعضاء الفريق",
                    "تنفيذ تدريب إدارة الوقت لتحسين الكفاءة",
                    "تشجيع استخدام وقت الإجازة لمنع الإرهاق",
                    "النظر في ساعات عمل مرنة أو خيارات عمل عن بعد جزئي",
                    "أتمتة المهام الروتينية لتقليل الوقت المستغرق في العمل الإداري"
                ]
            },
            
            # Remote work-related recommendations
            'no_remote_work': {
                'en': [
                    "Implement a hybrid work model allowing partial remote work",
                    "Develop clear remote work policies and productivity expectations",
                    "Provide necessary equipment for effective remote work",
                    "Schedule regular team-building activities for remote workers",
                    "Create a communication framework for effective remote collaboration"
                ],
                'ar': [
                    "تنفيذ نموذج عمل هجين يسمح بالعمل عن بعد الجزئي",
                    "تطوير سياسات واضحة للعمل عن بعد وتوقعات الإنتاجية",
                    "توفير المعدات اللازمة للعمل الفعال عن بعد",
                    "جدولة أنشطة منتظمة لبناء الفريق للعاملين عن بعد",
                    "إنشاء إطار اتصال للتعاون الفعال عن بعد"
                ]
            },
            
            # Department-specific recommendations
            'department_it': {
                'en': [
                    "Provide access to the latest technology and tools",
                    "Implement a technical skills development program",
                    "Create opportunities for innovation and creative problem-solving",
                    "Establish a competitive technical career ladder",
                    "Support participation in technical communities and open source projects"
                ],
                'ar': [
                    "توفير الوصول إلى أحدث التقنيات والأدوات",
                    "تنفيذ برنامج تطوير المهارات التقنية",
                    "خلق فرص للابتكار وحل المشكلات بشكل خلاق",
                    "إنشاء سلم وظيفي تقني تنافسي",
                    "دعم المشاركة في المجتمعات التقنية ومشاريع المصدر المفتوح"
                ]
            },
            
            'department_sales': {
                'en': [
                    "Review and optimize the commission structure",
                    "Provide advanced sales training and techniques",
                    "Implement a customer relationship management system",
                    "Create team-based incentives to foster collaboration",
                    "Recognize top performers through a sales achievement program"
                ],
                'ar': [
                    "مراجعة وتحسين هيكل العمولة",
                    "تقديم تدريب وتقنيات مبيعات متقدمة",
                    "تنفيذ نظام إدارة علاقات العملاء",
                    "إنشاء حوافز قائمة على الفريق لتعزيز التعاون",
                    "التعرف على أفضل الأداء من خلال برنامج إنجازات المبيعات"
                ]
            },
            
            'department_hr': {
                'en': [
                    "Provide opportunities to implement innovative HR practices",
                    "Support professional HR certifications and continuing education",
                    "Involve in strategic organizational decisions",
                    "Implement HR technology to streamline administrative tasks",
                    "Create opportunities to directly impact company culture and employee experience"
                ],
                'ar': [
                    "توفير فرص لتنفيذ ممارسات مبتكرة للموارد البشرية",
                    "دعم شهادات الموارد البشرية المهنية والتعليم المستمر",
                    "المشاركة في القرارات التنظيمية الاستراتيجية",
                    "تنفيذ تكنولوجيا الموارد البشرية لتبسيط المهام الإدارية",
                    "خلق فرص للتأثير المباشر على ثقافة الشركة وتجربة الموظف"
                ]
            },
            
            # General recommendations
            'general': {
                'en': [
                    "Schedule regular career development discussions",
                    "Provide opportunities for cross-functional project participation",
                    "Implement a formal recognition program for achievements",
                    "Conduct stay interviews to proactively address concerns",
                    "Improve internal communication about growth opportunities"
                ],
                'ar': [
                    "جدولة مناقشات منتظمة للتطوير المهني",
                    "توفير فرص للمشاركة في المشاريع متعددة الوظائف",
                    "تنفيذ برنامج رسمي للاعتراف بالإنجازات",
                    "إجراء مقابلات البقاء لمعالجة المخاوف بشكل استباقي",
                    "تحسين الاتصال الداخلي حول فرص النمو"
                ]
            }
        }
        
        return templates
    
    def generate_individual_recommendations(self, employee_data, shap_values, feature_names, num_recommendations=3, employee_name=None):
        """
        Generate personalized recommendations for an individual employee
        
        Args:
            employee_data (pd.Series): Employee data
            shap_values (np.ndarray): SHAP values for the employee
            feature_names (list): List of feature names
            num_recommendations (int): Number of recommendations to generate
            employee_name (str): Name of the employee (optional)
            
        Returns:
            list: List of personalized recommendations
        """
        # Default employee name if not provided
        if employee_name is None:
            employee_name = f"Employee {employee_data.name}" if hasattr(employee_data, 'name') else "the employee"
        
        # Map SHAP values to features
        feature_impact = dict(zip(feature_names, shap_values))
        
        # Identify risk categories based on feature values and impacts
        risk_categories = []
        
        # Check performance
        performance_features = [f for f in feature_names if 'performance' in f.lower()]
        if performance_features and any(feature_impact[f] > 0 for f in performance_features):
            risk_categories.append('low_performance')
        
        # Check overtime
        overtime_features = [f for f in feature_names if 'overtime' in f.lower()]
        if overtime_features and any(feature_impact[f] > 0 for f in overtime_features):
            risk_categories.append('high_overtime')
        
        # Check salary
        salary_features = [f for f in feature_names if 'salary' in f.lower()]
        if salary_features and any(feature_impact[f] > 0 for f in salary_features):
            risk_categories.append('low_salary')
        
        # Check promotions
        promotion_features = [f for f in feature_names if 'promotion' in f.lower()]
        if promotion_features and any(feature_impact[f] > 0 for f in promotion_features):
            risk_categories.append('no_promotions')
        
        # Check training
        training_features = [f for f in feature_names if 'training' in f.lower()]
        if training_features and any(feature_impact[f] > 0 for f in training_features):
            risk_categories.append('low_training')
        
        # Check satisfaction
        satisfaction_features = [f for f in feature_names if 'satisfaction' in f.lower()]
        if satisfaction_features and any(feature_impact[f] > 0 for f in satisfaction_features):
            risk_categories.append('low_satisfaction')
        
        # Check work hours
        hours_features = [f for f in feature_names if 'hours' in f.lower() and 'overtime' not in f.lower()]
        if hours_features and any(feature_impact[f] > 0 for f in hours_features):
            risk_categories.append('excessive_hours')
        
        # Check remote work
        remote_features = [f for f in feature_names if 'remote' in f.lower()]
        if remote_features and any(feature_impact[f] > 0 for f in remote_features):
            risk_categories.append('no_remote_work')
        
        # Check department-specific issues
        department_feature = next((f for f in feature_names if 'department' in f.lower()), None)
        if department_feature:
            if 'IT' in department_feature or 'technology' in department_feature.lower():
                risk_categories.append('department_it')
            elif 'Sales' in department_feature:
                risk_categories.append('department_sales')
            elif 'HR' in department_feature or 'human resources' in department_feature.lower():
                risk_categories.append('department_hr')
        
        # Always include general recommendations as a fallback
        risk_categories.append('general')
        
        # Generate recommendations based on identified risk categories
        all_recommendations = []
        for category in risk_categories:
            templates = self.recommendation_templates.get(category, self.recommendation_templates['general'])
            templates = templates.get(self.language, templates['en'])  # Fallback to English if language not available
            all_recommendations.extend(templates)
        
        # Fill in employee name in templates
        all_recommendations = [rec.format(employee_name=employee_name) for rec in all_recommendations]
        
        # Remove duplicates and shuffle
        all_recommendations = list(set(all_recommendations))
        random.shuffle(all_recommendations)
        
        # Return requested number of recommendations
        return all_recommendations[:num_recommendations]
    
    def generate_department_recommendations(self, department_data, risk_column='risk_probability', department_name=None):
        """
        Generate recommendations for a department based on aggregate risk factors
        
        Args:
            department_data (pd.DataFrame): DataFrame containing department employees
            risk_column (str): Column name containing risk probabilities
            department_name (str): Name of the department (optional)
            
        Returns:
            list: List of department-level recommendations
        """
        if department_name is None:
            department_name = department_data['Department'].iloc[0] if 'Department' in department_data.columns else "the department"
        
        # Calculate department metrics
        avg_risk = department_data[risk_column].mean()
        high_risk_count = (department_data[risk_column] >= 0.7).sum()
        high_risk_pct = high_risk_count / len(department_data) * 100
        
        recommendations = []
        
        # High-level risk recommendations
        if high_risk_pct >= 30:
            # Critical situation
            recommendations.append({
                'en': f"URGENT: {department_name} has a critical turnover risk ({high_risk_pct:.1f}%). Immediate intervention required.",
                'ar': f"عاجل: {department_name} لديه خطر دوران حرج ({high_risk_pct:.1f}٪). مطلوب تدخل فوري."
            })
            
            recommendations.append({
                'en': f"Conduct a department-wide assessment to identify systemic issues affecting {high_risk_count} high-risk employees.",
                'ar': f"إجراء تقييم على مستوى القسم لتحديد المشكلات المنهجية التي تؤثر على {high_risk_count} موظفين معرضين للخطر."
            })
            
            recommendations.append({
                'en': "Develop a comprehensive retention plan with executive sponsorship and additional resources.",
                'ar': "تطوير خطة شاملة للاحتفاظ بالموظفين مع رعاية تنفيذية وموارد إضافية."
            })
            
        elif high_risk_pct >= 15:
            # Moderate risk
            recommendations.append({
                'en': f"{department_name} has a concerning turnover risk ({high_risk_pct:.1f}%). Action recommended.",
                'ar': f"{department_name} لديه خطر دوران مثير للقلق ({high_risk_pct:.1f}٪). يوصى باتخاذ إجراء."
            })
            
            recommendations.append({
                'en': "Schedule targeted interviews with high-risk employees to identify common concerns.",
                'ar': "جدولة مقابلات مستهدفة مع الموظفين ذوي المخاطر العالية لتحديد المخاوف المشتركة."
            })
            
            recommendations.append({
                'en': "Provide managers with additional resources for retention conversations.",
                'ar': "تزويد المديرين بموارد إضافية لإجراء محادثات الاحتفاظ."
            })
            
        else:
            # Low risk
            recommendations.append({
                'en': f"{department_name} has a manageable turnover risk ({high_risk_pct:.1f}%). Continue monitoring.",
                'ar': f"{department_name} لديه خطر دوران قابل للإدارة ({high_risk_pct:.1f}٪). استمر في المراقبة."
            })
            
            recommendations.append({
                'en': "Implement periodic check-ins with identified high-risk individuals.",
                'ar': "تنفيذ عمليات تحقق دورية مع الأفراد ذوي المخاطر العالية المحددة."
            })
        
        # Look for patterns in department data
        if 'Performance_Score' in department_data.columns:
            avg_performance = department_data['Performance_Score'].mean()
            if avg_performance < 3:
                recommendations.append({
                    'en': f"Address below-average performance scores in {department_name} through targeted training and coaching.",
                    'ar': f"معالجة درجات الأداء دون المتوسط في {department_name} من خلال التدريب والتوجيه المستهدف."
                })
        
        if 'Training_Hours' in department_data.columns:
            avg_training = department_data['Training_Hours'].mean()
            if avg_training < 20:
                recommendations.append({
                    'en': f"Increase training investment for {department_name}. Current average ({avg_training:.1f} hours) is below target.",
                    'ar': f"زيادة استثمار التدريب لـ {department_name}. المتوسط الحالي ({avg_training:.1f} ساعة) أقل من الهدف."
                })
        
        if 'Employee_Satisfaction_Score' in department_data.columns:
            avg_satisfaction = department_data['Employee_Satisfaction_Score'].mean()
            if avg_satisfaction < 3:
                recommendations.append({
                    'en': f"Conduct engagement workshops to address low satisfaction scores in {department_name}.",
                    'ar': f"إجراء ورش عمل للمشاركة لمعالجة درجات الرضا المنخفضة في {department_name}."
                })
        
        # Department-specific recommendations
        dept_key = None
        if 'IT' in department_name:
            dept_key = 'department_it'
        elif 'Sales' in department_name:
            dept_key = 'department_sales'
        elif 'HR' in department_name:
            dept_key = 'department_hr'
        
        if dept_key and self.recommendation_templates.get(dept_key):
            dept_templates = self.recommendation_templates[dept_key][self.language]
            recommendations.append({
                'en': random.choice(self.recommendation_templates[dept_key]['en']),
                'ar': random.choice(self.recommendation_templates[dept_key]['ar'])
            })
        
        # Return recommendations in current language
        return [rec[self.language] for rec in recommendations]
    
    def set_language(self, language):
        """
        Set the language for recommendations
        
        Args:
            language (str): Language code ('en' or 'ar')
        """
        if language in ['en', 'ar']:
            self.language = language
