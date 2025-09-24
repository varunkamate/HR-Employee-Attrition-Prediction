# HR-Employee-Attrition-Prediction

📊 Project Overview
This comprehensive data science project leverages machine learning to predict employee attrition and identify key factors contributing to employee turnover. Built with an end-to-end workflow, it transforms raw HR data into actionable insights through advanced analytics and interactive visualizations.

🎯 Business Impact

Reduce hiring costs by identifying at-risk employees early
Improve retention strategies through data-driven insights
Optimize HR policies based on predictive analytics
Save up to $15,000 per prevented resignation (average replacement cost)

✨ Key Features
<table>
<tr>
<td width="50%">
🔍 Comprehensive Analytics

Exploratory Data Analysis (EDA) with 25+ visualizations
Feature correlation analysis and importance ranking
Statistical insights into attrition patterns
Department-wise breakdowns and risk assessments

</td>
<td width="50%">
🤖 Machine Learning Pipeline

Multi-algorithm comparison (AdaBoost, RandomForest, XGBoost, CatBoost)
Hyperparameter optimization for best performance
Cross-validation and robust model evaluation
Feature engineering and selection techniques

</td>
</tr>
<tr>
<td width="50%">

🖥️ Interactive Web Application

Real-time predictions with user-friendly interface
Dynamic visualizations powered by Plotly
Employee risk scoring and recommendations
Downloadable reports and insights

</td>
<td width="50%">
📈 Performance Metrics

ROC-AUC Score: 0.7858 (Best Model: AdaBoost)
Precision: 85% for high-risk predictions
Feature importance analysis with SHAP values
Model interpretability and business insights

</td>
</tr>
</table>

🚀 Quick Start
Prerequisites
bashPython 3.8+
pip (Python package manager)
🔧 Installation

Clone the repository

bash   git clone https://github.com/yourusername/hr-employee-attrition-prediction.git
   cd hr-employee-attrition-prediction

Create virtual environment

bash   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bash   pip install -r requirements.txt

Launch the application

bash   streamlit run app.py


📁 Repository Structure

hr-employee-attrition-prediction/
├── 📊 data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Original dataset
│   ├── processed_data.csv                        # Cleaned data
│   └── data_dictionary.md                        # Feature descriptions
├── 📓 notebooks/
│   ├── 01_exploratory_data_analysis.ipynb        # EDA and insights
│   ├── 02_feature_engineering.ipynb              # Feature creation
│   ├── 03_model_training.ipynb                   # ML model development
│   └── 04_model_evaluation.ipynb                 # Performance analysis
├── 🤖 models/
│   ├── best_attrition_model.pkl                  # Trained AdaBoost model
│   ├── model_pipeline.pkl                        # Complete ML pipeline
│   └── feature_importance.json                   # Feature rankings
├── 📱 src/
│   ├── app.py                                     # Streamlit web application
│   ├── data_preprocessing.py                     # Data cleaning utilities
│   ├── model_training.py                         # ML training scripts
│   └── visualization.py                          # Plotting functions
├── 📊 reports/
│   ├── model_performance_report.pdf              # Detailed analysis
│   ├── business_insights.md                      # Key findings
│   └── presentation.pptx                         # Executive summary
├── 🧪 tests/
│   ├── test_data_processing.py                   # Data tests
│   ├── test_model_performance.py                 # Model tests
│   └── test_app_functionality.py                 # App tests
├── 📋 requirements.txt                           # Python dependencies
├── 🐳 Dockerfile                                 # Container configuration
├── ⚙️ .github/workflows/ci.yml                   # CI/CD pipeline
└── 📖 README.md                                  # This file


🔬 Methodology
1. Data Exploration & Analysis

Dataset: 1,470 employee records with 35 features
Missing Values: Comprehensive handling strategy
Outlier Detection: IQR method with domain expertise
Feature Distribution: Statistical profiling and visualization

2. Feature Engineering

Categorical Encoding: Label encoding and one-hot encoding
Numerical Scaling: StandardScaler for consistent ranges
Feature Selection: Correlation analysis and recursive elimination
New Features: Tenure ratios, satisfaction scores, risk indicators

3. Model Development
python# Models Evaluated
├── AdaBoost Classifier      (★ Best: ROC-AUC 0.7858)
├── Random Forest           (ROC-AUC: 0.7654)
├── XGBoost                 (ROC-AUC: 0.7701)
├── CatBoost                (ROC-AUC: 0.7598)
├── Logistic Regression     (ROC-AUC: 0.7234)
└── SVM                     (ROC-AUC: 0.7189)
4. Model Evaluation

Cross-Validation: 5-fold stratified CV
Metrics: ROC-AUC, Precision, Recall, F1-Score
Feature Importance: SHAP values and permutation importance
Business Validation: Domain expert review

📊 Key Insights
🔍 Top Attrition Factors

Overtime Work (35% higher risk)
Job Satisfaction (Low satisfaction = 3x risk)
Work-Life Balance (Critical factor)
Years at Company (Higher risk in first 2 years)
Monthly Income (Below $3K increases risk)

💼 Department Analysis
DepartmentAttrition RateRisk LevelSales20.6%🔴 HighHuman Resources19.0%🟡 MediumResearch & Development13.8%🟢 Low
🎯 Actionable Recommendations

Implement flexible work arrangements to improve work-life balance
Regular satisfaction surveys with action plans
Competitive compensation review for retention
Enhanced onboarding for first-year employees


🖥️ Web Application Features
📱 Interactive Dashboard

Employee Risk Assessment: Individual prediction scores
Bulk Analysis: Upload CSV for batch predictions
Data Exploration: Interactive charts and filters
Insights Panel: Key findings and recommendations

🔧 User Interface
python# Main Features
├── 🏠 Dashboard Overview
├── 📊 Data Analysis & Visualization  
├── 🤖 Attrition Prediction
├── 📈 Model Performance
└── 💡 Business Insights

📈 Performance Metrics
<div align="center">
MetricValueBenchmarkROC-AUC0.7858> 0.75 ✅Precision0.8462> 0.80 ✅Recall0.6429> 0.60 ✅F1-Score0.7312> 0.70 ✅Accuracy0.8776> 0.85 ✅
</div>

