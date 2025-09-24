import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3d59, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .safe-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .risk-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

def load_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'Age': [25, 35, 45, 28, 52, 33],
        'DailyRate': [1102, 279, 1373, 1392, 591, 1005],
        'DistanceFromHome': [1, 8, 2, 3, 14, 10],
        'Education': [2, 4, 1, 4, 3, 3],
        'EnvironmentSatisfaction': [2, 3, 4, 4, 1, 4],
        'HourlyRate': [94, 61, 92, 56, 40, 89],
        'JobInvolvement': [3, 2, 2, 3, 2, 3],
        'JobLevel': [2, 1, 1, 1, 5, 3],
        'JobSatisfaction': [4, 2, 3, 3, 2, 4],
        'MonthlyIncome': [5993, 2090, 2909, 3468, 17048, 7298],
        'MonthlyRate': [19479, 24907, 2396, 23159, 16632, 11691],
        'NumCompaniesWorked': [8, 1, 6, 1, 4, 1],
        'OverTime': [1, 0, 0, 1, 0, 0],
        'PercentSalaryHike': [11, 23, 15, 11, 12, 13],
        'PerformanceRating': [3, 4, 3, 3, 3, 3],
        'RelationshipSatisfaction': [1, 4, 2, 3, 3, 3],
        'StockOptionLevel': [0, 1, 0, 0, 1, 0],
        'TotalWorkingYears': [8, 10, 7, 8, 26, 12],
        'TrainingTimesLastYear': [0, 3, 3, 3, 3, 2],
        'WorkLifeBalance': [1, 3, 3, 3, 2, 2],
        'YearsAtCompany': [6, 10, 0, 8, 17, 1],
        'YearsInCurrentRole': [4, 7, 0, 7, 7, 0],
        'YearsSinceLastPromotion': [0, 1, 0, 3, 3, 0],
        'YearsWithCurrManager': [5, 7, 0, 0, 9, 0],
        'Attrition': [1, 0, 1, 0, 0, 1]  # 1 = Yes, 0 = No
    }
    return pd.DataFrame(sample_data)

def create_model():
    """Create and train a Random Forest model with sample data"""
    df = load_sample_data()
    
    # Prepare features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist()

def main():
    # Header
    st.markdown('<h1 class="main-header">👥 HR Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Predict employee turnover using advanced machine learning algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model selection
        st.markdown("### Model Configuration")
        model_option = st.selectbox(
            "Choose Model",
            ["Random Forest", "XGBoost", "CatBoost", "AdaBoost"],
            index=0
        )
        
        # Load model button
        if st.button("🔄 Initialize Model", type="primary"):
            with st.spinner("Loading model..."):
                model, scaler, feature_names = create_model()
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names
                st.session_state.sample_data = load_sample_data()
            st.success("Model loaded successfully!")
        
        # Navigation
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Go to:",
            ["🔮 Prediction", "📈 Analytics", "📋 Dataset Overview"]
        )
    
    # Main content based on navigation
    if page == "🔮 Prediction":
        prediction_page()
    elif page == "📈 Analytics":
        analytics_page()
    else:
        dataset_overview_page()

def prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Employee Attrition Prediction</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please initialize the model from the sidebar first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Employee Information")
        
        # Create input form
        with st.form("prediction_form"):
            # Personal Information
            st.markdown("#### 👤 Personal Details")
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                age = st.slider("Age", 18, 65, 30)
                distance = st.slider("Distance from Home (km)", 1, 30, 10)
                education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)
            
            with col1_2:
                environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
                job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
                job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
            
            with col1_3:
                job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=3)
                overtime = st.selectbox("Overtime", ["No", "Yes"])
                work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
            
            # Financial Information
            st.markdown("#### 💰 Financial Details")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000)
                daily_rate = st.number_input("Daily Rate ($)", 100, 2000, 800)
                hourly_rate = st.number_input("Hourly Rate ($)", 30, 150, 65)
            
            with col2_2:
                monthly_rate = st.number_input("Monthly Rate ($)", 2000, 30000, 15000)
                percent_salary_hike = st.slider("Salary Hike Percentage", 10, 30, 15)
                stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            
            # Work Experience
            st.markdown("#### 💼 Work Experience")
            col3_1, col3_2 = st.columns(2)
            
            with col3_1:
                total_working_years = st.slider("Total Working Years", 0, 40, 10)
                years_at_company = st.slider("Years at Company", 0, 40, 5)
                years_current_role = st.slider("Years in Current Role", 0, 18, 3)
            
            with col3_2:
                years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
                years_with_manager = st.slider("Years with Current Manager", 0, 17, 3)
                num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
            
            # Additional Information
            performance_rating = st.selectbox("Performance Rating", [3, 4], index=0)
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
            training_times = st.slider("Training Times Last Year", 0, 6, 3)
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Attrition", type="primary")
            
            if submitted:
                # Prepare input data
                input_data = np.array([[
                    age, daily_rate, distance, education, environment_satisfaction,
                    hourly_rate, job_involvement, job_level, job_satisfaction,
                    monthly_income, monthly_rate, num_companies_worked,
                    1 if overtime == "Yes" else 0, percent_salary_hike,
                    performance_rating, relationship_satisfaction, stock_option_level,
                    total_working_years, training_times, work_life_balance,
                    years_at_company, years_current_role, years_since_promotion,
                    years_with_manager
                ]])
                
                # Scale input data
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                prediction = st.session_state.model.predict(input_scaled)[0]
                probability = st.session_state.model.predict_proba(input_scaled)[0]
                
                # Store results in session state for the results column
                st.session_state.prediction = prediction
                st.session_state.probability = probability
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            probability = st.session_state.probability
            
            # Simple Yes/No Result
            if prediction == 1:
                result_text = "YES - Employee will LEAVE"
                result_class = "risk-prediction"
                result_icon = "❌"
            else:
                result_text = "NO - Employee will STAY"
                result_class = "safe-prediction"
                result_icon = "✅"
            
            st.markdown(f"""
            <div class="prediction-box {result_class}">
                <h3>{result_icon} {result_text}</h3>
                <p style="font-size: 1.2rem; margin: 1rem 0;">Will this employee leave the company?</p>
                <p style="font-size: 3rem; font-weight: bold; margin: 0.5rem 0;">{"YES" if prediction == 1 else "NO"}</p>
                <p style="font-size: 1rem;">Prediction Confidence: {max(probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display detailed prediction
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box risk-prediction" style="margin-top: 1rem;">
                    <h3>⚠️ High Risk Details</h3>
                    <p style="font-size: 1.2rem;">This employee is likely to leave</p>
                    <p style="font-size: 2rem; font-weight: bold;">{probability[1]:.1%}</p>
                    <p>Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box safe-prediction" style="margin-top: 1rem;">
                    <h3>✅ Low Risk Details</h3>
                    <p style="font-size: 1.2rem;">This employee is likely to stay</p>
                    <p style="font-size: 2rem; font-weight: bold;">{probability[0]:.1%}</p>
                    <p>Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Stay', 'Will Leave'],
                'Probability': [probability[0], probability[1]]
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability', 
                        color='Probability',
                        color_continuous_scale='RdYlGn_r',
                        title="Attrition Probability")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            if prediction == 1:
                st.markdown("""
                <div class="info-box">
                    <strong>Action Items:</strong>
                    <ul>
                        <li>Schedule a retention meeting</li>
                        <li>Review compensation package</li>
                        <li>Assess work-life balance</li>
                        <li>Provide career development opportunities</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>Maintenance Actions:</strong>
                    <ul>
                        <li>Continue regular check-ins</li>
                        <li>Maintain current satisfaction levels</li>
                        <li>Consider for leadership roles</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def analytics_page():
    st.markdown('<h2 class="sub-header">📈 HR Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.sample_data is None:
        st.warning("⚠️ Please initialize the model from the sidebar first!")
        return
    
    df = st.session_state.sample_data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        attrition_rate = (df['Attrition'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Attrition Rate</h3>
            <h2>{attrition_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_age = df['Age'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>👥 Average Age</h3>
            <h2>{avg_age:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_income = df['MonthlyIncome'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Avg Income</h3>
            <h2>${avg_income:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_years = df['YearsAtCompany'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>⏰ Avg Tenure</h3>
            <h2>{avg_years:.1f} years</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by attrition
        fig1 = px.histogram(df, x='Age', color='Attrition', 
                           title='Age Distribution by Attrition',
                           labels={'Attrition': 'Will Leave'})
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Income vs Attrition
        fig2 = px.box(df, x='Attrition', y='MonthlyIncome', 
                      title='Monthly Income by Attrition',
                      labels={'Attrition': 'Will Leave'})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Overtime impact
        overtime_df = df.groupby(['OverTime', 'Attrition']).size().unstack().fillna(0)
        fig3 = px.bar(overtime_df, title='Overtime vs Attrition')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Job satisfaction correlation
        corr_data = df[['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'Attrition']].corr()
        fig4 = px.imshow(corr_data, title='Satisfaction Metrics Correlation')
        st.plotly_chart(fig4, use_container_width=True)

def dataset_overview_page():
    st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    if st.session_state.sample_data is None:
        st.warning("⚠️ Please initialize the model from the sidebar first!")
        return
    
    df = st.session_state.sample_data
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Statistics")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Features:** {len(df.columns) - 1}")
        st.write(f"**Attrition Cases:** {df['Attrition'].sum()}")
        st.write(f"**Retention Cases:** {len(df) - df['Attrition'].sum()}")
    
    with col2:
        st.markdown("### 🎯 Target Distribution")
        attrition_counts = df['Attrition'].value_counts()
        fig = px.pie(values=attrition_counts.values, 
                     names=['Will Stay', 'Will Leave'],
                     title='Attrition Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature information
    st.markdown("### 🔍 Feature Information")
    
    feature_info = {
        'Age': 'Employee age in years',
        'DailyRate': 'Daily rate of pay',
        'DistanceFromHome': 'Distance from home to workplace',
        'Education': 'Education level (1-5)',
        'EnvironmentSatisfaction': 'Environment satisfaction (1-4)',
        'JobSatisfaction': 'Job satisfaction level (1-4)',
        'MonthlyIncome': 'Monthly income in dollars',
        'OverTime': 'Whether employee works overtime (0/1)',
        'WorkLifeBalance': 'Work-life balance rating (1-4)',
        'YearsAtCompany': 'Number of years at current company'
    }
    
    info_df = pd.DataFrame(list(feature_info.items()), columns=['Feature', 'Description'])
    st.dataframe(info_df, use_container_width=True)
    
    # Raw data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="hr_attrition_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()