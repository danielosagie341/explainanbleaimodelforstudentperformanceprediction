import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Explainable AI - Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model1.pkl")
        return model
    except:
        st.error("Model file not found. Please ensure 'random_forest_model1.pkl' is in the project directory.")
        return None

# Initialize SHAP explainer
@st.cache_resource
def get_shap_explainer(model):
    if model is None:
        return None
    try:
        sample_data = np.array([
            [85, 75, 80, 78, 82, 88, 85],
            [60, 65, 70, 68, 72, 75, 70],
            [45, 50, 55, 52, 58, 60, 55],
            [95, 90, 95, 92, 94, 96, 90],
            [70, 68, 75, 72, 76, 80, 75]
        ])
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        st.error(f"Error initializing SHAP explainer: {e}")
        return None

# Feature names
features = [
    "Attendance (%)", "Midterm_Score", "Final_Score", 
    "Assignments_Avg", "Quizzes_Avg", 
    "Participation_Score", "Projects_Score"
]

def get_prediction_category(prediction):
    """Get prediction category and color"""
    if prediction >= 90:
        return "Excellent", "🟢", "#10b981"
    elif prediction >= 80:
        return "Good", "🔵", "#3b82f6"
    elif prediction >= 70:
        return "Average", "🟡", "#f59e0b"
    elif prediction >= 60:
        return "Below Average", "🟠", "#ef4444"
    else:
        return "Poor", "🔴", "#991b1b"

def generate_insights(prediction, inputs):
    """Generate intelligent insights"""
    insights = []
    attendance, midterm, final, assignments, quizzes, participation, projects = inputs
    
    # Performance insights
    category, _, _ = get_prediction_category(prediction)
    if category == "Excellent":
        insights.append("🎉 Outstanding performance! This student excels across all metrics.")
    elif category == "Good":
        insights.append("👍 Good performance with potential for further improvement.")
    elif category == "Average":
        insights.append("📊 Average performance - focused improvement in key areas can help.")
    else:
        insights.append("⚠️ Performance needs attention - consider intervention strategies.")
    
    # Specific factor insights
    if attendance < 70:
        insights.append("🚨 Critical: Low attendance is severely impacting performance.")
    elif attendance > 90:
        insights.append("✅ Excellent attendance positively contributes to success.")
    
    if abs(midterm - final) > 15:
        if final > midterm:
            insights.append("📈 Great improvement from midterm to final exam!")
        else:
            insights.append("📉 Performance declined from midterm to final - needs review.")
    
    if assignments < 70:
        insights.append("📝 Assignment performance is below expectations.")
    
    if participation < 60:
        insights.append("🗣️ Low participation may be affecting overall engagement.")
    
    return insights

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Explainable AI Student Performance Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Understanding AI decisions through transparent explanations**")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    explainer = get_shap_explainer(model)
    
    # Sidebar for inputs
    st.sidebar.header("📋 Student Information")
    st.sidebar.markdown("Enter the student's academic metrics:")
    
    # Input collection
    inputs = []
    cols_sidebar = st.sidebar.columns(1)
    
    with cols_sidebar[0]:
        attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
        midterm = st.number_input("📝 Midterm Score", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        final = st.number_input("🎯 Final Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        assignments = st.number_input("📚 Assignments Average", min_value=0.0, max_value=100.0, value=78.0, step=0.1)
        quizzes = st.number_input("❓ Quizzes Average", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        participation = st.number_input("🙋 Participation Score", min_value=0.0, max_value=100.0, value=88.0, step=0.1)
        projects = st.number_input("🔬 Projects Score", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
    
    inputs = [attendance, midterm, final, assignments, quizzes, participation, projects]
    
    # Predict button
    if st.sidebar.button("🚀 Predict Performance", type="primary"):
        
        # Make prediction
        prediction = model.predict([inputs])[0]
        category, emoji, color = get_prediction_category(prediction)
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Prediction result
            st.markdown(f"""
            <div class="metric-card">
                <h2>{emoji} {prediction:.1f}/100</h2>
                <h3>{category} Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Insights
            insights = generate_insights(prediction, inputs)
            st.markdown("### 💡 Key Insights")
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        with col2:
            # Feature importance plot
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 📊 Feature Importance Analysis")
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='viridis',
                    title="Which factors matter most for prediction?"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # SHAP Explanations
        if explainer is not None:
            st.markdown("---")
            st.markdown("## 🔍 Detailed Prediction Explanation (SHAP Analysis)")
            
            try:
                # Calculate SHAP values
                shap_values = explainer.shap_values(np.array([inputs]))
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("### 🎯 Individual Feature Contributions")
                    
                    # Create SHAP summary
                    shap_df = pd.DataFrame({
                        'Feature': features,
                        'Value': inputs,
                        'SHAP_Value': shap_values[0],
                        'Impact': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' 
                                  for x in shap_values[0]]
                    }).sort_values('SHAP_Value', key=abs, ascending=False)
                    
                    # Display SHAP values
                    for _, row in shap_df.iterrows():
                        impact_color = "#10b981" if row['SHAP_Value'] > 0 else "#ef4444"
                        st.markdown(f"""
                        **{row['Feature']}**: {row['Value']:.1f}  
                        <span style="color: {impact_color}">
                        SHAP: {row['SHAP_Value']:+.3f} ({row['Impact']})
                        </span>
                        """, unsafe_allow_html=True)
                        st.markdown("---")
                
                with col4:
                    st.markdown("### 📈 SHAP Waterfall Visualization")
                    
                    # Create waterfall chart with Plotly
                    base_value = explainer.expected_value
                    cumulative = base_value
                    
                    # Prepare data for waterfall
                    waterfall_data = []
                    colors = []
                    
                    for i, (feature, shap_val) in enumerate(zip(features, shap_values[0])):
                        waterfall_data.append({
                            'feature': feature,
                            'value': shap_val,
                            'cumulative': cumulative + shap_val,
                            'previous': cumulative
                        })
                        colors.append('#10b981' if shap_val > 0 else '#ef4444')
                        cumulative += shap_val
                    
                    # Sort by absolute impact
                    waterfall_data.sort(key=lambda x: abs(x['value']), reverse=True)
                    
                    fig = go.Figure()
                    
                    # Add bars for top 5 features
                    top_features = waterfall_data[:5]
                    for i, data in enumerate(top_features):
                        fig.add_trace(go.Bar(
                            x=[data['feature']],
                            y=[abs(data['value'])],
                            name=f"{data['value']:+.3f}",
                            marker_color='#10b981' if data['value'] > 0 else '#ef4444',
                            text=f"{data['value']:+.3f}",
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Top 5 Feature Contributions",
                        xaxis_title="Features",
                        yaxis_title="Impact on Prediction",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error generating SHAP explanations: {e}")
        
        # Input summary
        st.markdown("---")
        st.markdown("### 📋 Input Summary")
        input_df = pd.DataFrame({
            'Metric': features,
            'Value': inputs
        })
        
        col5, col6 = st.columns([1, 2])
        with col5:
            st.dataframe(input_df, hide_index=True)
        
        with col6:
            # Radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=inputs,
                theta=features,
                fill='toself',
                name='Student Performance',
                line_color='#667eea'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Performance Radar Chart",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Show feature importance when no prediction is made
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 📊 Model Feature Importance")
            st.markdown("This chart shows which factors the AI considers most important:")
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='plasma',
                title="Global Feature Importance Across All Predictions"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Information about the system
        st.markdown("### 🎯 About This System")
        st.info("""
        This **Explainable AI System** predicts student performance and provides detailed explanations:
        
        - **🤖 AI Prediction**: Uses Random Forest algorithm for accurate forecasting
        - **🔍 SHAP Explanations**: Shows exactly how each factor contributes to the prediction
        - **📊 Visual Analysis**: Interactive charts and graphs for better understanding
        - **💡 Actionable Insights**: Specific recommendations based on the analysis
        
        **Enter student information in the sidebar to get started!**
        """)

if __name__ == "__main__":
    main()