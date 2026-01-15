from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model1.pkl")

# List of input features expected by the model
features = [
    "Attendance (%)", "Midterm_Score", "Final_Score", 
    "Assignments_Avg", "Quizzes_Avg", 
    "Participation_Score", "Projects_Score"
]

# Initialize SHAP explainer (we'll do this once to avoid recomputation)
# Create sample data for SHAP explainer initialization
sample_data = np.array([
    [85, 75, 80, 78, 82, 88, 85],  # Good student
    [60, 65, 70, 68, 72, 75, 70],  # Average student  
    [45, 50, 55, 52, 58, 60, 55],  # Struggling student
    [95, 90, 95, 92, 94, 96, 90],  # Excellent student
    [70, 68, 75, 72, 76, 80, 75]   # Above average student
])

try:
    explainer = shap.TreeExplainer(model)
    shap_values_sample = explainer.shap_values(sample_data)
except Exception as e:
    print(f"SHAP initialization error: {e}")
    explainer = None

def create_feature_importance_plot():
    """Create feature importance plot"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance", fontsize=16, fontweight='bold')
            bars = plt.bar(range(len(features)), importances[indices], 
                          color=['#4f8cff', '#38b6ff', '#2196F3', '#1976D2', 
                                '#0D47A1', '#1565C0', '#1E88E5'])
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Importance", fontsize=12)
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode()
    except Exception as e:
        print(f"Feature importance plot error: {e}")
        return None

def create_shap_explanation(input_values):
    """Create SHAP explanation for a single prediction"""
    try:
        if explainer is None:
            return None, None
            
        # Get SHAP values for the input
        shap_values = explainer.shap_values(np.array([input_values]))
        
        # Create SHAP waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=np.array(input_values),
            feature_names=features
        ), show=False)
        plt.title("SHAP Explanation - How each feature affects the prediction", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        waterfall_plot = base64.b64encode(plot_data).decode()
        
        # Create SHAP values summary
        shap_summary = []
        for i, (feature, value, shap_val) in enumerate(zip(features, input_values, shap_values[0])):
            impact = "Positive" if shap_val > 0 else "Negative" if shap_val < 0 else "Neutral"
            shap_summary.append({
                'feature': feature,
                'value': value,
                'shap_value': round(shap_val, 3),
                'impact': impact,
                'abs_impact': abs(shap_val)
            })
        
        # Sort by absolute impact
        shap_summary.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        return waterfall_plot, shap_summary
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None, None

def get_prediction_insights(prediction, input_values):
    """Provide insights about the prediction"""
    insights = []
    
    # Performance categories
    if prediction >= 90:
        category = "Excellent"
        insights.append("This student shows excellent performance across all metrics!")
    elif prediction >= 80:
        category = "Good"
        insights.append("This student demonstrates good academic performance.")
    elif prediction >= 70:
        category = "Average"
        insights.append("This student shows average performance with room for improvement.")
    elif prediction >= 60:
        category = "Below Average"
        insights.append("This student needs additional support to improve performance.")
    else:
        category = "Poor"
        insights.append("This student requires immediate intervention and support.")
    
    # Specific insights based on input values
    attendance, midterm, final, assignments, quizzes, participation, projects = input_values
    
    if attendance < 70:
        insights.append("⚠️ Low attendance is likely impacting performance significantly.")
    elif attendance > 90:
        insights.append("✅ Excellent attendance contributes positively to performance.")
    
    if abs(midterm - final) > 15:
        if final > midterm:
            insights.append("📈 Significant improvement from midterm to final exam!")
        else:
            insights.append("📉 Performance declined from midterm to final exam.")
    
    if assignments < 70:
        insights.append("📝 Assignment performance needs improvement.")
    elif assignments > 85:
        insights.append("📝 Strong assignment performance!")
    
    return category, insights

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect input values from form
            inputs = [float(request.form[feature]) for feature in features]
            
            # Make prediction
            prediction = model.predict([inputs])[0]
            prediction = round(prediction, 2)
            
            # Get prediction insights
            category, insights = get_prediction_insights(prediction, inputs)
            
            # Create feature importance plot
            importance_plot = create_feature_importance_plot()
            
            # Create SHAP explanation
            waterfall_plot, shap_summary = create_shap_explanation(inputs)
            
            return render_template("index.html", 
                                 features=features, 
                                 prediction=prediction,
                                 category=category,
                                 insights=insights,
                                 importance_plot=importance_plot,
                                 waterfall_plot=waterfall_plot,
                                 shap_summary=shap_summary,
                                 input_values=dict(zip(features, inputs)))
        except Exception as e:
            return render_template("index.html", 
                                 features=features, 
                                 prediction=f"Error: {str(e)}")
    
    # For GET requests, show feature importance
    importance_plot = create_feature_importance_plot()
    return render_template("index.html", 
                         features=features, 
                         importance_plot=importance_plot)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        inputs = [float(data[feature]) for feature in features]
        prediction = model.predict([inputs])[0]
        
        category, insights = get_prediction_insights(prediction, inputs)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'category': category,
            'insights': insights,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
