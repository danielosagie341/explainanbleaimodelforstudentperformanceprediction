from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import werkzeug
import sys 

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model1.pkl")

features = [
    "Attendance (%)", "Midterm_Score", "Final_Score", 
    "Assignments_Avg", "Quizzes_Avg", 
    "Participation_Score", "Projects_Score"
]

feature_info = {
    "Attendance (%)": {"min": 0, "max": 100, "optional": False, "description": "Percentage of classes attended"},
    "Midterm_Score": {"min": 0, "max": 100, "optional": False, "description": "Score out of 100"},
    "Final_Score": {"min": 0, "max": 100, "optional": False, "description": "Score out of 100"},
    "Assignments_Avg": {"min": 0, "max": 100, "optional": False, "description": "Average assignment score"},
    "Quizzes_Avg": {"min": 0, "max": 100, "optional": False, "description": "Average quiz score"},
    "Participation_Score": {"min": 0, "max": 100, "optional": False, "description": "Class participation score"},
    "Projects_Score": {"min": 0, "max": 100, "optional": False, "description": "Project work score"}
}

# STARTUP: Initialize SHAP
explainer = None
try:
    print("Attempting to initialize SHAP...", file=sys.stdout)
    explainer = shap.TreeExplainer(model)
    sample_data = np.array([[85, 75, 80, 78, 82, 88, 85]])
    _ = explainer.shap_values(sample_data) 
    print("SHAP initialized successfully.", file=sys.stdout)
except Exception as e:
    print(f"⚠️ SHAP initialization failed: {e}", file=sys.stderr)
    explainer = None

def create_feature_importance_plot():
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title("Global Feature Importance", fontsize=16, fontweight='bold')
            plt.bar(range(len(features)), importances[indices], 
                    color=['#4f8cff', '#38b6ff', '#2196F3', '#1976D2', '#0D47A1', '#1565C0', '#1E88E5'])
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Importance", fontsize=12)
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            return base64.b64encode(plot_data).decode()
    except Exception as e:
        print(f"Feature importance error: {e}", file=sys.stderr)
    return "" 

def create_shap_explanation(input_values):
    try:
        if explainer is None:
            return "", [] 
            
        shap_values = explainer.shap_values(np.array([input_values]))
        
        plt.figure(figsize=(10, 6))
        
        base_val = explainer.expected_value
        if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
            if len(base_val) > 1:
                base_val = base_val[1]
                vals = shap_values[1][0] 
            else:
                base_val = base_val[0]
                vals = shap_values[0]
        else:
            vals = shap_values[0]

        shap.waterfall_plot(shap.Explanation(
            values=vals, 
            base_values=base_val, 
            data=np.array(input_values),
            feature_names=features
        ), show=False)
        
        plt.title("Why this prediction? (Waterfall Plot)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        waterfall_plot = base64.b64encode(plot_data).decode()
        
        shap_summary = []
        for i, (feature, value, shap_val) in enumerate(zip(features, input_values, vals)):
            impact = "Positive" if shap_val > 0 else "Negative" if shap_val < 0 else "Neutral"
            shap_summary.append({
                'feature': feature,
                'value': value,
                'shap_value': round(shap_val, 3),
                'impact': impact,
                'abs_impact': abs(shap_val)
            })
        
        shap_summary.sort(key=lambda x: x['abs_impact'], reverse=True)
        return waterfall_plot, shap_summary
    except Exception as e:
        print(f"SHAP chart error: {e}", file=sys.stderr)
        return "", [] 

def get_prediction_insights(prediction, input_values, shap_summary=None):
    insights = []
    
    # 1. Performance Category Summary (Guarantees at least 1 insight)
    if prediction >= 90: 
        category = "Excellent"
        insights.append("🌟 **Summary:** Outstanding performance. You are maximizing your potential across most metrics.")
    elif prediction >= 80: 
        category = "Good"
        insights.append("👍 **Summary:** Strong performance. Minor improvements could push this to an 'Excellent' rating.")
    elif prediction >= 70: 
        category = "Average"
        insights.append("📊 **Summary:** Consistent average performance. You are passing, but there is clear room for growth.")
    elif prediction >= 60: 
        category = "Below Average"
        insights.append("⚠️ **Summary:** You are at risk of falling behind. Immediate attention to key areas is required.")
    else: 
        category = "Poor"
        insights.append("🚨 **Summary:** Critical intervention needed. Current trajectory suggests a high risk of failure.")
    
    # 2. Heuristic Analysis (Fallback if SHAP fails, guarantees explainability logic)
    attendance, midterm, final, assignments, quizzes, participation, projects = input_values
    
    # Attendance Check
    if attendance < 75:
        insights.append("🗓️ **Attendance:** Your attendance is below 75%. Attending more classes is the easiest way to boost your grade.")
    elif attendance > 90:
        insights.append("✅ **Attendance:** Your high attendance is a strong stabilizing factor for your grade.")
        
    # Exam Trend
    if final > midterm + 5:
        insights.append("📈 **Trend:** Great job improving from Midterm to Finals. You learned the material well.")
    elif midterm > final + 5:
        insights.append("📉 **Trend:** Your performance dropped on the Final exam compared to the Midterm.")

    # Effort Check (Assignments & Participation)
    if assignments < 70 or participation < 70:
        insights.append("📝 **Engagement:** Low scores in assignments or participation suggest you might be missing easy points.")

    # 3. deeply integrated SHAP Analysis (If available)
    if shap_summary and len(shap_summary) > 0:
        top_driver = shap_summary[0]
        if top_driver['shap_value'] > 0:
            insights.append(f"🏆 **Top Strength:** According to the AI, **{top_driver['feature']}** is the strongest factor lifting your score.")
        else:
            insights.append(f"🛑 **Primary Drag:** The AI identified **{top_driver['feature']}** as the main factor pulling your score down.")

    return category, insights

@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["POST"])
def index():
    if request.method == "POST":
        print(f"Processing request on: {request.path}", file=sys.stdout)
        try:
            # 1. Parse Input
            if request.is_json:
                data = request.get_json()
                inputs = [float(data.get(feature, 0)) for feature in features]
            else:
                inputs = [float(request.form[feature]) for feature in features]

            # 2. Run Prediction
            input_df = pd.DataFrame([inputs], columns=features)
            prediction = model.predict(input_df)[0]
            prediction = round(prediction, 2)
            
            # 3. Generate Visuals
            importance_plot = create_feature_importance_plot()
            waterfall_plot, shap_summary = create_shap_explanation(inputs)
            
            # 4. Generate Insights
            category, insights = get_prediction_insights(prediction, inputs, shap_summary)
            
            response_data = {
                "prediction": prediction,
                "category": category,
                "insights": insights,
                "importance_plot": importance_plot or "",
                "waterfall_plot": waterfall_plot or "",
                "shap_summary": shap_summary or [],
                "success": True
            }

            # 5. Return JSON if API
            if request.path == '/predict' or request.is_json:
                return jsonify(response_data)
            
            # 6. Return HTML Form (Fallback)
            return render_template("index.html", 
                                 features=features, feature_info=feature_info,
                                 prediction=prediction, category=category, insights=insights,
                                 importance_plot=importance_plot, waterfall_plot=waterfall_plot,
                                 shap_summary=shap_summary, input_values=dict(zip(features, inputs)))

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message, file=sys.stderr)
            if request.path == '/predict' or request.is_json:
                return jsonify({"error": error_message, "success": False}), 400
            return render_template("index.html", features=features, feature_info=feature_info, 
                                 prediction=error_message, category="Error", insights=[], 
                                 importance_plot="", waterfall_plot="", shap_summary=[], input_values={})
    
    # GET request
    importance_plot = create_feature_importance_plot()
    return render_template("index.html", features=features, feature_info=feature_info, importance_plot=importance_plot or "")

if __name__ == "__main__":
    app.run(debug=True)