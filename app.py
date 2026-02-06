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
    # Warmup
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
            plt.title("Feature Importance", fontsize=16, fontweight='bold')
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
        
        plt.title("SHAP Explanation", fontsize=14, fontweight='bold')
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

def get_prediction_insights(prediction, input_values):
    insights = []
    if prediction >= 90: category = "Excellent"
    elif prediction >= 80: category = "Good"
    elif prediction >= 70: category = "Average"
    elif prediction >= 60: category = "Below Average"
    else: category = "Poor"
    
    if category == "Excellent": insights.append("This student shows excellent performance!")
    elif category == "Below Average" or category == "Poor": insights.append("This student requires additional support.")
    else: insights.append("This student is performing steadily.")

    attendance, midterm, final = input_values[0], input_values[1], input_values[2]
    
    if attendance < 70: insights.append("⚠️ Low attendance (-70%) is impacting performance.")
    if abs(midterm - final) > 15:
        if final > midterm: insights.append("📈 Significant improvement from midterm to final!")
        else: insights.append("📉 Performance declined from midterm to final.")
    
    if explainer is None:
        insights.append("ℹ️ AI Analysis (SHAP) disabled to save server resources.")

    return category, insights

@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["POST"])
def index():
    if request.method == "POST":
        print(f"Processing request on: {request.path}", file=sys.stdout)
        try:
            # 1. Parse Input Data (Handle both JSON and Form)
            if request.is_json:
                data = request.get_json()
                inputs = [float(data.get(feature, 0)) for feature in features]
            else:
                inputs = [float(request.form[feature]) for feature in features]

            # 2. Run Model (using DataFrame to fix warnings)
            input_df = pd.DataFrame([inputs], columns=features)
            prediction = model.predict(input_df)[0]
            prediction = round(prediction, 2)
            
            # 3. Generate Visuals
            category, insights = get_prediction_insights(prediction, inputs)
            importance_plot = create_feature_importance_plot()
            waterfall_plot, shap_summary = create_shap_explanation(inputs)
            
            response_data = {
                "prediction": prediction,
                "category": category,
                "insights": insights,
                "importance_plot": importance_plot or "",
                "waterfall_plot": waterfall_plot or "",
                "shap_summary": shap_summary or [],
                "success": True
            }

            # 4. CRITICAL FIX: Return JSON if route is /predict OR headers were JSON
            if request.path == '/predict' or request.is_json:
                return jsonify(response_data)
            
            # 5. Default Fallback: Return HTML (Only for root URL form posts)
            return render_template("index.html", 
                                 features=features, feature_info=feature_info,
                                 prediction=prediction, category=category, insights=insights,
                                 importance_plot=importance_plot, waterfall_plot=waterfall_plot,
                                 shap_summary=shap_summary, input_values=dict(zip(features, inputs)))

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message, file=sys.stderr)
            
            # Return JSON error for API calls
            if request.path == '/predict' or request.is_json:
                return jsonify({"error": error_message, "success": False}), 400

            # Return HTML error for Page calls
            return render_template("index.html", 
                                 features=features, feature_info=feature_info, 
                                 prediction=error_message, category="Error",
                                 insights=[], importance_plot="", waterfall_plot="",
                                 shap_summary=[], input_values={})
    
    # GET request
    importance_plot = create_feature_importance_plot()
    return render_template("index.html", features=features, feature_info=feature_info, importance_plot=importance_plot or "")

if __name__ == "__main__":
    app.run(debug=True)