from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_model1.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature names for the model (must match training data)
FEATURE_NAMES = [
    'Attendance (%)', 'Midterm_Score', 'Final_Score', 
    'Assignments_Avg', 'Quizzes_Avg', 
    'Participation_Score', 'Projects_Score'
]

# Feature descriptions and units
FEATURE_INFO = {
    'Attendance (%)': {
        'description': 'Student attendance percentage',
        'unit': 'percentage (0-100)',
        'example': '85',
        'optional': False
    },
    'Midterm_Score': {
        'description': 'Midterm examination score',
        'unit': 'score out of 100',
        'example': '78',
        'optional': False
    },
    'Final_Score': {
        'description': 'Final examination score',
        'unit': 'score out of 100',
        'example': '82',
        'optional': False
    },
    'Assignments_Avg': {
        'description': 'Average assignment scores',
        'unit': 'score out of 100',
        'example': '75',
        'optional': False
    },
    'Quizzes_Avg': {
        'description': 'Average quiz scores',
        'unit': 'score out of 100',
        'example': '80',
        'optional': True
    },
    'Participation_Score': {
        'description': 'Class participation score',
        'unit': 'score out of 100',
        'example': '90',
        'optional': True
    },
    'Projects_Score': {
        'description': 'Project/coursework average score',
        'unit': 'score out of 100',
        'example': '85',
        'optional': True
    }
}

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES, feature_info=FEATURE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        print(f"Received data: {data}")  # Debug logging
        
        # Extract features in the correct order
        features = []
        feature_values = {}
        
        # Set default values for optional fields if not provided
        defaults = {
            'Quizzes_Avg': 75,  # Average quiz score
            'Participation_Score': 80,  # Good participation
            'Projects_Score': 75  # Average project score
        }
        
        for feature in FEATURE_NAMES:
            # Use provided value or default for optional fields
            if feature in data and data[feature] is not None and data[feature] != '':
                value = float(data[feature])
            elif feature in defaults:
                value = defaults[feature]
                print(f"Using default value for {feature}: {value}")
            else:
                value = 0.0
                
            features.append(value)
            feature_values[feature] = value
        
        # Convert to pandas DataFrame with proper feature names (this is what the model expects)
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        print(f"Features DataFrame: {features_df}")  # Debug logging
        
        # Make prediction using DataFrame with feature names
        if model is None:
            raise Exception("Model not loaded")
        
        prediction = model.predict(features_df)[0]
        
        # Add logic for extreme cases (more realistic)
        # If all major scores are very low, adjust prediction downward
        major_scores = [
            feature_values.get('Attendance (%)', 0),
            feature_values.get('Midterm_Score', 0),
            feature_values.get('Final_Score', 0),
            feature_values.get('Assignments_Avg', 0)
        ]
        
        # If all major scores are extremely low (< 20), cap the prediction
        if all(score < 20 for score in major_scores):
            prediction = min(prediction, 25)  # Very poor performance
            print(f"Applied extreme low performance cap: {prediction}")
        elif all(score < 40 for score in major_scores):
            prediction = min(prediction, 45)  # Poor performance
            print(f"Applied low performance cap: {prediction}")
        
        print(f"Final prediction: {prediction}")  # Debug logging
        
        # Get feature importances from the model
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feature in enumerate(FEATURE_NAMES):
                feature_importance[feature] = float(importances[i])
        
        # Create simple explanations based on feature values and importance
        explanations = generate_simple_explanations(feature_values, feature_importance, prediction)
        
        # Determine performance category
        if prediction >= 85:
            category = "Excellent"
            category_color = "#10b981"
        elif prediction >= 75:
            category = "Good"
            category_color = "#06b6d4"
        elif prediction >= 65:
            category = "Average"
            category_color = "#f59e0b"
        else:
            category = "Below Average"
            category_color = "#ef4444"
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'category': category,
            'category_color': category_color,
            'feature_importance': feature_importance,
            'explanations': explanations
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")  # Debug logging
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def generate_simple_explanations(features, importance, prediction):
    """Generate simple explanations without SHAP"""
    explanations = []
    
    # Sort features by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 5 most important features
    top_features = sorted_features[:5]
    
    for feature, imp in top_features:
        value = features[feature]
        
        if feature == 'Attendance (%)':
            if value >= 90:
                explanations.append(f"Excellent attendance ({value:.1f}%) strongly supports high performance")
            elif value >= 80:
                explanations.append(f"Good attendance ({value:.1f}%) positively impacts performance")
            elif value >= 70:
                explanations.append(f"Average attendance ({value:.1f}%) may limit performance potential")
            else:
                explanations.append(f"Poor attendance ({value:.1f}%) significantly hurts academic performance")
        
        elif feature == 'Final_Score':
            if value >= 85:
                explanations.append(f"Strong final exam score ({value:.1f}) indicates solid understanding")
            elif value >= 75:
                explanations.append(f"Good final exam score ({value:.1f}) shows competent grasp of material")
            elif value >= 65:
                explanations.append(f"Average final exam score ({value:.1f}) suggests room for improvement")
            else:
                explanations.append(f"Low final exam score ({value:.1f}) indicates need for additional support")
        
        elif feature == 'Midterm_Score':
            if value >= 85:
                explanations.append(f"Excellent midterm performance ({value:.1f}) shows strong foundation")
            elif value >= 75:
                explanations.append(f"Good midterm score ({value:.1f}) indicates solid preparation")
            else:
                explanations.append(f"Midterm score ({value:.1f}) suggests need for improved study strategies")
        
        elif feature == 'Assignments_Avg':
            if value >= 85:
                explanations.append(f"High assignment average ({value:.1f}) shows consistent effort and understanding")
            elif value >= 75:
                explanations.append(f"Good assignment performance ({value:.1f}) indicates regular engagement")
            else:
                explanations.append(f"Assignment average ({value:.1f}) suggests need for more consistent work")
        
        elif feature == 'Participation_Score':
            if value >= 85:
                explanations.append(f"High participation score ({value:.1f}) shows active classroom engagement")
            elif value >= 75:
                explanations.append(f"Good participation ({value:.1f}) indicates positive classroom involvement")
            else:
                explanations.append(f"Participation score ({value:.1f}) suggests need for more active engagement")
    
    return explanations

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print(" Starting Explainable AI Student Performance Predictor...")
    print(f" Model Status: {' Loaded' if model else ' Not Loaded'}")
    print(" Access the application at: http://localhost:5000")
    print(" The app will open automatically in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)