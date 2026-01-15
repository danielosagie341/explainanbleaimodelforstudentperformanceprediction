import pandas as pd
import numpy as np
import joblib

# Load the model
try:
    model = joblib.load('random_forest_model1.pkl')
    print("✅ Model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {model.get_params()}")
    
    # Test with zeros
    feature_names = [
        'Attendance (%)', 'Midterm_Score', 'Final_Score', 
        'Assignments_Avg', 'Quizzes_Avg', 
        'Participation_Score', 'Projects_Score'
    ]
    
    # Test case 1: All zeros
    zeros_df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=feature_names)
    zeros_prediction = model.predict(zeros_df)[0]
    print(f"\n🧪 All zeros prediction: {zeros_prediction:.1f}")
    
    # Test case 2: All 100s
    hundreds_df = pd.DataFrame([[100, 100, 100, 100, 100, 100, 100]], columns=feature_names)
    hundreds_prediction = model.predict(hundreds_df)[0]
    print(f"🧪 All 100s prediction: {hundreds_prediction:.1f}")
    
    # Test case 3: Realistic values
    realistic_df = pd.DataFrame([[85, 80, 85, 80, 75, 85, 80]], columns=feature_names)
    realistic_prediction = model.predict(realistic_df)[0]
    print(f"🧪 Realistic values prediction: {realistic_prediction:.1f}")
    
    # Check feature importances
    if hasattr(model, 'feature_importances_'):
        print(f"\n📊 Feature Importances:")
        for i, feature in enumerate(feature_names):
            print(f"  {feature}: {model.feature_importances_[i]:.3f}")
    
    # Check if model has any base prediction
    print(f"\n🔍 Model info:")
    print(f"  Number of estimators: {model.n_estimators}")
    print(f"  Random state: {model.random_state}")
    
except Exception as e:
    print(f"❌ Error loading or testing model: {e}")