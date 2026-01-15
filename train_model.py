import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(n_samples=1000):
    """Generate realistic student performance data"""
    
    # Generate base features
    attendance = np.random.normal(80, 15, n_samples)
    attendance = np.clip(attendance, 0, 100)  # Clip to valid range
    
    # Midterm scores (influenced by attendance)
    midterm = np.random.normal(70, 12, n_samples) + (attendance - 80) * 0.3
    midterm = np.clip(midterm, 0, 100)
    
    # Assignment scores (slightly correlated with attendance and midterm)
    assignments = np.random.normal(75, 10, n_samples) + (attendance - 80) * 0.2 + (midterm - 70) * 0.1
    assignments = np.clip(assignments, 0, 100)
    
    # Quiz scores (similar pattern)
    quizzes = np.random.normal(78, 11, n_samples) + (assignments - 75) * 0.3 + np.random.normal(0, 5, n_samples)
    quizzes = np.clip(quizzes, 0, 100)
    
    # Participation scores (correlated with attendance)
    participation = np.random.normal(85, 8, n_samples) + (attendance - 80) * 0.4
    participation = np.clip(participation, 0, 100)
    
    # Project scores
    projects = np.random.normal(80, 12, n_samples) + (assignments - 75) * 0.2 + np.random.normal(0, 6, n_samples)
    projects = np.clip(projects, 0, 100)
    
    # Final exam scores (most important, influenced by all previous)
    final = (midterm * 0.4 + assignments * 0.2 + quizzes * 0.2 + 
             attendance * 0.15 + participation * 0.05) + np.random.normal(0, 8, n_samples)
    final = np.clip(final, 0, 100)
    
    # Calculate total performance (target variable)
    # Weighted combination of all factors
    total_performance = (
        attendance * 0.15 +      # 15% weight
        midterm * 0.20 +         # 20% weight  
        final * 0.25 +           # 25% weight
        assignments * 0.15 +     # 15% weight
        quizzes * 0.10 +         # 10% weight
        participation * 0.05 +   # 5% weight
        projects * 0.10          # 10% weight
    ) + np.random.normal(0, 3, n_samples)  # Add some noise
    
    total_performance = np.clip(total_performance, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Attendance (%)': attendance,
        'Midterm_Score': midterm,
        'Final_Score': final,
        'Assignments_Avg': assignments,
        'Quizzes_Avg': quizzes,
        'Participation_Score': participation,
        'Projects_Score': projects,
        'Total_Performance': total_performance
    })
    
    return data

def train_model():
    """Train the Random Forest model"""
    print("Generating student performance data...")
    
    # Generate data
    data = generate_student_data(n_samples=2000)
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(data.describe())
    
    # Prepare features and target
    feature_columns = [
        'Attendance (%)', 'Midterm_Score', 'Final_Score', 
        'Assignments_Avg', 'Quizzes_Avg', 
        'Participation_Score', 'Projects_Score'
    ]
    
    X = data[feature_columns]
    y = data['Total_Performance']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.6, color='blue')
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Actual Performance')
    plt.ylabel('Predicted Performance')
    plt.title(f'Training Set\\nR² = {train_r2:.3f}')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Actual Performance')
    plt.ylabel('Predicted Performance')
    plt.title(f'Test Set\\nR² = {test_r2:.3f}')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the model
    print("\\nSaving model...")
    joblib.dump(model, 'random_forest_model1.pkl')
    print("Model saved as 'random_forest_model1.pkl'")
    
    # Save sample data for testing
    sample_data = data.head(10)
    sample_data.to_csv('sample_data.csv', index=False)
    print("Sample data saved as 'sample_data.csv'")
    
    return model, X_test, y_test, data

def test_model_predictions():
    """Test the trained model with sample predictions"""
    print("\\nTesting model with sample predictions...")
    
    # Load the saved model
    model = joblib.load('random_forest_model1.pkl')
    
    # Test cases
    test_cases = [
        {
            'name': 'Excellent Student',
            'values': [95, 90, 95, 92, 94, 96, 90],
            'expected': 'High performance'
        },
        {
            'name': 'Average Student',
            'values': [75, 70, 75, 72, 76, 80, 75],
            'expected': 'Average performance'
        },
        {
            'name': 'Struggling Student',
            'values': [45, 50, 55, 52, 58, 60, 55],
            'expected': 'Low performance'
        },
        {
            'name': 'Inconsistent Student',
            'values': [60, 80, 70, 85, 65, 75, 80],
            'expected': 'Variable performance'
        }
    ]
    
    feature_names = [
        'Attendance (%)', 'Midterm_Score', 'Final_Score', 
        'Assignments_Avg', 'Quizzes_Avg', 
        'Participation_Score', 'Projects_Score'
    ]
    
    for test_case in test_cases:
        prediction = model.predict([test_case['values']])[0]
        print(f"\\n{test_case['name']}:")
        print(f"  Input: {dict(zip(feature_names, test_case['values']))}")
        print(f"  Predicted Performance: {prediction:.2f}/100")
        print(f"  Expected: {test_case['expected']}")

if __name__ == "__main__":
    print("=" * 60)
    print("EXPLAINABLE AI - STUDENT PERFORMANCE PREDICTION")
    print("Model Training and Validation Script")
    print("=" * 60)
    
    # Train the model
    model, X_test, y_test, data = train_model()
    
    # Test with sample predictions
    test_model_predictions()
    
    print("\\n" + "=" * 60)
    print("Training completed successfully!")
    print("Files created:")
    print("  - random_forest_model1.pkl (trained model)")
    print("  - sample_data.csv (sample data for testing)")
    print("  - feature_importance.png (visualization)")
    print("  - model_performance.png (validation plots)")
    print("\\nYou can now run the Flask app or Streamlit app!")
    print("=" * 60)