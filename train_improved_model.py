import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def generate_realistic_student_data(n_samples=2000):
    """Generate more realistic student performance data with edge cases"""
    
    # Generate different performance tiers
    n_failing = int(n_samples * 0.1)  # 10% failing students
    n_struggling = int(n_samples * 0.2)  # 20% struggling students  
    n_average = int(n_samples * 0.4)  # 40% average students
    n_good = int(n_samples * 0.2)  # 20% good students
    n_excellent = n_samples - (n_failing + n_struggling + n_average + n_good)  # 10% excellent
    
    all_data = []
    
    # 1. Failing students (very low scores)
    for _ in range(n_failing):
        attendance = np.random.uniform(0, 40)  # Very poor attendance
        midterm = np.random.uniform(0, 35) + attendance * 0.2  # Low scores
        final = np.random.uniform(0, 40) + midterm * 0.3
        assignments = np.random.uniform(0, 45) + attendance * 0.3
        quizzes = np.random.uniform(0, 40) + assignments * 0.2
        participation = np.random.uniform(0, 50) + attendance * 0.4
        projects = np.random.uniform(0, 45) + assignments * 0.2
        
        # Cap all scores at reasonable maximums for this tier
        midterm = min(midterm, 45)
        final = min(final, 50)
        assignments = min(assignments, 55)
        quizzes = min(quizzes, 50)
        participation = min(participation, 60)
        projects = min(projects, 55)
        
        # Calculate realistic total (should be very low)
        total = (attendance * 0.15 + midterm * 0.20 + final * 0.25 + 
                assignments * 0.15 + quizzes * 0.10 + participation * 0.05 + projects * 0.10)
        total = max(0, min(total, 40))  # Cap between 0-40 for failing tier
        
        all_data.append([attendance, midterm, final, assignments, quizzes, participation, projects, total])
    
    # 2. Struggling students (below average)
    for _ in range(n_struggling):
        attendance = np.random.uniform(40, 70)
        midterm = np.random.uniform(30, 60) + attendance * 0.3
        final = np.random.uniform(35, 65) + midterm * 0.2
        assignments = np.random.uniform(40, 70) + attendance * 0.2
        quizzes = np.random.uniform(35, 65) + assignments * 0.2
        participation = np.random.uniform(50, 75) + attendance * 0.3
        projects = np.random.uniform(40, 70) + assignments * 0.2
        
        # Cap scores
        midterm = min(midterm, 70)
        final = min(final, 75)
        assignments = min(assignments, 75)
        quizzes = min(quizzes, 70)
        participation = min(participation, 80)
        projects = min(projects, 75)
        
        total = (attendance * 0.15 + midterm * 0.20 + final * 0.25 + 
                assignments * 0.15 + quizzes * 0.10 + participation * 0.05 + projects * 0.10)
        total = max(35, min(total, 65))  # Cap between 35-65
        
        all_data.append([attendance, midterm, final, assignments, quizzes, participation, projects, total])
    
    # 3. Average students
    for _ in range(n_average):
        attendance = np.random.uniform(65, 85)
        midterm = np.random.uniform(55, 80) + np.random.normal(0, 5)
        final = np.random.uniform(60, 85) + np.random.normal(0, 5)
        assignments = np.random.uniform(60, 85) + np.random.normal(0, 5)
        quizzes = np.random.uniform(60, 85) + np.random.normal(0, 5)
        participation = np.random.uniform(70, 90) + np.random.normal(0, 3)
        projects = np.random.uniform(65, 85) + np.random.normal(0, 5)
        
        # Clip to valid ranges
        midterm = np.clip(midterm, 0, 100)
        final = np.clip(final, 0, 100)
        assignments = np.clip(assignments, 0, 100)
        quizzes = np.clip(quizzes, 0, 100)
        participation = np.clip(participation, 0, 100)
        projects = np.clip(projects, 0, 100)
        
        total = (attendance * 0.15 + midterm * 0.20 + final * 0.25 + 
                assignments * 0.15 + quizzes * 0.10 + participation * 0.05 + projects * 0.10)
        total = np.clip(total + np.random.normal(0, 3), 60, 80)
        
        all_data.append([attendance, midterm, final, assignments, quizzes, participation, projects, total])
    
    # 4. Good students
    for _ in range(n_good):
        attendance = np.random.uniform(80, 95)
        midterm = np.random.uniform(70, 90) + np.random.normal(0, 5)
        final = np.random.uniform(75, 95) + np.random.normal(0, 5)
        assignments = np.random.uniform(75, 95) + np.random.normal(0, 5)
        quizzes = np.random.uniform(75, 95) + np.random.normal(0, 5)
        participation = np.random.uniform(80, 95) + np.random.normal(0, 3)
        projects = np.random.uniform(75, 95) + np.random.normal(0, 5)
        
        # Clip to valid ranges
        midterm = np.clip(midterm, 0, 100)
        final = np.clip(final, 0, 100)
        assignments = np.clip(assignments, 0, 100)
        quizzes = np.clip(quizzes, 0, 100)
        participation = np.clip(participation, 0, 100)
        projects = np.clip(projects, 0, 100)
        
        total = (attendance * 0.15 + midterm * 0.20 + final * 0.25 + 
                assignments * 0.15 + quizzes * 0.10 + participation * 0.05 + projects * 0.10)
        total = np.clip(total + np.random.normal(0, 2), 75, 90)
        
        all_data.append([attendance, midterm, final, assignments, quizzes, participation, projects, total])
    
    # 5. Excellent students
    for _ in range(n_excellent):
        attendance = np.random.uniform(90, 100)
        midterm = np.random.uniform(85, 100)
        final = np.random.uniform(85, 100)
        assignments = np.random.uniform(85, 100)
        quizzes = np.random.uniform(85, 100)
        participation = np.random.uniform(90, 100)
        projects = np.random.uniform(85, 100)
        
        total = (attendance * 0.15 + midterm * 0.20 + final * 0.25 + 
                assignments * 0.15 + quizzes * 0.10 + participation * 0.05 + projects * 0.10)
        total = np.clip(total + np.random.normal(0, 1), 85, 100)
        
        all_data.append([attendance, midterm, final, assignments, quizzes, participation, projects, total])
    
    # Convert to DataFrame
    columns = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
              'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Performance']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_improved_model():
    """Train an improved Random Forest model with realistic data"""
    print("=" * 60)
    print("IMPROVED STUDENT PERFORMANCE PREDICTOR")
    print("Training with Realistic Performance Distribution")
    print("=" * 60)
    
    # Generate realistic data
    print("Generating realistic student performance data...")
    data = generate_realistic_student_data(n_samples=2000)
    
    # Display statistics
    print("\nDataset Statistics:")
    print(data.describe())
    
    # Show performance distribution
    print(f"\nPerformance Distribution:")
    print(f"Failing (0-40):     {len(data[data['Total_Performance'] <= 40]):3d} students ({len(data[data['Total_Performance'] <= 40])/len(data)*100:.1f}%)")
    print(f"Below Avg (40-65):  {len(data[(data['Total_Performance'] > 40) & (data['Total_Performance'] <= 65)]):3d} students ({len(data[(data['Total_Performance'] > 40) & (data['Total_Performance'] <= 65)])/len(data)*100:.1f}%)")
    print(f"Average (65-75):    {len(data[(data['Total_Performance'] > 65) & (data['Total_Performance'] <= 75)]):3d} students ({len(data[(data['Total_Performance'] > 65) & (data['Total_Performance'] <= 75)])/len(data)*100:.1f}%)")
    print(f"Good (75-85):       {len(data[(data['Total_Performance'] > 75) & (data['Total_Performance'] <= 85)]):3d} students ({len(data[(data['Total_Performance'] > 75) & (data['Total_Performance'] <= 85)])/len(data)*100:.1f}%)")
    print(f"Excellent (85-100): {len(data[data['Total_Performance'] > 85]):3d} students ({len(data[data['Total_Performance'] > 85])/len(data)*100:.1f}%)")
    
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
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Test extreme cases
    print(f"\n🧪 Testing Extreme Cases:")
    
    # All zeros
    zeros_test = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=feature_columns)
    zeros_pred = model.predict(zeros_test)[0]
    print(f"All zeros → {zeros_pred:.1f}")
    
    # All perfect scores
    perfect_test = pd.DataFrame([[100, 100, 100, 100, 100, 100, 100]], columns=feature_columns)
    perfect_pred = model.predict(perfect_test)[0]
    print(f"All 100s → {perfect_pred:.1f}")
    
    # Save the model
    model_filename = 'random_forest_model1.pkl'
    joblib.dump(model, model_filename)
    print(f"\n✅ Model saved as '{model_filename}'")
    
    return model

if __name__ == "__main__":
    model = train_improved_model()