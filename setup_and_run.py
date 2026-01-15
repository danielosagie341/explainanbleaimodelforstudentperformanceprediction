#!/usr/bin/env python3
"""
Setup and Run Script for Explainable AI Student Performance Predictor
This script handles the complete setup and running of the application.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages!")
        return False

def check_model_exists():
    """Check if the trained model exists"""
    model_path = "random_forest_model1.pkl"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        return False

def train_model():
    """Train the machine learning model"""
    print("🤖 Training machine learning model...")
    try:
        import train_model
        print("✅ Model training completed!")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_flask_app():
    """Run the Flask application"""
    print("🌐 Starting Flask application...")
    print("📍 Application will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\\n👋 Flask application stopped.")

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🌟 Starting Streamlit application...")
    print("📍 Application will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the server")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\\n👋 Streamlit application stopped.")

def show_menu():
    """Display the main menu"""
    print("\\n" + "="*60)
    print("🧠 EXPLAINABLE AI STUDENT PERFORMANCE PREDICTOR")
    print("="*60)
    print("1. 🔧 Setup (Install dependencies)")
    print("2. 🤖 Train Model")
    print("3. 🌐 Run Flask Web App")
    print("4. 🌟 Run Streamlit Dashboard")
    print("5. 📊 Check System Status")
    print("6. ❓ Help")
    print("7. 🚪 Exit")
    print("="*60)

def check_system_status():
    """Check the status of all system components"""
    print("\\n🔍 SYSTEM STATUS CHECK")
    print("-" * 40)
    
    # Python version
    python_ok = check_python_version()
    
    # Required packages
    required_packages = [
        'flask', 'numpy', 'pandas', 'scikit-learn', 
        'matplotlib', 'seaborn', 'shap', 'streamlit', 'plotly'
    ]
    
    packages_ok = True
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"✅ {package}")
        else:
            print(f"❌ {package} (missing)")
            packages_ok = False
    
    # Model file
    model_ok = check_model_exists()
    
    # Overall status
    print("\\n📋 OVERALL STATUS:")
    if python_ok and packages_ok and model_ok:
        print("✅ System is ready to run!")
    else:
        print("⚠️  System needs setup. Please run setup option first.")
    
    return python_ok and packages_ok and model_ok

def show_help():
    """Show help information"""
    print("\\n" + "="*60)
    print("📚 HELP & INFORMATION")
    print("="*60)
    print("""
🎯 PROJECT OVERVIEW:
This is an Explainable AI system for predicting student performance.
It uses machine learning to make predictions and SHAP to explain 
why those predictions were made.

🚀 QUICK START:
1. Run option 1 (Setup) to install dependencies
2. Run option 2 (Train Model) to create the ML model
3. Run option 3 (Flask App) or 4 (Streamlit) to start the application

📱 APPLICATIONS:
- Flask App: Traditional web interface with forms
- Streamlit: Interactive dashboard with real-time visualizations

🔧 FEATURES:
✅ Machine Learning Predictions (Random Forest)
✅ SHAP Explanations (Why predictions are made)
✅ Feature Importance Analysis
✅ Interactive Visualizations
✅ Performance Insights and Recommendations

📊 INPUT FEATURES:
- Attendance (%)
- Midterm Score
- Final Score
- Assignments Average
- Quizzes Average  
- Participation Score
- Projects Score

🎓 USE CASES:
- Early identification of at-risk students
- Understanding factors affecting performance
- Data-driven educational interventions
- Transparent AI decision making

📞 SUPPORT:
- Check README.md for detailed documentation
- Ensure all requirements are installed
- Model training may take a few minutes

🌐 URLs:
- Flask App: http://localhost:5000
- Streamlit App: http://localhost:8501
""")
    print("="*60)

def main():
    """Main application loop"""
    print("🎓 Welcome to the Explainable AI Student Performance Predictor!")
    
    while True:
        show_menu()
        choice = input("\\n🔤 Select an option (1-7): ").strip()
        
        if choice == '1':
            print("\\n🔧 SETUP")
            if check_python_version():
                install_requirements()
            input("\\nPress Enter to continue...")
            
        elif choice == '2':
            print("\\n🤖 TRAIN MODEL")
            if not check_model_exists():
                train_model()
            else:
                retrain = input("Model already exists. Retrain? (y/N): ").lower()
                if retrain == 'y':
                    train_model()
            input("\\nPress Enter to continue...")
            
        elif choice == '3':
            print("\\n🌐 FLASK WEB APP")
            if check_model_exists():
                run_flask_app()
            else:
                print("❌ Model not found! Please train the model first (option 2).")
            input("\\nPress Enter to continue...")
            
        elif choice == '4':
            print("\\n🌟 STREAMLIT DASHBOARD")
            if check_model_exists():
                run_streamlit_app()
            else:
                print("❌ Model not found! Please train the model first (option 2).")
            input("\\nPress Enter to continue...")
            
        elif choice == '5':
            check_system_status()
            input("\\nPress Enter to continue...")
            
        elif choice == '6':
            show_help()
            input("\\nPress Enter to continue...")
            
        elif choice == '7':
            print("\\n👋 Thank you for using the Explainable AI Student Performance Predictor!")
            print("🎓 Good luck with your project!")
            break
            
        else:
            print("\\n❌ Invalid option! Please select 1-7.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n👋 Goodbye!")
    except Exception as e:
        print(f"\\n❌ An error occurred: {e}")
        print("Please check the documentation or contact support.")