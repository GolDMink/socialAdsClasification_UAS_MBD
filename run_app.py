#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Social Network Ads Predictor
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'Social_Network_Ads.csv',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'best_social_ads_model.pkl',
        'scaler.pkl',
        'label_encoder.pkl',
        'feature_info.pkl',
        'model_metrics.pkl'
    ]
    
    missing_models = []
    for file in model_files:
        if not os.path.exists(file):
            missing_models.append(file)
    
    if missing_models:
        print("WARNING: Missing model files:")
        for file in missing_models:
            print(f"   - {file}")
        print("\nTIP: Please run the Jupyter notebook 'social_network_ads_classification.ipynb' first to generate model files.")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("INSTALL: Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("OK: Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install requirements")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("START: Starting Streamlit application...")
        print("WEB: The app will open in your browser at: http://localhost:8501")
        print("STOP: Press Ctrl+C to stop the application")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\nBYE: Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run Streamlit: {e}")

def main():
    """Main function"""
    print("Social Network Ads Predictor")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("ERROR: Please run this script from the klasifikasiSocialAds directory")
        sys.exit(1)
    
    # Check required files
    if not check_requirements():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Ask if user wants to install requirements
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import joblib
        import sklearn
        print("OK: All required packages are installed")
    except ImportError:
        print("INSTALL: Some packages are missing")
        response = input("Do you want to install requirements? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("ERROR: Please install requirements manually: pip install -r requirements.txt")
            sys.exit(1)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main() 