"""
Quick Start Guide - Predictive Maintenance System
Run this script to see a complete demo of the system
"""

import os
import sys


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def print_step(step_num, title):
    """Print step header"""
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'─'*70}\n")


def check_dependencies():
    """Check if all dependencies are installed"""
    print_step(1, "Checking Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_mps():
    """Check MPS availability"""
    print_step(2, "Checking Hardware Acceleration")
    
    try:
        import torch
        
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) is AVAILABLE")
            print("   Your Mac's GPU will be used for acceleration!")
            return 'mps'
        elif torch.cuda.is_available():
            print("✅ CUDA GPU is AVAILABLE")
            print("   NVIDIA GPU will be used for acceleration!")
            return 'cuda'
        else:
            print("⚠️  No GPU acceleration available")
            print("   Training will use CPU (slower but will work)")
            return 'cpu'
    except Exception as e:
        print(f"❌ Error checking hardware: {e}")
        return None


def generate_sample_data():
    """Generate sample data for demo"""
    print_step(3, "Generating Sample Data")
    
    try:
        from utils import generate_sample_data
        
        print("Generating 5 machines with 1 week of hourly data...")
        df = generate_sample_data(num_machines=5, hours_per_machine=168, 
                                  output_path='demo_data.csv')
        print("\n✅ Sample data generated successfully!")
        print("   File: demo_data.csv")
        return True
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False


def show_training_example():
    """Show training example"""
    print_step(4, "Training the Model")
    
    print("To train the model on your data, run:\n")
    print("  python train_and_evaluate.py demo_data.csv\n")
    
    print("This will:")
    print("  • Split data into train/validation/test sets")
    print("  • Preprocess and engineer features")
    print("  • Train the Temporal Fusion Transformer")
    print("  • Evaluate performance on test set")
    print("  • Generate visualizations and reports")
    print("  • Save the trained model\n")
    
    print("Training typically takes:")
    print("  • Mac M1/M2/M3 (MPS): 2-5 minutes")
    print("  • CUDA GPU: 2-5 minutes")
    print("  • CPU: 10-20 minutes\n")


def show_inference_example():
    """Show inference example"""
    print_step(5, "Making Predictions")
    
    print("After training, use the model for predictions:\n")
    
    code = """
from inference import PredictiveMaintenanceInference
import pandas as pd

# Initialize inference engine
inference = PredictiveMaintenanceInference('best_tft_model.pth')

# Load your current sensor data
data = pd.read_csv('current_readings.csv')

# Predict for all machines
predictions = inference.predict_all_machines(data)

# View results for a specific machine
result = predictions['M01']
print(f"Machine: {result['machine_id']}")
print(f"RUL: {result['predicted_rul_days']:.1f} days")
print(f"Status: {result['health_status']}")
print(f"Next Maintenance: {result['next_maintenance_date']}")
"""
    
    print(code)


def show_file_structure():
    """Show project file structure"""
    print_step(6, "Project Structure")
    
    structure = """
predictive-maintenance/
│
├── predictive_maintenance_tft.py   # Core TFT model implementation
├── train_and_evaluate.py           # Training pipeline
├── inference.py                    # Real-time prediction engine
├── utils.py                        # Data utilities
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
│
├── demo_data.csv                   # Generated sample data
├── best_tft_model.pth             # Trained model (after training)
├── prediction_results.png          # Visualization (after training)
└── maintenance_report.csv          # Recommendations (after training)
"""
    
    print(structure)


def show_key_features():
    """Show key features"""
    print_step(7, "Key Features")
    
    features = """
✨ MODEL CAPABILITIES:
  • Predicts Remaining Useful Life (RUL) for industrial machines
  • Provides uncertainty bounds (10th, 50th, 90th percentiles)
  • Multi-machine learning with shared knowledge transfer
  • Feature importance analysis
  • Health status classification (Good/Warning/Critical)

🚀 TECHNICAL HIGHLIGHTS:
  • Temporal Fusion Transformer architecture
  • LSTM + Multi-head Self-Attention
  • Optimized for Mac MPS (Metal Performance Shaders)
  • Also supports CUDA GPU and CPU
  • Early stopping and learning rate scheduling
  • Quantile regression for uncertainty

📊 PREPROCESSING:
  • Automatic feature engineering
  • Rolling statistics (3h, 6h, 12h windows)
  • Temporal feature extraction
  • Standardization and encoding
  • Missing data handling
"""
    
    print(features)


def show_next_steps():
    """Show next steps"""
    print_step(8, "Next Steps")
    
    steps = """
1. PREPARE YOUR DATA:
   • Ensure your CSV has all required columns
   • Validate with: python utils.py validate your_data.csv

2. TRAIN THE MODEL:
   • Run: python train_and_evaluate.py your_data.csv
   • Wait for training to complete
   • Review metrics and visualizations

3. MAKE PREDICTIONS:
   • Load the trained model
   • Feed recent sensor data (last 24+ hours)
   • Get RUL predictions and recommendations

4. INTEGRATE INTO PRODUCTION:
   • Set up automated data collection
   • Schedule periodic predictions
   • Create alerts for critical machines
   • Track prediction accuracy over time

5. CUSTOMIZE:
   • Adjust hyperparameters in train_and_evaluate.py
   • Add domain-specific features in preprocessing
   • Modify health status thresholds
   • Implement custom RUL calculation logic
"""
    
    print(steps)


def show_troubleshooting():
    """Show common issues and solutions"""
    print_header("TROUBLESHOOTING")
    
    issues = """
COMMON ISSUES:

1. "MPS backend not available"
   → Update to Python 3.8+ and PyTorch 2.0+
   → Check: python -c "import torch; print(torch.backends.mps.is_available())"

2. "Insufficient data for prediction"
   → Need minimum 24 hours of historical data per machine
   → Check your timestamp formatting

3. "Out of memory"
   → Reduce batch_size (try 16 or 8)
   → Reduce sequence_length if possible
   → Close other applications

4. "Poor prediction accuracy"
   → Check data quality (missing values, outliers)
   → Increase training data
   → Adjust hyperparameters
   → Verify RUL calculation logic

5. "Slow training"
   → Ensure MPS/CUDA is being used (check device output)
   → Reduce model size (hidden_dim, num_layers)
   → Use smaller dataset for initial testing
"""
    
    print(issues)


def show_resources():
    """Show additional resources"""
    print_header("RESOURCES")
    
    resources = """
📚 DOCUMENTATION:
  • Full guide: README.md
  • Code documentation: See docstrings in Python files
  
🔧 UTILITIES:
  • Generate sample data: python utils.py generate 5 168 sample.csv
  • Validate data: python utils.py validate your_data.csv
  
📊 MODEL DETAILS:
  • Architecture: Temporal Fusion Transformer (TFT)
  • Input: 24-hour sequences of sensor readings
  • Output: RUL prediction with confidence intervals
  • Features: 62 engineered features per timestep
  
💡 TIPS:
  • Start with sample data to understand the workflow
  • Monitor training progress and stop if overfitting
  • Use validation metrics to tune hyperparameters
  • Keep historical predictions to track model performance
  
🤝 BEST PRACTICES:
  • Collect at least 2-3 months of historical data
  • Include data from both normal and degraded states
  • Validate predictions against actual maintenance events
  • Retrain model periodically with new data
"""
    
    print(resources)


def main():
    """Main quick start guide"""
    
    print_header("PREDICTIVE MAINTENANCE - QUICK START GUIDE")
    
    print("Welcome to the Predictive Maintenance System!")
    print("This guide will help you get started with the TFT-based RUL prediction.\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("   pip install -r requirements.txt\n")
        return
    
    # Check hardware
    device = check_mps()
    if device is None:
        return
    
    # Ask if user wants to generate sample data
    print("\n" + "─"*70)
    response = input("\nWould you like to generate sample data for testing? (y/n): ").lower()
    
    if response == 'y':
        generate_sample_data()
    else:
        print("\n⚠️  Skipping sample data generation")
        print("   Make sure you have your own data ready!")
    
    # Show training example
    show_training_example()
    
    # Show inference example
    show_inference_example()
    
    # Show project structure
    show_file_structure()
    
    # Show key features
    show_key_features()
    
    # Show next steps
    show_next_steps()
    
    # Show troubleshooting
    show_troubleshooting()
    
    # Show resources
    show_resources()
    
    # Final message
    print_header("READY TO START!")
    
    print("You're all set! Here's what to do next:\n")
    print("1. If you generated sample data, train on it:")
    print("   python train_and_evaluate.py demo_data.csv\n")
    print("2. Or prepare and train on your own data:")
    print("   python train_and_evaluate.py your_data.csv\n")
    print("3. After training, make predictions:")
    print("   See examples in inference.py\n")
    print("Good luck with your predictive maintenance system! 🚀\n")


if __name__ == "__main__":
    main()
