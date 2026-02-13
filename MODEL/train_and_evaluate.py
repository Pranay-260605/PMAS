"""
Training and Evaluation Script for Predictive Maintenance TFT Model
Complete pipeline from data loading to prediction
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Import our TFT model
from predictive_maintenance_tft import (
    DataPreprocessor, TimeSeriesDataset, TemporalFusionTransformer,
    train_model, predict_with_uncertainty, calculate_health_status,
    calculate_next_maintenance, get_device, DEVICE
)


def load_and_prepare_data(file_path):
    """Load and split data into train/val/test sets"""
    
    print("📂 Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"✓ Loaded {len(df)} records for {df['machine_id'].nunique()} machines")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
    
    # Split data per machine to ensure all machines appear in all splits
    # This prevents unseen machine IDs in validation/test sets
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for machine_id in df['machine_id'].unique():
        machine_data = df[df['machine_id'] == machine_id].copy()
        
        # Split this machine's data: 70% train, 15% val, 15% test
        n = len(machine_data)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)
        
        train_dfs.append(machine_data.iloc[:train_idx])
        val_dfs.append(machine_data.iloc[train_idx:val_idx])
        test_dfs.append(machine_data.iloc[val_idx:])
    
    # Combine all machines
    train_df = pd.concat(train_dfs, ignore_index=True).sort_values('timestamp')
    val_df = pd.concat(val_dfs, ignore_index=True).sort_values('timestamp')
    test_df = pd.concat(test_dfs, ignore_index=True).sort_values('timestamp')
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_df)} samples ({train_df['machine_id'].nunique()} machines)")
    print(f"  Validation: {len(val_df)} samples ({val_df['machine_id'].nunique()} machines)")
    print(f"  Test: {len(test_df)} samples ({test_df['machine_id'].nunique()} machines)")
    
    return train_df, val_df, test_df


def prepare_datasets(train_df, val_df, test_df, sequence_length=24, prediction_horizon=12):
    """Preprocess data and create PyTorch datasets"""
    
    print("\n🔄 Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess training data (fit scalers)
    train_processed = preprocessor.preprocess(train_df, is_training=True)
    print(f"✓ Training data preprocessed")
    
    # Preprocess validation data (transform only)
    val_processed = preprocessor.preprocess(val_df, is_training=False)
    print(f"✓ Validation data preprocessed")
    
    # Preprocess test data (transform only)
    test_processed = preprocessor.preprocess(test_df, is_training=False)
    print(f"✓ Test data preprocessed")
    
    # Create datasets
    print(f"\n📦 Creating time series sequences (length={sequence_length}, horizon={prediction_horizon})...")
    
    train_dataset = TimeSeriesDataset(train_processed, sequence_length, prediction_horizon)
    val_dataset = TimeSeriesDataset(val_processed, sequence_length, prediction_horizon)
    test_dataset = TimeSeriesDataset(test_processed, sequence_length, prediction_horizon)
    
    print(f"✓ Training sequences: {len(train_dataset)}")
    print(f"✓ Validation sequences: {len(val_dataset)}")
    print(f"✓ Test sequences: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, preprocessor


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """Create PyTorch DataLoaders"""
    
    # MPS-optimized: use pin_memory=False for MPS, True for CUDA
    pin_memory = torch.cuda.is_available()
    
    # Determine optimal number of workers for Mac
    num_workers = 0 if DEVICE.type == 'mps' else 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader, preprocessor):
    """Evaluate model performance"""
    
    print("\n📊 Evaluating model on test set...")
    
    predictions, uncertainties, targets = predict_with_uncertainty(model, test_loader)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions.flatten() - targets.flatten()))
    rmse = np.sqrt(np.mean((predictions.flatten() - targets.flatten()) ** 2))
    mape = np.mean(np.abs((predictions.flatten() - targets.flatten()) / (targets.flatten() + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((targets.flatten() - predictions.flatten()) ** 2)
    ss_tot = np.sum((targets.flatten() - np.mean(targets.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Average Uncertainty: {np.mean(uncertainties):.2f} hours")
    print("="*60 + "\n")
    
    return predictions, uncertainties, targets


def plot_results(predictions, targets, uncertainties, save_path='prediction_results.png'):
    """Plot prediction results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Predictions vs Actual
    axes[0, 0].scatter(targets, predictions, alpha=0.5)
    axes[0, 0].plot([targets.min(), targets.max()], 
                     [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual RUL (hours)')
    axes[0, 0].set_ylabel('Predicted RUL (hours)')
    axes[0, 0].set_title('Predictions vs Actual RUL')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = predictions.flatten() - targets.flatten()
    axes[0, 1].scatter(predictions, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted RUL (hours)')
    axes[0, 1].set_ylabel('Residuals (hours)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Prediction Error (hours)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty vs Error
    abs_errors = np.abs(residuals)
    axes[1, 1].scatter(uncertainties, abs_errors, alpha=0.5)
    axes[1, 1].set_xlabel('Prediction Uncertainty (hours)')
    axes[1, 1].set_ylabel('Absolute Error (hours)')
    axes[1, 1].set_title('Uncertainty vs Absolute Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Results plot saved to: {save_path}")
    
    return fig


def generate_maintenance_report(model, test_df, preprocessor, sequence_length=24):
    """Generate maintenance recommendations for each machine"""
    
    print("\n📋 Generating Maintenance Report...")
    print("="*60)
    
    report_data = []
    
    for machine_id in test_df['machine_id'].unique():
        # Get latest data for this machine
        machine_data = test_df[test_df['machine_id'] == machine_id].sort_values('timestamp')
        
        if len(machine_data) < sequence_length:
            continue
        
        # Get last sequence
        latest_data = machine_data.iloc[-sequence_length:]
        
        # Preprocess
        processed = preprocessor.preprocess(latest_data, is_training=False)
        
        # Create dataset
        dataset = TimeSeriesDataset(processed, sequence_length=sequence_length, prediction_horizon=1)
        
        if len(dataset) == 0:
            continue
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Predict
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            sequences = batch['sequence'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            
            pred, quantiles, feature_weights = model(sequences, static)
            
            predicted_rul = pred.item()
            lower_bound = quantiles[0].item()
            upper_bound = quantiles[2].item()
        
        # Get current status
        health_status = calculate_health_status(predicted_rul)
        current_timestamp = machine_data.iloc[-1]['timestamp']
        next_maintenance = calculate_next_maintenance(current_timestamp, predicted_rul)
        
        # Get top 5 important features
        feature_importance = feature_weights[0].cpu().numpy()
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        
        report_entry = {
            'machine_id': machine_id,
            'current_timestamp': current_timestamp,
            'predicted_rul': predicted_rul,
            'rul_lower_bound': lower_bound,
            'rul_upper_bound': upper_bound,
            'health_status': health_status,
            'next_maintenance_date': next_maintenance,
            'days_until_maintenance': predicted_rul / 24,
        }
        
        report_data.append(report_entry)
        
        # Print machine report
        print(f"\n🔧 Machine: {machine_id}")
        print(f"   Status: {health_status}")
        print(f"   Predicted RUL: {predicted_rul:.1f} hours ({predicted_rul/24:.1f} days)")
        print(f"   Confidence Interval: [{lower_bound:.1f}, {upper_bound:.1f}] hours")
        print(f"   Next Maintenance: {next_maintenance.strftime('%Y-%m-%d %H:%M')}")
        
        if health_status == "Critical":
            print(f"   ⚠️  URGENT: Schedule maintenance immediately!")
        elif health_status == "Warning":
            print(f"   ⚡ ATTENTION: Schedule maintenance within {int(predicted_rul/24)} days")
        else:
            print(f"   ✓  GOOD: Machine operating normally")
    
    print("\n" + "="*60 + "\n")
    
    return pd.DataFrame(report_data)


def main():
    """Main training and evaluation pipeline"""
    
    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        print("Usage: python train_and_evaluate.py <data_file.csv>")
        print("\nFor demo purposes, using synthetic data structure...")
        data_file = None
    
    if data_file is None:
        print("\n⚠️  No data file provided. This is a template script.")
        print("To run the full pipeline:")
        print("1. Prepare your CSV file with the required columns")
        print("2. Run: python train_and_evaluate.py your_data.csv")
        return
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(data_file)
    
    # Prepare datasets
    sequence_length = 24  # 24 hours of history
    prediction_horizon = 12  # Predict 12 hours ahead
    
    train_dataset, val_dataset, test_dataset, preprocessor = prepare_datasets(
        train_df, val_df, test_df, sequence_length, prediction_horizon
    )
    
    # Create dataloaders
    batch_size = 32
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Initialize model
    print("\n🤖 Initializing Temporal Fusion Transformer...")
    input_dim = train_dataset.sequences.shape[2]  # Number of features
    num_machines = len(train_df['machine_id'].unique())
    
    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        num_machines=num_machines
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {trainable_params:,} trainable parameters")
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=50, learning_rate=0.001
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_tft_model.pth'))
    print("✓ Best model loaded")
    
    # Evaluate on test set
    predictions, uncertainties, targets = evaluate_model(model, test_loader, preprocessor)
    
    # Plot results
    plot_results(predictions, targets, uncertainties)
    
    # Generate maintenance report
    maintenance_report = generate_maintenance_report(
        model, test_df, preprocessor, sequence_length
    )
    
    # Save report
    report_path = 'maintenance_report.csv'
    maintenance_report.to_csv(report_path, index=False)
    print(f"✓ Maintenance report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated files in current directory:")
    print("  • best_tft_model.pth - Trained model weights")
    print("  • prediction_results.png - Visualization of predictions")
    print("  • maintenance_report.csv - Maintenance recommendations")
    

if __name__ == "__main__":
    main()
