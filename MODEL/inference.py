"""
Real-time Inference Script for Predictive Maintenance
Use trained TFT model to predict RUL for new sensor data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import json
from datetime import datetime, timedelta

from predictive_maintenance_tft import (
    DataPreprocessor, TimeSeriesDataset, TemporalFusionTransformer,
    calculate_health_status, calculate_next_maintenance, DEVICE
)


class PredictiveMaintenanceInference:
    """Real-time inference engine for predictive maintenance"""
    
    def __init__(self, model_path, preprocessor_path=None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model weights (.pth file)
            preprocessor_path: Path to saved preprocessor (optional)
        """
        self.model = None
        self.preprocessor = None
        self.sequence_length = 24
        self.prediction_horizon = 12
        
        self.load_model(model_path, preprocessor_path)
    
    def load_model(self, model_path, preprocessor_path=None):
        """Load trained model and preprocessor"""
        
        print(f"📦 Loading model from {model_path}...")
        
        # Initialize model (you need to know the architecture)
        # For now, using default parameters
        self.model = TemporalFusionTransformer(
            input_dim=62,  # Adjust based on your actual features
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            num_machines=5
        ).to(DEVICE)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # Load preprocessor if provided
        if preprocessor_path:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("✓ Preprocessor loaded")
        else:
            self.preprocessor = DataPreprocessor()
            print("⚠ No preprocessor provided - creating new instance")
    
    def predict_single_machine(self, sensor_data_df, machine_id):
        """
        Predict RUL for a single machine
        
        Args:
            sensor_data_df: DataFrame with recent sensor readings
            machine_id: Machine identifier
            
        Returns:
            Dictionary with prediction results
        """
        
        # Filter data for specific machine
        machine_data = sensor_data_df[sensor_data_df['machine_id'] == machine_id].copy()
        
        if len(machine_data) < self.sequence_length:
            return {
                'error': f'Insufficient data. Need at least {self.sequence_length} records, got {len(machine_data)}',
                'machine_id': machine_id
            }
        
        # Get latest sequence
        machine_data = machine_data.sort_values('timestamp').iloc[-self.sequence_length:]
        
        # Preprocess
        processed = self.preprocessor.preprocess(machine_data, is_training=False)
        
        # Create dataset
        dataset = TimeSeriesDataset(
            processed, 
            sequence_length=self.sequence_length, 
            prediction_horizon=self.prediction_horizon
        )
        
        if len(dataset) == 0:
            return {
                'error': 'Could not create valid sequence from data',
                'machine_id': machine_id
            }
        
        # Get prediction
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            batch = next(iter(loader))
            sequences = batch['sequence'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            
            pred, quantiles, feature_weights = self.model(sequences, static)
            
            predicted_rul = pred.item()
            lower_bound = quantiles[0].item()  # 10th percentile
            median = quantiles[1].item()       # 50th percentile
            upper_bound = quantiles[2].item()  # 90th percentile
        
        # Calculate derived metrics
        health_status = calculate_health_status(predicted_rul)
        current_timestamp = machine_data.iloc[-1]['timestamp']
        next_maintenance = calculate_next_maintenance(current_timestamp, predicted_rul)
        
        # Calculate urgency score (0-100)
        urgency_score = max(0, min(100, 100 * (1 - predicted_rul / 168)))
        
        # Get feature importance
        feature_importance = feature_weights[0].cpu().numpy()
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        
        # Prepare result
        result = {
            'machine_id': machine_id,
            'timestamp': current_timestamp.isoformat(),
            'predicted_rul_hours': round(predicted_rul, 2),
            'predicted_rul_days': round(predicted_rul / 24, 2),
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'median': round(median, 2),
                'upper': round(upper_bound, 2)
            },
            'health_status': health_status,
            'urgency_score': round(urgency_score, 1),
            'next_maintenance_date': next_maintenance.isoformat(),
            'recommendation': self._generate_recommendation(predicted_rul, health_status),
            'current_metrics': {
                'vibration': float(machine_data.iloc[-1]['vibration']),
                'temperature': float(machine_data.iloc[-1]['process_temperature']),
                'power_consumption': float(machine_data.iloc[-1]['power_consumption']),
                'operating_hours': float(machine_data.iloc[-1]['operating_hours'])
            }
        }
        
        return result
    
    def predict_all_machines(self, sensor_data_df):
        """
        Predict RUL for all machines in the dataset
        
        Args:
            sensor_data_df: DataFrame with sensor readings for multiple machines
            
        Returns:
            Dictionary with predictions for each machine
        """
        
        results = {}
        machine_ids = sensor_data_df['machine_id'].unique()
        
        print(f"\n🔍 Analyzing {len(machine_ids)} machines...")
        
        for machine_id in machine_ids:
            try:
                result = self.predict_single_machine(sensor_data_df, machine_id)
                results[machine_id] = result
                
                if 'error' not in result:
                    status_emoji = {
                        'Good': '✓',
                        'Warning': '⚡',
                        'Critical': '⚠️'
                    }
                    emoji = status_emoji.get(result['health_status'], '?')
                    print(f"{emoji} {machine_id}: {result['health_status']} - RUL: {result['predicted_rul_days']:.1f} days")
                else:
                    print(f"❌ {machine_id}: {result['error']}")
                    
            except Exception as e:
                results[machine_id] = {
                    'error': str(e),
                    'machine_id': machine_id
                }
                print(f"❌ {machine_id}: Error - {str(e)}")
        
        return results
    
    def _generate_recommendation(self, rul, health_status):
        """Generate maintenance recommendation based on RUL and health status"""
        
        if health_status == 'Critical':
            return f"URGENT: Schedule immediate maintenance. Estimated failure in {rul:.1f} hours."
        elif health_status == 'Warning':
            days = rul / 24
            return f"Schedule maintenance within {days:.0f} days to prevent unexpected downtime."
        else:
            days = rul / 24
            return f"Machine operating normally. Next maintenance recommended in {days:.0f} days."
    
    def generate_summary_report(self, predictions):
        """Generate a summary report of all predictions"""
        
        critical_machines = []
        warning_machines = []
        good_machines = []
        
        for machine_id, pred in predictions.items():
            if 'error' in pred:
                continue
                
            if pred['health_status'] == 'Critical':
                critical_machines.append(machine_id)
            elif pred['health_status'] == 'Warning':
                warning_machines.append(machine_id)
            else:
                good_machines.append(machine_id)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_machines': len(predictions),
            'critical_count': len(critical_machines),
            'warning_count': len(warning_machines),
            'good_count': len(good_machines),
            'critical_machines': critical_machines,
            'warning_machines': warning_machines,
            'good_machines': good_machines,
            'recommendations': []
        }
        
        # Add recommendations
        if critical_machines:
            summary['recommendations'].append({
                'priority': 'URGENT',
                'action': f'Immediate maintenance required for {len(critical_machines)} machine(s)',
                'machines': critical_machines
            })
        
        if warning_machines:
            summary['recommendations'].append({
                'priority': 'HIGH',
                'action': f'Schedule maintenance soon for {len(warning_machines)} machine(s)',
                'machines': warning_machines
            })
        
        return summary
    
    def export_predictions(self, predictions, output_path):
        """Export predictions to JSON file"""
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✓ Predictions exported to {output_path}")


def demo_inference():
    """Demonstration of inference pipeline"""
    
    print("\n" + "="*60)
    print("PREDICTIVE MAINTENANCE - INFERENCE DEMO")
    print("="*60 + "\n")
    
    print("This is a demonstration of the inference pipeline.")
    print("\nTo use in production:")
    print("1. Train your model using train_and_evaluate.py")
    print("2. Load the trained model and preprocessor")
    print("3. Feed real-time sensor data")
    print("4. Get predictions and recommendations")
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("="*60)
    
    code_example = """
# Initialize inference engine
inference = PredictiveMaintenanceInference(
    model_path='best_tft_model.pth',
    preprocessor_path='preprocessor.pkl'  # Optional
)

# Load new sensor data
new_data = pd.read_csv('current_sensor_readings.csv')

# Predict for all machines
predictions = inference.predict_all_machines(new_data)

# Generate summary
summary = inference.generate_summary_report(predictions)

# Export results
inference.export_predictions(predictions, 'predictions.json')

# Or predict for a specific machine
result = inference.predict_single_machine(new_data, 'M01')
print(f"Machine M01 RUL: {result['predicted_rul_hours']} hours")
print(f"Health Status: {result['health_status']}")
print(f"Recommendation: {result['recommendation']}")
"""
    
    print(code_example)
    
    print("\n" + "="*60)
    print("PREDICTION OUTPUT FORMAT:")
    print("="*60)
    
    example_output = {
        'machine_id': 'M01',
        'timestamp': '2025-01-15T10:30:00',
        'predicted_rul_hours': 85.2,
        'predicted_rul_days': 3.55,
        'confidence_interval': {
            'lower': 72.1,
            'median': 85.2,
            'upper': 98.3
        },
        'health_status': 'Warning',
        'urgency_score': 49.3,
        'next_maintenance_date': '2025-01-18T23:42:00',
        'recommendation': 'Schedule maintenance within 4 days to prevent unexpected downtime.',
        'current_metrics': {
            'vibration': 0.385,
            'temperature': 58.9,
            'power_consumption': 2.63,
            'operating_hours': 1234
        }
    }
    
    print(json.dumps(example_output, indent=2))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_inference()
