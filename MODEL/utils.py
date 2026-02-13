"""
Utility Script for Data Validation and Sample Data Generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_data(df):
    """
    Validate that the input data has all required columns and proper format
    
    Args:
        df: Input dataframe
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    
    required_columns = [
        'timestamp', 'machine_id', 'process_temperature', 'air_temperature',
        'vibration', 'torque', 'rpm', 'current', 'operating_hours',
        'time_since_last_maintenance', 'last_maintenance_Type',
        'idle_duration', 'power_consumption'
    ]
    
    errors = []
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types and ranges
    if 'timestamp' in df.columns:
        try:
            pd.to_datetime(df['timestamp'])
        except:
            errors.append("Timestamp column cannot be converted to datetime")
    
    # Check for null values in critical columns
    critical_cols = ['timestamp', 'machine_id', 'vibration', 'torque', 'rpm']
    for col in critical_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")
    
    # Check machine_id format
    if 'machine_id' in df.columns:
        unique_machines = df['machine_id'].unique()
        print(f"Found {len(unique_machines)} unique machines: {sorted(unique_machines)}")
    
    # Check for negative values where they shouldn't exist
    positive_columns = ['vibration', 'rpm', 'operating_hours', 'power_consumption']
    for col in positive_columns:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Column '{col}' contains negative values")
    
    # Check data continuity
    if 'timestamp' in df.columns and 'machine_id' in df.columns:
        df_sorted = df.sort_values(['machine_id', 'timestamp'])
        for machine in df['machine_id'].unique():
            machine_data = df_sorted[df_sorted['machine_id'] == machine]
            if len(machine_data) < 24:
                errors.append(f"Machine {machine} has less than 24 hours of data")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        print("✅ Data validation passed!")
        print(f"  Total records: {len(df)}")
        print(f"  Machines: {df['machine_id'].nunique()}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("❌ Data validation failed!")
        for error in errors:
            print(f"  - {error}")
    
    return is_valid, errors


def generate_sample_data(num_machines=5, hours_per_machine=168, output_path='sample_data.csv'):
    """
    Generate synthetic sensor data for testing
    
    Args:
        num_machines: Number of machines to simulate
        hours_per_machine: Hours of data per machine
        output_path: Path to save the generated CSV
    """
    
    print(f"🔧 Generating sample data for {num_machines} machines...")
    
    np.random.seed(42)
    
    data = []
    start_date = datetime(2015, 1, 1)
    
    machine_ids = [f'M{str(i+1).zfill(2)}' for i in range(num_machines)]
    
    for machine_id in machine_ids:
        # Base parameters for this machine (each machine is slightly different)
        base_temp = np.random.uniform(55, 65)
        base_vibration = np.random.uniform(0.3, 0.5)
        base_rpm = np.random.uniform(1200, 1500)
        
        # Degradation rate (some machines degrade faster)
        degradation_rate = np.random.uniform(0.0001, 0.0005)
        
        for hour in range(hours_per_machine):
            timestamp = start_date + timedelta(hours=hour)
            
            # Simulate gradual degradation over time
            degradation = degradation_rate * hour
            
            # Add daily patterns (higher load during working hours)
            hour_of_day = timestamp.hour
            daily_factor = 1.0 + 0.2 * np.sin((hour_of_day - 6) * np.pi / 12)
            
            # Add random noise
            noise = np.random.normal(0, 0.05)
            
            # Process temperature increases with degradation and daily pattern
            process_temp = base_temp + degradation * 50 + daily_factor * 3 + noise * 2
            
            # Air temperature varies throughout the day
            air_temp = 15 + 10 * np.sin((hour_of_day - 6) * np.pi / 12) + noise * 2
            
            # Vibration increases with degradation
            vibration = base_vibration + degradation * 2 + noise * 0.05
            
            # RPM varies with load
            rpm = base_rpm * daily_factor + noise * 50
            
            # Torque and current correlate with RPM and load
            torque = 40 + rpm / 30 + noise * 5
            current = 5 + rpm / 200 + noise * 0.5
            
            # Operating hours
            operating_hours = hour + 1
            
            # Time since last maintenance (reset every week)
            time_since_last = hour % 168
            
            # Last maintenance type
            if time_since_last == 0 and hour > 0:
                last_maintenance = 'Preventive'
            else:
                last_maintenance = 'None'
            
            # Idle duration (random short periods)
            idle_duration = np.random.uniform(0.05, 0.15)
            
            # Power consumption
            power_consumption = rpm * current / 1000 + noise * 0.2
            
            data.append({
                'timestamp': timestamp.strftime('%m/%d/%y %H:%M'),
                'machine_id': machine_id,
                'process_temperature': process_temp,
                'air_temperature': air_temp,
                'vibration': vibration,
                'torque': torque,
                'rpm': rpm,
                'current': current,
                'operating_hours': operating_hours,
                'time_since_last_maintenance': time_since_last,
                'last_maintenance_Type': last_maintenance,
                'idle_duration': idle_duration,
                'power_consumption': power_consumption
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} records")
    print(f"  Machines: {machine_ids}")
    print(f"  Hours per machine: {hours_per_machine}")
    print(f"  Total duration: {hours_per_machine/24:.1f} days")
    print(f"  Saved to: {output_path}")
    
    # Validate the generated data
    print("\n📊 Validating generated data...")
    validate_data(df)
    
    return df


def analyze_data_statistics(df):
    """
    Print statistical summary of the data
    
    Args:
        df: Input dataframe
    """
    
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Null: {df[col].isnull().sum()}")
    
    print("\n" + "="*60)
    print("MACHINE SUMMARY")
    print("="*60)
    
    if 'machine_id' in df.columns:
        for machine_id in sorted(df['machine_id'].unique()):
            machine_data = df[df['machine_id'] == machine_id]
            print(f"\n{machine_id}:")
            print(f"  Records: {len(machine_data)}")
            print(f"  Avg Vibration: {machine_data['vibration'].mean():.3f}")
            print(f"  Avg Temperature: {machine_data['process_temperature'].mean():.2f}")
            print(f"  Avg Power: {machine_data['power_consumption'].mean():.2f}")


def check_data_quality(df):
    """
    Perform comprehensive data quality checks
    
    Args:
        df: Input dataframe
    """
    
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    total_records = len(df)
    
    # Check completeness
    print("\n1. COMPLETENESS")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_records) * 100
        if null_pct > 0:
            print(f"  ❌ {col}: {null_pct:.2f}% missing ({null_count} records)")
        else:
            print(f"  ✅ {col}: Complete")
    
    # Check for duplicates
    print("\n2. UNIQUENESS")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"  ❌ Found {dup_count} duplicate records")
    else:
        print(f"  ✅ No duplicate records")
    
    # Check for outliers (using IQR method)
    print("\n3. OUTLIERS (using IQR method)")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / total_records) * 100
        
        if outlier_pct > 5:
            print(f"  ⚠️  {col}: {outlier_pct:.2f}% outliers ({outliers} records)")
        else:
            print(f"  ✅ {col}: {outlier_pct:.2f}% outliers")
    
    # Check temporal consistency
    if 'timestamp' in df.columns and 'machine_id' in df.columns:
        print("\n4. TEMPORAL CONSISTENCY")
        df_sorted = df.sort_values(['machine_id', 'timestamp'])
        
        for machine_id in df['machine_id'].unique():
            machine_data = df_sorted[df_sorted['machine_id'] == machine_id]
            timestamps = pd.to_datetime(machine_data['timestamp'])
            
            # Check for gaps
            time_diffs = timestamps.diff().dropna()
            expected_diff = timedelta(hours=1)
            
            gaps = (time_diffs > expected_diff).sum()
            if gaps > 0:
                print(f"  ⚠️  {machine_id}: {gaps} time gaps detected")
            else:
                print(f"  ✅ {machine_id}: Continuous time series")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("DATA UTILITY TOOLS")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'generate':
            # Generate sample data
            num_machines = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            hours = int(sys.argv[3]) if len(sys.argv) > 3 else 168
            output = sys.argv[4] if len(sys.argv) > 4 else 'sample_data.csv'
            
            generate_sample_data(num_machines, hours, output)
            
        elif command == 'validate':
            # Validate existing data
            if len(sys.argv) < 3:
                print("Usage: python utils.py validate <data_file.csv>")
                sys.exit(1)
            
            data_file = sys.argv[2]
            df = pd.read_csv(data_file)
            
            print(f"\n📂 Loaded data from: {data_file}")
            validate_data(df)
            analyze_data_statistics(df)
            check_data_quality(df)
            
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  generate [num_machines] [hours] [output_file]")
            print("  validate <data_file.csv>")
    else:
        print("Usage:")
        print("  Generate sample data:")
        print("    python utils.py generate [num_machines] [hours] [output_file]")
        print("    Example: python utils.py generate 5 168 sample_data.csv")
        print("\n  Validate data:")
        print("    python utils.py validate <data_file.csv>")
        print("    Example: python utils.py validate my_data.csv")
