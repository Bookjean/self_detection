#!/usr/bin/env python3
"""Plot compensated sensor data.

Plots raw and compensated values together for each sensor.
Compensated values are plotted on top with transparency.

Usage:
    python scripts/plot_compensated_data.py <log_file>
    python scripts/plot_compensated_data.py ~/rb10_Proximity/logs/compensated_mlp_*.txt
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def load_log_file(filepath: str):
    """Load log file and parse data.
    
    Expected format:
        # timestamp j1 j2 j3 j4 j5 j6 raw1 raw2 ... compensated1 compensated2 ...
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # Parse header to find column indices
                if line.startswith('#') and 'raw1' in line:
                    header = line[1:].strip().split()
                    continue
                continue
            
            # Parse data line
            values = line.split()
            if len(values) > 0:
                try:
                    # Convert to float
                    row = [float(v) for v in values]
                    data.append(row)
                except ValueError:
                    continue
    
    if not data:
        raise ValueError(f"No data found in {filepath}")
    
    data = np.array(data)
    
    # Try to parse header if available
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') and 'raw1' in line:
                    header = line[1:].strip().split()
                    break
    except:
        header = None
    
    return data, header


def find_column_indices(header, num_sensors=8):
    """Find column indices for raw, compensated, and mlp_out values.
    
    Returns:
        (raw_indices, comp_indices, mlp_out_indices, num_sensors)
    """
    if header is None:
        # Assume standard format: timestamp, j1-j6, raw1-rawN, compensated1-compensatedN, mlp_out1-mlp_outN
        # timestamp = 0, joints = 1-6, raw = 7-6+num_sensors, compensated = 7+num_sensors, mlp_out = 7+num_sensors*2
        raw_start = 7  # After timestamp (1) + joints (6)
        comp_start = raw_start + num_sensors
        mlp_out_start = comp_start + num_sensors
        raw_indices = list(range(raw_start, raw_start + num_sensors))
        comp_indices = list(range(comp_start, comp_start + num_sensors))
        mlp_out_indices = list(range(mlp_out_start, mlp_out_start + num_sensors))
        return raw_indices, comp_indices, mlp_out_indices, num_sensors
    
    # Find indices from header
    raw_indices_dict = {}
    comp_indices_dict = {}
    mlp_out_indices_dict = {}
    
    for i, col in enumerate(header):
        col_lower = col.lower().strip()
        if col_lower.startswith('raw'):
            try:
                # Extract sensor number (raw1, raw2, etc.)
                sensor_num = int(col_lower.replace('raw', ''))
                raw_indices_dict[sensor_num] = i
            except:
                pass
        elif col_lower.startswith('compensated'):
            try:
                # Extract sensor number (compensated1, compensated2, etc.)
                sensor_num = int(col_lower.replace('compensated', ''))
                comp_indices_dict[sensor_num] = i
            except:
                pass
        elif col_lower.startswith('mlp_out'):
            try:
                # Extract sensor number (mlp_out1, mlp_out2, etc.)
                sensor_num = int(col_lower.replace('mlp_out', ''))
                mlp_out_indices_dict[sensor_num] = i
            except:
                pass
    
    # Get number of sensors from found indices
    num_sensors = max(len(raw_indices_dict), len(comp_indices_dict), len(mlp_out_indices_dict), num_sensors)
    
    # Create ordered lists
    raw_indices = [raw_indices_dict.get(i+1, -1) for i in range(num_sensors)]
    comp_indices = [comp_indices_dict.get(i+1, -1) for i in range(num_sensors)]
    mlp_out_indices = [mlp_out_indices_dict.get(i+1, -1) for i in range(num_sensors)]
    
    # Filter out -1 (missing sensors)
    raw_indices = [idx for idx in raw_indices if idx >= 0]
    comp_indices = [idx for idx in comp_indices if idx >= 0]
    mlp_out_indices = [idx for idx in mlp_out_indices if idx >= 0]
    
    return raw_indices, comp_indices, mlp_out_indices, num_sensors


def plot_sensor_data(data, header, output_path=None, num_sensors=8):
    """Plot raw, compensated, and mlp_out values for each sensor."""
    # Find column indices
    raw_indices, comp_indices, mlp_out_indices, num_sensors = find_column_indices(header, num_sensors)
    
    timestamp_idx = 0
    
    # Validate indices
    if len(raw_indices) == 0 or len(comp_indices) == 0:
        # Fallback: assume standard format
        raw_start = 7  # After timestamp (1) + joints (6)
        comp_start = raw_start + num_sensors
        mlp_out_start = comp_start + num_sensors
        
        raw_indices = list(range(raw_start, raw_start + num_sensors))
        comp_indices = list(range(comp_start, comp_start + num_sensors))
        mlp_out_indices = list(range(mlp_out_start, mlp_out_start + num_sensors))
        print(f"Warning: Using fallback column detection. raw={raw_start}-{raw_start+num_sensors-1}, comp={comp_start}-{comp_start+num_sensors-1}, mlp_out={mlp_out_start}-{mlp_out_start+num_sensors-1}")
    else:
        print(f"Found {len(raw_indices)} raw sensors, {len(comp_indices)} compensated sensors, {len(mlp_out_indices)} mlp_out sensors")
    
    # Extract timestamp (relative time in seconds)
    timestamps = data[:, timestamp_idx]
    if timestamps[0] > 1e9:  # Likely absolute timestamp, convert to relative
        timestamps = timestamps - timestamps[0]
    
    # Create subplots: 각 센서마다 2개씩 (raw/compensated, mlp_out)
    # 8개 센서면 4x4 그리드, 4개 센서면 2x4 그리드
    if num_sensors <= 4:
        nrows, ncols = 2, 4  # 2 rows, 4 cols (각 센서마다 2개 subplot)
    else:
        nrows, ncols = 4, 4  # 4 rows, 4 cols (각 센서마다 2개 subplot)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
    axes = axes.flatten()
    
    for sensor_idx in range(num_sensors):
        # 첫 번째 subplot: Raw vs Compensated
        ax1 = axes[sensor_idx * 2]
        
        # 두 번째 subplot: MLP Out
        ax2 = axes[sensor_idx * 2 + 1]
        
        # Get data for this sensor
        if sensor_idx < len(raw_indices) and sensor_idx < len(comp_indices):
            raw_col = raw_indices[sensor_idx]
            comp_col = comp_indices[sensor_idx]
            
            if raw_col >= data.shape[1] or comp_col >= data.shape[1]:
                ax1.text(0.5, 0.5, f'Column index out of range\nraw_col={raw_col}, comp_col={comp_col}', 
                       ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Sensor {sensor_idx+1}: Raw vs Compensated')
                continue
            
            raw_values = data[:, raw_col]
            comp_values = data[:, comp_col]
            
            # Plot 1: Raw vs Compensated
            ax1.plot(timestamps, raw_values, 
                   label=f'Raw {sensor_idx+1}', 
                   color='blue', 
                   alpha=1.0,  # 완전 불투명 (100%)
                   linewidth=2.5,  # 더 두껍게
                   zorder=2)  # 높은 z-order로 변경 (앞에 표시)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Raw Distance (mm)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)
            
            # Create right y-axis for compensated values
            ax1_twin = ax1.twinx()
            ax1_twin.plot(timestamps, comp_values, 
                    label=f'Compensated {sensor_idx+1}', 
                    color='red', 
                    alpha=0.6,  # 60% 투명도로 낮춤 (더 투명하게)
                    linewidth=1.5,
                    zorder=1)  # 낮은 z-order (배경)
            ax1_twin.set_ylabel('Compensated Distance (mm)', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            
            ax1.set_title(f'Sensor {sensor_idx+1}: Raw (left) vs Compensated (right)')
            
            # Plot 2: MLP Out
            if sensor_idx < len(mlp_out_indices):
                mlp_out_col = mlp_out_indices[sensor_idx]
                if mlp_out_col < data.shape[1]:
                    mlp_out_values = data[:, mlp_out_col]
                    ax2.plot(timestamps, mlp_out_values, 
                           label=f'MLP Out {sensor_idx+1}', 
                           color='green', 
                           alpha=0.8,
                           linewidth=2.0)
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('MLP Out (baseline prediction)', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='best', fontsize=9)
                    ax2.set_title(f'Sensor {sensor_idx+1}: MLP Out (Baseline)')
                else:
                    ax2.text(0.5, 0.5, f'MLP Out data not available', 
                           ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(f'Sensor {sensor_idx+1}: MLP Out')
            else:
                ax2.text(0.5, 0.5, f'MLP Out data not available', 
                       ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Sensor {sensor_idx+1}: MLP Out')
        else:
            ax1.text(0.5, 0.5, f'No data for sensor {sensor_idx+1}', 
                   ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'Sensor {sensor_idx+1}: Raw vs Compensated')
            ax2.text(0.5, 0.5, f'No data for sensor {sensor_idx+1}', 
                   ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Sensor {sensor_idx+1}: MLP Out')
    
    # Hide unused subplots
    total_plots = num_sensors * 2
    for idx in range(total_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot compensated sensor data')
    parser.add_argument('log_file', type=str, nargs='?', default=None,
                       help='Log file path (supports glob patterns). If not specified, uses latest file from ~/rb10_Proximity/logs/')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output image file (if not specified, shows interactive plot)')
    parser.add_argument('--num-sensors', type=int, default=8,
                       help='Number of sensors (default: 8)')
    
    args = parser.parse_args()
    
    # If log_file not specified, find latest from default directory
    if args.log_file is None:
        default_log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        pattern = os.path.join(default_log_dir, 'compensated_*.txt')
        log_files = glob.glob(pattern)
        
        if not log_files:
            print(f"Error: No log files found in {default_log_dir}")
            print(f"Please specify a log file: python plot_compensated_data.py <log_file>")
            sys.exit(1)
        
        # Use most recent file
        log_file = sorted(log_files, key=os.path.getmtime, reverse=True)[0]
        print(f"Using latest log file: {os.path.basename(log_file)}")
    else:
        # Handle glob patterns
        log_files = glob.glob(args.log_file)
        if not log_files:
            print(f"Error: No files found matching {args.log_file}")
            sys.exit(1)
        
        # Use the first file (or most recent)
        log_file = sorted(log_files)[-1] if len(log_files) > 1 else log_files[0]
        
        if len(log_files) > 1:
            print(f"Found {len(log_files)} files, using: {log_file}")
    
    try:
        print(f"Loading: {log_file}")
        data, header = load_log_file(log_file)
        print(f"Loaded {len(data)} samples")
        
        if header:
            print(f"Columns: {len(header)}")
            print(f"Header: {header[:10]}...")
        
        # Determine output path
        # If --output is specified, use it; otherwise show plot interactively
        output_path = args.output
        
        plot_sensor_data(data, header, output_path, args.num_sensors)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

