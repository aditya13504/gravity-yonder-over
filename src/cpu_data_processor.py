"""
CPU-based Data Processing for Educational Physics Applications
Replaces cuDF with pandas and numpy for CPU-only operation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class CuDFDataProcessor:
    """
    CPU-based data processor that mimics cuDF functionality
    Uses pandas and numpy for all operations
    """
    
    def __init__(self):
        self.framework = "pandas"
        self.device = "cpu"
        logger.info("Initialized CPU-based data processor (pandas backend)")
    
    def create_physics_dataframe(self, simulation_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create a structured DataFrame from physics simulation data
        
        Args:
            simulation_data: Dictionary containing arrays of physics data
            
        Returns:
            Structured pandas DataFrame
        """
        # Extract data arrays
        times = simulation_data.get('times', np.array([]))
        positions = simulation_data.get('positions', np.array([]))
        velocities = simulation_data.get('velocities', np.array([]))
        masses = simulation_data.get('masses', np.array([]))
        
        if len(positions.shape) == 3:  # [time, bodies, dims]
            n_times, n_bodies, n_dims = positions.shape
        else:
            n_times, n_bodies, n_dims = len(times), 1, 3
            positions = positions.reshape(n_times, n_bodies, n_dims)
            velocities = velocities.reshape(n_times, n_bodies, n_dims)
        
        # Create structured data
        data_list = []
        
        for t_idx in range(n_times):
            time = times[t_idx] if len(times) > t_idx else t_idx * 0.01
            
            for body_idx in range(n_bodies):
                mass = masses[body_idx] if len(masses) > body_idx else 1.0
                
                pos = positions[t_idx, body_idx]
                vel = velocities[t_idx, body_idx]
                
                row = {
                    'time': time,
                    'body_id': body_idx,
                    'mass': mass,
                    'pos_x': pos[0],
                    'pos_y': pos[1],
                    'pos_z': pos[2],
                    'vel_x': vel[0],
                    'vel_y': vel[1],
                    'vel_z': vel[2],
                }
                
                # Compute derived quantities
                row['speed'] = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                row['distance'] = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                row['kinetic_energy'] = 0.5 * mass * row['speed']**2
                
                data_list.append(row)
        
        df = pd.DataFrame(data_list)
        return df
    
    def compute_aggregated_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute aggregated statistics from physics data
        """
        stats = {}
        
        # Basic statistics
        stats['total_bodies'] = df['body_id'].nunique()
        stats['total_timesteps'] = df['time'].nunique()
        stats['time_range'] = {
            'min': df['time'].min(),
            'max': df['time'].max(),
            'duration': df['time'].max() - df['time'].min()
        }
        
        # Speed statistics
        stats['speed_stats'] = {
            'min': df['speed'].min(),
            'max': df['speed'].max(),
            'mean': df['speed'].mean(),
            'std': df['speed'].std()
        }
        
        # Distance statistics
        stats['distance_stats'] = {
            'min': df['distance'].min(),
            'max': df['distance'].max(),
            'mean': df['distance'].mean(),
            'std': df['distance'].std()
        }
        
        # Energy statistics
        stats['energy_stats'] = {
            'total_kinetic': df.groupby('time')['kinetic_energy'].sum().mean(),
            'kinetic_variation': df.groupby('time')['kinetic_energy'].sum().std()
        }
        
        # Per-body statistics
        body_stats = {}
        for body_id in df['body_id'].unique():
            body_data = df[df['body_id'] == body_id]
            body_stats[f'body_{body_id}'] = {
                'mass': body_data['mass'].iloc[0],
                'avg_speed': body_data['speed'].mean(),
                'max_speed': body_data['speed'].max(),
                'avg_distance': body_data['distance'].mean(),
                'trajectory_length': self._compute_trajectory_length(body_data)
            }
        
        stats['body_statistics'] = body_stats
        
        return stats
    
    def _compute_trajectory_length(self, body_data: pd.DataFrame) -> float:
        """Compute total trajectory length for a body"""
        body_data = body_data.sort_values('time')
        positions = body_data[['pos_x', 'pos_y', 'pos_z']].values
        
        if len(positions) < 2:
            return 0.0
        
        # Compute distances between consecutive points
        diffs = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return distances.sum()
    
    def filter_by_time_range(self, df: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
        """Filter DataFrame by time range"""
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        return df[mask].copy()
    
    def filter_by_body_ids(self, df: pd.DataFrame, body_ids: List[int]) -> pd.DataFrame:
        """Filter DataFrame by specific body IDs"""
        mask = df['body_id'].isin(body_ids)
        return df[mask].copy()
    
    def compute_pairwise_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise distances between all bodies at each time step
        """
        distance_data = []
        
        for time in df['time'].unique():
            time_data = df[df['time'] == time]
            bodies = time_data.to_dict('records')
            
            for i, body1 in enumerate(bodies):
                for j, body2 in enumerate(bodies[i+1:], i+1):
                    distance = np.sqrt(
                        (body1['pos_x'] - body2['pos_x'])**2 +
                        (body1['pos_y'] - body2['pos_y'])**2 +
                        (body1['pos_z'] - body2['pos_z'])**2
                    )
                    
                    distance_data.append({
                        'time': time,
                        'body1_id': body1['body_id'],
                        'body2_id': body2['body_id'],
                        'distance': distance,
                        'relative_speed': abs(body1['speed'] - body2['speed'])
                    })
        
        return pd.DataFrame(distance_data)
    
    def compute_center_of_mass(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute center of mass over time"""
        com_data = []
        
        for time in df['time'].unique():
            time_data = df[df['time'] == time]
            total_mass = time_data['mass'].sum()
            
            com_x = (time_data['mass'] * time_data['pos_x']).sum() / total_mass
            com_y = (time_data['mass'] * time_data['pos_y']).sum() / total_mass
            com_z = (time_data['mass'] * time_data['pos_z']).sum() / total_mass
            
            com_data.append({
                'time': time,
                'com_x': com_x,
                'com_y': com_y,
                'com_z': com_z,
                'total_mass': total_mass
            })
        
        return pd.DataFrame(com_data)
    
    def export_to_formats(self, df: pd.DataFrame, base_filename: str, 
                         formats: List[str] = ['csv', 'json', 'parquet']) -> Dict[str, str]:
        """
        Export DataFrame to multiple formats
        
        Args:
            df: DataFrame to export
            base_filename: Base filename without extension
            formats: List of formats to export ['csv', 'json', 'parquet']
            
        Returns:
            Dictionary mapping format to filename
        """
        exported_files = {}
        
        if 'csv' in formats:
            csv_file = f"{base_filename}.csv"
            df.to_csv(csv_file, index=False)
            exported_files['csv'] = csv_file
            logger.info(f"Exported to CSV: {csv_file}")
        
        if 'json' in formats:
            json_file = f"{base_filename}.json"
            df.to_json(json_file, orient='records', indent=2)
            exported_files['json'] = json_file
            logger.info(f"Exported to JSON: {json_file}")
        
        if 'parquet' in formats:
            try:
                parquet_file = f"{base_filename}.parquet"
                df.to_parquet(parquet_file, index=False)
                exported_files['parquet'] = parquet_file
                logger.info(f"Exported to Parquet: {parquet_file}")
            except ImportError:
                logger.warning("Parquet export requires pyarrow or fastparquet")
        
        return exported_files
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        df_optimized = df.copy()
        
        # Downcast numeric types
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert categorical columns
        if 'body_id' in df_optimized.columns:
            df_optimized['body_id'] = df_optimized['body_id'].astype('category')
        
        # Report memory savings
        memory_before = df.memory_usage(deep=True).sum()
        memory_after = df_optimized.memory_usage(deep=True).sum()
        savings = (memory_before - memory_after) / memory_before * 100
        
        logger.info(f"Memory optimization: {savings:.1f}% reduction "
                   f"({memory_before/1024/1024:.1f}MB -> {memory_after/1024/1024:.1f}MB)")
        
        return df_optimized
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-series based features to the DataFrame"""
        df_enhanced = df.copy()
        
        # Sort by body and time
        df_enhanced = df_enhanced.sort_values(['body_id', 'time'])
        
        # Add time-based features for each body
        for body_id in df_enhanced['body_id'].unique():
            mask = df_enhanced['body_id'] == body_id
            body_data = df_enhanced[mask].copy()
            
            # Velocity changes (acceleration proxy)
            body_data['speed_change'] = body_data['speed'].diff()
            body_data['distance_change'] = body_data['distance'].diff()
            
            # Rolling statistics
            window = min(10, len(body_data))
            if window > 1:
                body_data['speed_rolling_mean'] = body_data['speed'].rolling(window=window).mean()
                body_data['speed_rolling_std'] = body_data['speed'].rolling(window=window).std()
            
            # Update the main DataFrame
            df_enhanced.loc[mask, 'speed_change'] = body_data['speed_change']
            df_enhanced.loc[mask, 'distance_change'] = body_data['distance_change']
            if window > 1:
                df_enhanced.loc[mask, 'speed_rolling_mean'] = body_data['speed_rolling_mean']
                df_enhanced.loc[mask, 'speed_rolling_std'] = body_data['speed_rolling_std']
        
        # Fill NaN values
        df_enhanced = df_enhanced.fillna(0)
        
        return df_enhanced
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the data processor"""
        return {
            'framework': self.framework,
            'device': self.device,
            'backend': 'pandas + numpy',
            'parallel_processing': 'CPU multiprocessing available',
            'memory_efficient': True,
            'gpu_acceleration': False
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and report issues"""
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            issues.append(f"Missing values found: {missing_counts.to_dict()}")
        
        # Check for infinite or extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                issues.append(f"Infinite values found in column: {col}")
            
            if (np.abs(df[col]) > 1e10).any():
                issues.append(f"Extremely large values found in column: {col}")
        
        # Check for duplicate time-body combinations
        duplicates = df.duplicated(subset=['time', 'body_id']).sum()
        if duplicates > 0:
            issues.append(f"Duplicate time-body combinations: {duplicates}")
        
        # Check for negative masses
        if (df['mass'] < 0).any():
            issues.append("Negative masses found")
        
        return {
            'issues': issues,
            'data_quality': 'good' if not issues else 'issues_found',
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
