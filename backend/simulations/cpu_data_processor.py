"""
CPU-based Data Processing for Physics Simulations
Replaces cuDF with pandas and numpy for CPU-only operation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import concurrent.futures
import multiprocessing as mp

logger = logging.getLogger(__name__)

class CPUDataProcessor:
    """
    CPU-optimized data processing for physics simulations
    Uses pandas, numpy, and multiprocessing for performance
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or min(mp.cpu_count(), 4)
        logger.info(f"Initialized CPU data processor with {self.n_workers} workers")
    
    def process_simulation_data(self, positions: np.ndarray, velocities: np.ndarray,
                              masses: np.ndarray, time_stamps: np.ndarray) -> pd.DataFrame:
        """
        Process N-body simulation data into structured DataFrame
        
        Args:
            positions: Position data [time_steps, n_bodies, 3]
            velocities: Velocity data [time_steps, n_bodies, 3]
            masses: Body masses [n_bodies]
            time_stamps: Time array [time_steps]
            
        Returns:
            Structured DataFrame with simulation data
        """
        n_steps, n_bodies, _ = positions.shape
        
        # Create structured data
        data_list = []
        
        for t_idx, t in enumerate(time_stamps):
            for body_idx in range(n_bodies):
                row = {
                    'time': t,
                    'body_id': body_idx,
                    'mass': masses[body_idx],
                    'pos_x': positions[t_idx, body_idx, 0],
                    'pos_y': positions[t_idx, body_idx, 1],
                    'pos_z': positions[t_idx, body_idx, 2],
                    'vel_x': velocities[t_idx, body_idx, 0],
                    'vel_y': velocities[t_idx, body_idx, 1],
                    'vel_z': velocities[t_idx, body_idx, 2],
                }
                
                # Compute derived quantities
                row['speed'] = np.sqrt(row['vel_x']**2 + row['vel_y']**2 + row['vel_z']**2)
                row['distance_from_origin'] = np.sqrt(row['pos_x']**2 + row['pos_y']**2 + row['pos_z']**2)
                row['kinetic_energy'] = 0.5 * row['mass'] * row['speed']**2
                
                data_list.append(row)
        
        df = pd.DataFrame(data_list)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add physics-based features
        df = self._add_physics_features(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based derived features"""
        df = df.copy()
        
        # Group by body to compute time-based features
        for body_id in df['body_id'].unique():
            mask = df['body_id'] == body_id
            body_data = df[mask].sort_values('time')
            
            # Compute accelerations (numerical derivatives)
            dt = np.diff(body_data['time'].values)
            dt = np.append(dt, dt[-1])  # Handle last element
            
            dvx_dt = np.gradient(body_data['vel_x'].values, body_data['time'].values)
            dvy_dt = np.gradient(body_data['vel_y'].values, body_data['time'].values)
            dvz_dt = np.gradient(body_data['vel_z'].values, body_data['time'].values)
            
            df.loc[mask, 'acc_x'] = dvx_dt
            df.loc[mask, 'acc_y'] = dvy_dt
            df.loc[mask, 'acc_z'] = dvz_dt
            df.loc[mask, 'acceleration'] = np.sqrt(dvx_dt**2 + dvy_dt**2 + dvz_dt**2)
        
        return df
    
    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add physics-based derived features"""
        df = df.copy()
        
        # Group by time to compute pairwise features
        for time in df['time'].unique():
            time_mask = df['time'] == time
            time_data = df[time_mask].copy()
            
            # Compute center of mass
            total_mass = time_data['mass'].sum()
            com_x = (time_data['mass'] * time_data['pos_x']).sum() / total_mass
            com_y = (time_data['mass'] * time_data['pos_y']).sum() / total_mass
            com_z = (time_data['mass'] * time_data['pos_z']).sum() / total_mass
            
            # Distance from center of mass
            df.loc[time_mask, 'dist_from_com'] = np.sqrt(
                (time_data['pos_x'] - com_x)**2 + 
                (time_data['pos_y'] - com_y)**2 + 
                (time_data['pos_z'] - com_z)**2
            )
            
            # Angular momentum (simplified)
            df.loc[time_mask, 'angular_momentum'] = (
                time_data['mass'] * time_data['distance_from_origin'] * time_data['speed']
            )
        
        return df
    
    def compute_energy_conservation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute total energy over time to check conservation
        """
        G = 6.674e-11
        energy_data = []
        
        for time in df['time'].unique():
            time_data = df[df['time'] == time]
            
            # Kinetic energy
            total_ke = time_data['kinetic_energy'].sum()
            
            # Potential energy (simplified pairwise)
            total_pe = 0.0
            bodies = time_data.to_dict('records')
            
            for i, body1 in enumerate(bodies):
                for j, body2 in enumerate(bodies[i+1:], i+1):
                    r = np.sqrt(
                        (body1['pos_x'] - body2['pos_x'])**2 +
                        (body1['pos_y'] - body2['pos_y'])**2 +
                        (body1['pos_z'] - body2['pos_z'])**2
                    ) + 1e-10
                    
                    total_pe += -G * body1['mass'] * body2['mass'] / r
            
            energy_data.append({
                'time': time,
                'kinetic_energy': total_ke,
                'potential_energy': total_pe,
                'total_energy': total_ke + total_pe
            })
        
        return pd.DataFrame(energy_data)
    
    def aggregate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute aggregate statistics for the simulation
        """
        stats = {}
        
        # Overall statistics
        stats['n_bodies'] = df['body_id'].nunique()
        stats['n_timesteps'] = df['time'].nunique()
        stats['simulation_duration'] = df['time'].max() - df['time'].min()
        
        # Energy statistics
        energy_df = self.compute_energy_conservation(df)
        stats['energy_conservation'] = {
            'initial_energy': energy_df['total_energy'].iloc[0],
            'final_energy': energy_df['total_energy'].iloc[-1],
            'energy_drift': abs(energy_df['total_energy'].iloc[-1] - energy_df['total_energy'].iloc[0]),
            'max_energy': energy_df['total_energy'].max(),
            'min_energy': energy_df['total_energy'].min()
        }
        
        # Per-body statistics
        body_stats = {}
        for body_id in df['body_id'].unique():
            body_data = df[df['body_id'] == body_id]
            body_stats[f'body_{body_id}'] = {
                'mass': body_data['mass'].iloc[0],
                'max_speed': body_data['speed'].max(),
                'min_speed': body_data['speed'].min(),
                'avg_speed': body_data['speed'].mean(),
                'max_distance': body_data['distance_from_origin'].max(),
                'min_distance': body_data['distance_from_origin'].min(),
                'trajectory_length': self._compute_trajectory_length(body_data)
            }
        
        stats['body_statistics'] = body_stats
        
        return stats
    
    def _compute_trajectory_length(self, body_data: pd.DataFrame) -> float:
        """Compute the total length of a body's trajectory"""
        body_data = body_data.sort_values('time')
        positions = body_data[['pos_x', 'pos_y', 'pos_z']].values
        
        if len(positions) < 2:
            return 0.0
        
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        return distances.sum()
    
    def export_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Export processed data to CSV"""
        df.to_csv(filepath, index=False)
        logger.info(f"Exported data to {filepath}")
    
    def export_to_json(self, data: Dict[str, Any], filepath: str) -> None:
        """Export statistics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported statistics to {filepath}")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        """
        df_optimized = df.copy()
        
        # Downcast numeric types
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert to categorical where appropriate
        if 'body_id' in df_optimized.columns:
            df_optimized['body_id'] = df_optimized['body_id'].astype('category')
        
        memory_usage_before = df.memory_usage(deep=True).sum()
        memory_usage_after = df_optimized.memory_usage(deep=True).sum()
        
        logger.info(f"Memory optimization: {memory_usage_before/1024/1024:.2f}MB -> {memory_usage_after/1024/1024:.2f}MB")
        
        return df_optimized
    
    def parallel_process_chunks(self, data_chunks: List[np.ndarray], 
                              process_func, *args, **kwargs) -> List[Any]:
        """
        Process data chunks in parallel using multiprocessing
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_func, chunk, *args, **kwargs) 
                      for chunk in data_chunks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results
    
    def create_summary_report(self, df: pd.DataFrame) -> str:
        """
        Create a human-readable summary report
        """
        stats = self.aggregate_statistics(df)
        
        report = f"""
Physics Simulation Data Summary Report
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Simulation Overview:
- Number of bodies: {stats['n_bodies']}
- Number of timesteps: {stats['n_timesteps']}
- Simulation duration: {stats['simulation_duration']:.2f} time units

Energy Conservation:
- Initial total energy: {stats['energy_conservation']['initial_energy']:.2e}
- Final total energy: {stats['energy_conservation']['final_energy']:.2e}
- Energy drift: {stats['energy_conservation']['energy_drift']:.2e}

Body Statistics:
"""
        
        for body_name, body_stat in stats['body_statistics'].items():
            report += f"""
{body_name}:
  - Mass: {body_stat['mass']:.2e}
  - Speed range: {body_stat['min_speed']:.2f} - {body_stat['max_speed']:.2f}
  - Average speed: {body_stat['avg_speed']:.2f}
  - Distance range: {body_stat['min_distance']:.2f} - {body_stat['max_distance']:.2f}
  - Trajectory length: {body_stat['trajectory_length']:.2f}
"""
        
        return report
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the data processor"""
        return {
            'framework': 'pandas + numpy',
            'n_workers': self.n_workers,
            'cpu_count': mp.cpu_count(),
            'processing_backend': 'CPU multiprocessing'
        }
