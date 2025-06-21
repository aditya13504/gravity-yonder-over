"""
cuDF Data Handler for High-Performance Gravity Simulations
Provides GPU-accelerated data processing for large-scale physics simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Try to import cuDF for GPU acceleration
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("cuDF GPU acceleration available")
except ImportError:
    CUDF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("cuDF not available, falling back to pandas CPU processing")
    # Create dummy cudf module for compatibility
    class cudf:
        @staticmethod
        def DataFrame(*args, **kwargs):
            return pd.DataFrame(*args, **kwargs)
        
        @staticmethod
        def Series(*args, **kwargs):
            return pd.Series(*args, **kwargs)

class GravityDataProcessor:
    """
    High-performance data processor for gravity simulations
    Uses cuDF when available, falls back to pandas
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUDF_AVAILABLE
        self.backend = "cuDF (GPU)" if self.use_gpu else "pandas (CPU)"
        logger.info(f"Initialized GravityDataProcessor with {self.backend}")
        
    def create_dataframe(self, data: Union[Dict, List, np.ndarray]) -> Union[cudf.DataFrame, pd.DataFrame]:
        """Create a DataFrame using the appropriate backend"""
        if self.use_gpu:
            return cudf.DataFrame(data)
        else:
            return pd.DataFrame(data)
    
    def create_series(self, data: Union[List, np.ndarray], name: str = None) -> Union[cudf.Series, pd.Series]:
        """Create a Series using the appropriate backend"""
        if self.use_gpu:
            return cudf.Series(data, name=name)
        else:
            return pd.Series(data, name=name)
    
    def process_trajectory_data(self, trajectory_dict: Dict[str, np.ndarray]) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Process trajectory data into structured DataFrame
        
        Args:
            trajectory_dict: Dictionary with body names as keys and position arrays as values
            
        Returns:
            Structured DataFrame with trajectory data
        """
        processed_data = []
        
        for body_name, positions in trajectory_dict.items():
            for i, pos in enumerate(positions):
                processed_data.append({
                    'body': body_name,
                    'time_step': i,
                    'x': pos[0] if len(pos) > 0 else 0,
                    'y': pos[1] if len(pos) > 1 else 0,
                    'z': pos[2] if len(pos) > 2 else 0,
                    'distance_from_origin': np.linalg.norm(pos)
                })
        
        return self.create_dataframe(processed_data)
    
    def calculate_orbital_parameters(self, df: Union[cudf.DataFrame, pd.DataFrame], 
                                   central_body: str = None) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Calculate orbital parameters for trajectory data
        
        Args:
            df: Trajectory DataFrame
            central_body: Name of central body (if None, assumes origin)
            
        Returns:
            DataFrame with orbital parameters
        """
        if central_body:
            # Calculate relative positions
            central_positions = df[df['body'] == central_body][['x', 'y', 'z', 'time_step']]
            
            orbital_data = []
            for body in df['body'].unique():
                if body != central_body:
                    body_data = df[df['body'] == body]
                    
                    # Merge with central body positions
                    if self.use_gpu:
                        merged = body_data.merge(central_positions, on='time_step', suffixes=('', '_central'))
                    else:
                        merged = body_data.merge(central_positions, on='time_step', suffixes=('', '_central'))
                    
                    # Calculate relative positions
                    merged['rel_x'] = merged['x'] - merged['x_central']
                    merged['rel_y'] = merged['y'] - merged['y_central']
                    merged['rel_z'] = merged['z'] - merged['z_central']
                    merged['orbital_radius'] = (merged['rel_x']**2 + merged['rel_y']**2 + merged['rel_z']**2)**0.5
                    
                    orbital_data.append(merged)
            
            if orbital_data:
                return self._concat_dataframes(orbital_data)
            else:
                return self.create_dataframe([])
        else:
            # Calculate parameters relative to origin
            df_copy = df.copy()
            df_copy['orbital_radius'] = (df_copy['x']**2 + df_copy['y']**2 + df_copy['z']**2)**0.5
            df_copy['velocity_x'] = df_copy.groupby('body')['x'].diff()
            df_copy['velocity_y'] = df_copy.groupby('body')['y'].diff()
            df_copy['velocity_z'] = df_copy.groupby('body')['z'].diff()
            df_copy['speed'] = (df_copy['velocity_x']**2 + df_copy['velocity_y']**2 + df_copy['velocity_z']**2)**0.5
            
            return df_copy
    
    def calculate_energy_momentum(self, df: Union[cudf.DataFrame, pd.DataFrame], 
                                masses: Dict[str, float]) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Calculate energy and momentum for each body
        
        Args:
            df: Trajectory DataFrame with velocity data
            masses: Dictionary mapping body names to masses
            
        Returns:
            DataFrame with energy and momentum calculations
        """
        df_copy = df.copy()
        
        # Add mass column
        if self.use_gpu:
            df_copy['mass'] = df_copy['body'].map(masses)
        else:
            df_copy['mass'] = df_copy['body'].map(masses)
        
        # Calculate kinetic energy
        if 'speed' in df_copy.columns:
            df_copy['kinetic_energy'] = 0.5 * df_copy['mass'] * df_copy['speed']**2
        
        # Calculate momentum
        if all(col in df_copy.columns for col in ['velocity_x', 'velocity_y', 'velocity_z']):
            df_copy['momentum_x'] = df_copy['mass'] * df_copy['velocity_x']
            df_copy['momentum_y'] = df_copy['mass'] * df_copy['velocity_y']
            df_copy['momentum_z'] = df_copy['mass'] * df_copy['velocity_z']
            df_copy['momentum_magnitude'] = (df_copy['momentum_x']**2 + 
                                           df_copy['momentum_y']**2 + 
                                           df_copy['momentum_z']**2)**0.5
        
        return df_copy
    
    def calculate_gravitational_potential(self, df: Union[cudf.DataFrame, pd.DataFrame],
                                        masses: Dict[str, float], 
                                        G: float = 6.67430e-11) -> Union[cudf.DataFrame, pd.DataFrame]:
        """
        Calculate gravitational potential energy
        
        Args:
            df: Trajectory DataFrame
            masses: Dictionary mapping body names to masses
            G: Gravitational constant
            
        Returns:
            DataFrame with potential energy calculations
        """
        df_with_potential = df.copy()
        potential_energies = []
        
        # Group by time step for pairwise calculations
        time_steps = df['time_step'].unique()
        
        for time_step in time_steps:
            step_data = df[df['time_step'] == time_step]
            bodies = step_data['body'].unique()
            
            total_potential = 0
            for i, body1 in enumerate(bodies):
                for j, body2 in enumerate(bodies):
                    if i < j:  # Avoid double counting
                        pos1 = step_data[step_data['body'] == body1][['x', 'y', 'z']].iloc[0]
                        pos2 = step_data[step_data['body'] == body2][['x', 'y', 'z']].iloc[0]
                        
                        distance = ((pos1['x'] - pos2['x'])**2 + 
                                  (pos1['y'] - pos2['y'])**2 + 
                                  (pos1['z'] - pos2['z'])**2)**0.5
                        
                        if distance > 0:
                            potential = -G * masses[body1] * masses[body2] / distance
                            total_potential += potential
            
            potential_energies.append(total_potential)
        
        # Create potential energy lookup
        potential_lookup = dict(zip(time_steps, potential_energies))
        
        if self.use_gpu:
            df_with_potential['gravitational_potential'] = df_with_potential['time_step'].map(potential_lookup)
        else:
            df_with_potential['gravitational_potential'] = df_with_potential['time_step'].map(potential_lookup)
        
        return df_with_potential
    
    def analyze_stability(self, df: Union[cudf.DataFrame, pd.DataFrame]) -> Dict[str, Union[float, bool]]:
        """
        Analyze orbital stability metrics
        
        Args:
            df: Trajectory DataFrame with orbital parameters
            
        Returns:
            Dictionary with stability analysis
        """
        analysis = {}
        
        if 'orbital_radius' in df.columns:
            # Calculate orbital radius statistics
            if self.use_gpu:
                radius_stats = df['orbital_radius'].describe()
                analysis['radius_mean'] = radius_stats['mean']
                analysis['radius_std'] = radius_stats['std']
                analysis['radius_min'] = radius_stats['min']
                analysis['radius_max'] = radius_stats['max']
            else:
                analysis['radius_mean'] = df['orbital_radius'].mean()
                analysis['radius_std'] = df['orbital_radius'].std()
                analysis['radius_min'] = df['orbital_radius'].min()
                analysis['radius_max'] = df['orbital_radius'].max()
            
            # Stability criteria
            analysis['is_stable'] = analysis['radius_std'] / analysis['radius_mean'] < 0.1
            analysis['eccentricity_estimate'] = (analysis['radius_max'] - analysis['radius_min']) / (analysis['radius_max'] + analysis['radius_min'])
        
        if 'kinetic_energy' in df.columns and 'gravitational_potential' in df.columns:
            df_copy = df.copy()
            df_copy['total_energy'] = df_copy['kinetic_energy'] + df_copy['gravitational_potential']
            
            if self.use_gpu:
                energy_stats = df_copy['total_energy'].describe()
                analysis['energy_conservation'] = energy_stats['std'] / abs(energy_stats['mean']) if energy_stats['mean'] != 0 else float('inf')
            else:
                analysis['energy_conservation'] = df_copy['total_energy'].std() / abs(df_copy['total_energy'].mean()) if df_copy['total_energy'].mean() != 0 else float('inf')
        
        return analysis
    
    def export_to_csv(self, df: Union[cudf.DataFrame, pd.DataFrame], filename: str):
        """Export DataFrame to CSV file"""
        if self.use_gpu:
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)
    
    def export_to_parquet(self, df: Union[cudf.DataFrame, pd.DataFrame], filename: str):
        """Export DataFrame to Parquet file (more efficient for large datasets)"""
        if self.use_gpu:
            df.to_parquet(filename)
        else:
            df.to_parquet(filename)
    
    def _concat_dataframes(self, dfs: List[Union[cudf.DataFrame, pd.DataFrame]]) -> Union[cudf.DataFrame, pd.DataFrame]:
        """Concatenate list of DataFrames using appropriate backend"""
        if self.use_gpu:
            return cudf.concat(dfs, ignore_index=True)
        else:
            return pd.concat(dfs, ignore_index=True)
    
    def optimize_memory(self, df: Union[cudf.DataFrame, pd.DataFrame]) -> Union[cudf.DataFrame, pd.DataFrame]:
        """Optimize DataFrame memory usage"""
        if self.use_gpu:
            # For cuDF, memory is already optimized on GPU
            return df
        else:
            # For pandas, optimize dtypes
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            return df

# Utility functions for common operations
def load_simulation_data(filepath: str, use_gpu: bool = True) -> Union[cudf.DataFrame, pd.DataFrame]:
    """Load simulation data from file"""
    processor = GravityDataProcessor(use_gpu=use_gpu)
    
    if filepath.endswith('.csv'):
        if processor.use_gpu:
            return cudf.read_csv(filepath)
        else:
            return pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        if processor.use_gpu:
            return cudf.read_parquet(filepath)
        else:
            return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def batch_process_simulations(simulation_files: List[str], 
                            output_dir: str = "./processed/",
                            use_gpu: bool = True) -> List[str]:
    """
    Batch process multiple simulation files
    
    Args:
        simulation_files: List of simulation file paths
        output_dir: Directory to save processed files
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        List of output file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    processor = GravityDataProcessor(use_gpu=use_gpu)
    output_files = []
    
    for i, filepath in enumerate(simulation_files):
        logger.info(f"Processing file {i+1}/{len(simulation_files)}: {filepath}")
        
        try:
            # Load data
            df = load_simulation_data(filepath, use_gpu=use_gpu)
            
            # Process data
            df = processor.optimize_memory(df)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(output_dir, f"{base_name}_processed.parquet")
            
            # Export processed data
            processor.export_to_parquet(df, output_path)
            output_files.append(output_path)
            
            logger.info(f"Processed and saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            continue
    
    return output_files

# Example usage and testing
if __name__ == "__main__":
    # Test the data processor
    processor = GravityDataProcessor(use_gpu=True)
    
    # Create sample trajectory data
    sample_data = {
        'Earth': np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]),
        'Moon': np.array([[384400000, 0, 0], [384400000.1, 1000, 0], [384400000.2, 2000, 0]])
    }
    
    # Process the data
    df = processor.process_trajectory_data(sample_data)
    print(f"Created DataFrame with {len(df)} rows using {processor.backend}")
    print(df.head())
    
    # Calculate orbital parameters
    orbital_df = processor.calculate_orbital_parameters(df, central_body='Earth')
    print(f"Orbital parameters calculated")
    
    # Calculate energy and momentum
    masses = {'Earth': 5.972e24, 'Moon': 7.342e22}
    energy_df = processor.calculate_energy_momentum(orbital_df, masses)
    print("Energy and momentum calculated")
    
    # Analyze stability
    stability = processor.analyze_stability(energy_df)
    print(f"Stability analysis: {stability}")
