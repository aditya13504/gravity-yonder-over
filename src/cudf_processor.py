"""
cuDF Data Processing Pipeline for High-Performance Physics Simulations
Handles large-scale simulation data processing with GPU acceleration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    # Try to import cuDF for GPU-accelerated data processing
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
    logger.info("✅ cuDF available for GPU-accelerated data processing")
    
    # cuDF-specific optimizations
    def create_cudf_dataframe(data: Union[Dict, np.ndarray, pd.DataFrame]) -> cudf.DataFrame:
        """Create cuDF DataFrame from various input types"""
        if isinstance(data, dict):
            return cudf.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return cudf.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return cudf.from_pandas(data)
        else:
            return cudf.DataFrame(data)
    
    def cudf_to_pandas(df: cudf.DataFrame) -> pd.DataFrame:
        """Convert cuDF DataFrame to pandas for compatibility"""
        return df.to_pandas()
    
except ImportError:
    CUDF_AVAILABLE = False
    logger.warning("⚠️ cuDF not available. Using pandas fallback for data processing.")
    
    # Fallback to pandas
    import pandas as pd
    
    def create_cudf_dataframe(data: Union[Dict, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fallback: Create pandas DataFrame"""
        if isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data)
    
    def cudf_to_pandas(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: Return as-is since already pandas"""
        return df

class CuDFDataProcessor:
    """
    High-performance data processor using cuDF for physics simulations
    Handles large datasets efficiently with GPU acceleration when available
    """
    
    def __init__(self):
        self.cudf_available = CUDF_AVAILABLE
        self.processing_stats = {
            "total_operations": 0,
            "gpu_operations": 0,
            "total_processing_time": 0.0,
            "data_points_processed": 0
        }
        
        logger.info(f"🔧 CuDF Processor initialized (GPU: {self.cudf_available})")
    
    def process_simulation_data(self, raw_data: Dict[str, np.ndarray], 
                              operation_type: str = "physics") -> Dict[str, Any]:
        """
        Process raw simulation data with cuDF acceleration
        
        Args:
            raw_data: Dictionary containing raw simulation arrays
            operation_type: Type of processing (physics, orbital, relativistic)
        
        Returns:
            Processed data ready for visualization
        """
        start_time = time.time()
        self.processing_stats["total_operations"] += 1
        
        try:
            if operation_type == "physics":
                result = self._process_physics_data(raw_data)
            elif operation_type == "orbital":
                result = self._process_orbital_data(raw_data)
            elif operation_type == "relativistic":
                result = self._process_relativistic_data(raw_data)
            else:
                result = self._process_generic_data(raw_data)
            
            processing_time = time.time() - start_time
            self.processing_stats["total_processing_time"] += processing_time
            
            if self.cudf_available:
                self.processing_stats["gpu_operations"] += 1
            
            # Count data points
            data_points = sum(len(arr) if hasattr(arr, '__len__') else 1 
                            for arr in raw_data.values() if isinstance(arr, np.ndarray))
            self.processing_stats["data_points_processed"] += data_points
            
            logger.info(f"⚡ Processed {operation_type} data in {processing_time:.3f}s "
                       f"({data_points:,} points)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {operation_type} data: {e}")
            return self._fallback_processing(raw_data)
    
    def _process_physics_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process gravitational physics simulation data"""
        
        # Create cuDF dataframe for processing
        if "coordinates" in data:
            coords = data["coordinates"]
            df_data = {
                "x": coords["x"].flatten(),
                "y": coords["y"].flatten(), 
                "z": coords["z"].flatten(),
                "phi": data["phi"].flatten(),
                "rho": data.get("rho", np.zeros_like(data["phi"])).flatten()
            }
            
            if "fx" in data:
                df_data.update({
                    "fx": data["fx"].flatten(),
                    "fy": data["fy"].flatten(),
                    "fz": data["fz"].flatten()
                })
        else:
            # Assume structured data
            df_data = {key: val.flatten() if hasattr(val, 'flatten') else val 
                      for key, val in data.items()}
        
        df = create_cudf_dataframe(df_data)
        
        # Compute derived quantities
        if "x" in df.columns and "y" in df.columns and "z" in df.columns:
            df["r"] = self._compute_distance(df["x"], df["y"], df["z"])
            df["phi_normalized"] = df["phi"] / df["phi"].abs().max()
            
            # Compute force magnitude
            if all(col in df.columns for col in ["fx", "fy", "fz"]):
                df["force_magnitude"] = self._compute_distance(df["fx"], df["fy"], df["fz"])
        
        # Statistical analysis
        stats = self._compute_statistical_summary(df, ["phi", "rho", "r"])
        
        # Prepare for visualization
        viz_data = self._prepare_visualization_data(df, "physics")
        
        return {
            "processed_dataframe": cudf_to_pandas(df),
            "statistics": stats,
            "visualization_data": viz_data,
            "processing_info": {
                "method": "cuDF" if self.cudf_available else "pandas",
                "data_points": len(df),
                "columns": list(df.columns)
            }
        }
    
    def _process_orbital_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process orbital mechanics simulation data"""
        
        positions = data.get("positions", np.array([]))
        velocities = data.get("velocities", np.array([]))
        times = data.get("times", np.array([]))
        
        if len(positions) == 0:
            return self._fallback_processing(data)
        
        # Create orbital dataframe
        df_data = {
            "t": times,
            "x": positions[:, 0],
            "y": positions[:, 1], 
            "z": positions[:, 2],
            "vx": velocities[:, 0],
            "vy": velocities[:, 1],
            "vz": velocities[:, 2]
        }
        
        df = create_cudf_dataframe(df_data)
        
        # Compute orbital elements
        df["r"] = self._compute_distance(df["x"], df["y"], df["z"])
        df["v"] = self._compute_distance(df["vx"], df["vy"], df["vz"])
        
        # Energy and angular momentum
        GM = data.get("GM", 3.986e14)  # Earth's GM
        df["kinetic_energy"] = 0.5 * df["v"]**2
        df["potential_energy"] = -GM / df["r"]
        df["total_energy"] = df["kinetic_energy"] + df["potential_energy"]
        
        # Angular momentum components
        df["Lx"] = df["y"] * df["vz"] - df["z"] * df["vy"]
        df["Ly"] = df["z"] * df["vx"] - df["x"] * df["vz"]
        df["Lz"] = df["x"] * df["vy"] - df["y"] * df["vx"]
        df["L_magnitude"] = self._compute_distance(df["Lx"], df["Ly"], df["Lz"])
        
        # Detect orbital characteristics
        orbital_analysis = self._analyze_orbital_characteristics(df)
        
        # Statistical summary
        stats = self._compute_statistical_summary(df, ["r", "v", "total_energy", "L_magnitude"])
        
        # Visualization data
        viz_data = self._prepare_visualization_data(df, "orbital")
        
        return {
            "processed_dataframe": cudf_to_pandas(df),
            "orbital_analysis": orbital_analysis,
            "statistics": stats,
            "visualization_data": viz_data,
            "processing_info": {
                "method": "cuDF" if self.cudf_available else "pandas",
                "data_points": len(df),
                "orbital_period": orbital_analysis.get("period", "N/A")
            }
        }
    
    def _process_relativistic_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process relativistic physics simulation data"""
        
        coords = data.get("coordinates", {})
        if "r" not in coords or "t" not in coords:
            return self._fallback_processing(data)
        
        # Create relativistic dataframe
        df_data = {
            "r": coords["r"].flatten(),
            "t": coords["t"].flatten(),
            "metric_tt": data.get("metric_tt", np.ones_like(coords["r"])).flatten(),
            "metric_rr": data.get("metric_rr", np.ones_like(coords["r"])).flatten(),
            "potential": data.get("potential", np.zeros_like(coords["r"])).flatten()
        }
        
        df = create_cudf_dataframe(df_data)
        
        # Compute relativistic effects
        rs = data.get("schwarzschild_radius", 2.95e3)  # Default for 1 solar mass
        
        df["rs_ratio"] = rs / df["r"]
        df["time_dilation"] = 1 / self._safe_sqrt(1 - df["rs_ratio"])
        df["space_contraction"] = self._safe_sqrt(1 - df["rs_ratio"])
        df["redshift"] = self._safe_sqrt((1 - df["rs_ratio"]) / (1 + df["rs_ratio"]))
        
        # Photon sphere and ISCO
        photon_sphere = 1.5 * rs
        isco = 3 * rs  # Innermost stable circular orbit
        
        df["within_photon_sphere"] = df["r"] < photon_sphere
        df["within_isco"] = df["r"] < isco
        df["beyond_event_horizon"] = df["r"] > rs
        
        # Relativistic analysis
        relativistic_analysis = self._analyze_relativistic_effects(df, rs)
        
        # Statistical summary
        stats = self._compute_statistical_summary(df, ["time_dilation", "redshift", "potential"])
        
        # Visualization data
        viz_data = self._prepare_visualization_data(df, "relativistic")
        
        return {
            "processed_dataframe": cudf_to_pandas(df),
            "relativistic_analysis": relativistic_analysis,
            "statistics": stats,
            "visualization_data": viz_data,
            "critical_radii": {
                "schwarzschild_radius": rs,
                "photon_sphere": photon_sphere,
                "isco": isco
            },
            "processing_info": {
                "method": "cuDF" if self.cudf_available else "pandas",
                "data_points": len(df)
            }
        }
    
    def _compute_distance(self, x, y, z):
        """Compute 3D distance with cuDF/pandas compatibility"""
        if self.cudf_available:
            return cp.sqrt(x**2 + y**2 + z**2)
        else:
            return np.sqrt(x**2 + y**2 + z**2)
    
    def _safe_sqrt(self, x):
        """Safe square root to avoid domain errors"""
        if self.cudf_available:
            return cp.sqrt(cp.maximum(x, 1e-10))
        else:
            return np.sqrt(np.maximum(x, 1e-10))
    
    def _compute_statistical_summary(self, df, columns: List[str]) -> Dict[str, Any]:
        """Compute statistical summary of key columns"""
        stats = {}
        
        for col in columns:
            if col in df.columns:
                col_data = df[col]
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()) if hasattr(col_data, 'median') else float(col_data.quantile(0.5))
                }
        
        return stats
    
    def _analyze_orbital_characteristics(self, df) -> Dict[str, Any]:
        """Analyze orbital characteristics from trajectory data"""
        analysis = {}
        
        # Orbital period estimation
        if "t" in df.columns and len(df) > 10:
            r_series = df["r"]
            # Find local minima (periapsis passages)
            if self.cudf_available:
                r_np = r_series.to_pandas().values
            else:
                r_np = r_series.values
            
            # Simple period estimation from oscillations
            mean_r = r_np.mean()
            below_mean = r_np < mean_r
            crossings = np.where(np.diff(below_mean.astype(int)))[0]
            
            if len(crossings) >= 2:
                period_estimate = 2 * (df["t"].iloc[crossings[-1]] - df["t"].iloc[crossings[0]]) / (len(crossings) - 1)
                analysis["period"] = float(period_estimate)
            else:
                analysis["period"] = None
        
        # Eccentricity estimation
        if "r" in df.columns:
            r_min = float(df["r"].min())
            r_max = float(df["r"].max())
            analysis["apoapsis"] = r_max
            analysis["periapsis"] = r_min
            analysis["eccentricity"] = (r_max - r_min) / (r_max + r_min)
        
        # Energy classification
        if "total_energy" in df.columns:
            energy = float(df["total_energy"].mean())
            analysis["average_energy"] = energy
            analysis["orbit_type"] = "elliptical" if energy < 0 else "hyperbolic"
        
        return analysis
    
    def _analyze_relativistic_effects(self, df, rs: float) -> Dict[str, Any]:
        """Analyze relativistic effects in the data"""
        analysis = {}
        
        # Maximum time dilation
        if "time_dilation" in df.columns:
            max_dilation = float(df["time_dilation"].max())
            analysis["max_time_dilation"] = max_dilation
            analysis["severe_dilation_regions"] = int((df["time_dilation"] > 2).sum())
        
        # Redshift analysis
        if "redshift" in df.columns:
            max_redshift = float(df["redshift"].max())
            min_redshift = float(df["redshift"].min())
            analysis["redshift_range"] = [min_redshift, max_redshift]
        
        # Danger zones
        if "beyond_event_horizon" in df.columns:
            safe_points = int(df["beyond_event_horizon"].sum())
            total_points = len(df)
            analysis["safety_percentage"] = (safe_points / total_points) * 100
        
        return analysis
    
    def _prepare_visualization_data(self, df, data_type: str) -> Dict[str, Any]:
        """Prepare data optimized for Plotly visualization"""
        viz_data = {}
        
        # Convert to pandas for Plotly compatibility
        pandas_df = cudf_to_pandas(df)
        
        if data_type == "physics":
            # 3D field visualization data
            if all(col in pandas_df.columns for col in ["x", "y", "z", "phi"]):
                viz_data["field_3d"] = {
                    "x": pandas_df["x"].values,
                    "y": pandas_df["y"].values,
                    "z": pandas_df["z"].values,
                    "values": pandas_df["phi"].values,
                    "type": "scatter3d"
                }
            
            # Contour data (2D slice)
            if "r" in pandas_df.columns and "phi" in pandas_df.columns:
                viz_data["contour"] = {
                    "r": pandas_df["r"].values,
                    "phi": pandas_df["phi"].values,
                    "type": "contour"
                }
        
        elif data_type == "orbital":
            # 3D trajectory
            if all(col in pandas_df.columns for col in ["x", "y", "z"]):
                viz_data["trajectory_3d"] = {
                    "x": pandas_df["x"].values,
                    "y": pandas_df["y"].values,
                    "z": pandas_df["z"].values,
                    "type": "scatter3d"
                }
            
            # Energy vs time
            if all(col in pandas_df.columns for col in ["t", "total_energy"]):
                viz_data["energy_time"] = {
                    "t": pandas_df["t"].values,
                    "energy": pandas_df["total_energy"].values,
                    "type": "scatter"
                }
        
        elif data_type == "relativistic":
            # Metric visualization
            if all(col in pandas_df.columns for col in ["r", "metric_tt"]):
                viz_data["metric"] = {
                    "r": pandas_df["r"].values,
                    "metric_tt": pandas_df["metric_tt"].values,
                    "type": "scatter"
                }
            
            # Time dilation effects
            if all(col in pandas_df.columns for col in ["r", "time_dilation"]):
                viz_data["time_dilation"] = {
                    "r": pandas_df["r"].values,
                    "dilation": pandas_df["time_dilation"].values,
                    "type": "scatter"
                }
        
        return viz_data
    
    def _process_generic_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generic data processing fallback"""
        # Convert all arrays to a single dataframe
        df_data = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    df_data[key] = val
                else:
                    # Flatten multi-dimensional arrays
                    df_data[f"{key}_flat"] = val.flatten()
        
        df = create_cudf_dataframe(df_data)
        stats = self._compute_statistical_summary(df, list(df.columns))
        
        return {
            "processed_dataframe": cudf_to_pandas(df),
            "statistics": stats,
            "processing_info": {
                "method": "cuDF" if self.cudf_available else "pandas",
                "data_points": len(df)
            }
        }
    
    def _fallback_processing(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback processing when errors occur"""
        logger.warning("⚠️ Using fallback data processing")
        
        # Simple conversion to pandas
        processed_data = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                processed_data[key] = val
        
        return {
            "processed_dataframe": pd.DataFrame(processed_data),
            "statistics": {},
            "processing_info": {
                "method": "fallback",
                "data_points": len(list(processed_data.values())[0]) if processed_data else 0
            }
        }
    
    def batch_process_simulations(self, simulation_list: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple simulations in batch for efficiency"""
        start_time = time.time()
        results = []
        
        logger.info(f"🔄 Starting batch processing of {len(simulation_list)} simulations")
        
        for i, sim_data in enumerate(simulation_list):
            try:
                result = self.process_simulation_data(
                    sim_data["data"], 
                    sim_data.get("type", "generic")
                )
                result["simulation_id"] = sim_data.get("id", i)
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Failed to process simulation {i}: {e}")
                results.append({"error": str(e), "simulation_id": i})
        
        total_time = time.time() - start_time
        logger.info(f"✅ Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        stats = self.processing_stats.copy()
        
        if stats["total_operations"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_operations"]
            stats["gpu_utilization"] = (stats["gpu_operations"] / stats["total_operations"]) * 100
        else:
            stats["average_processing_time"] = 0
            stats["gpu_utilization"] = 0
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            "available": True,
            "cudf_enabled": self.cudf_available,
            "operations_completed": self.processing_stats["total_operations"],
            "data_points_processed": self.processing_stats["data_points_processed"],
            "performance_stats": self.get_processing_statistics()
        }
    
    def clear_cache(self):
        """Clear any cached processing data"""
        # Reset statistics
        self.processing_stats = {
            "total_operations": 0,
            "gpu_operations": 0,
            "total_processing_time": 0.0,
            "data_points_processed": 0
        }
        logger.info("🧹 CuDF processor cache cleared")
