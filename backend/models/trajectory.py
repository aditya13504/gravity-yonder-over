import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import json

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: Optional[np.ndarray] = None
    
class Trajectory:
    """Represents the trajectory of a celestial body"""
    
    def __init__(self, body_name: str):
        self.body_name = body_name
        self.points: List[TrajectoryPoint] = []
        self._position_interpolator = None
        self._velocity_interpolator = None
        
    def add_point(self, time: float, position: np.ndarray, 
                  velocity: np.ndarray, acceleration: Optional[np.ndarray] = None):
        """Add a point to the trajectory"""
        self.points.append(TrajectoryPoint(time, position.copy(), velocity.copy(), 
                                         acceleration.copy() if acceleration is not None else None))
        self._invalidate_interpolators()
        
    def _invalidate_interpolators(self):
        """Invalidate cached interpolators when data changes"""
        self._position_interpolator = None
        self._velocity_interpolator = None
        
    def _build_interpolators(self):
        """Build cubic spline interpolators for smooth trajectory"""
        if len(self.points) < 2:
            return
            
        times = [p.time for p in self.points]
        positions = np.array([p.position for p in self.points])
        velocities = np.array([p.velocity for p in self.points])
        
        # Create separate interpolators for each dimension
        self._position_interpolator = [
            CubicSpline(times, positions[:, i]) 
            for i in range(positions.shape[1])
        ]
        self._velocity_interpolator = [
            CubicSpline(times, velocities[:, i]) 
            for i in range(velocities.shape[1])
        ]
        
    def get_position(self, time: float) -> np.ndarray:
        """Get interpolated position at given time"""
        if len(self.points) == 0:
            raise ValueError("No trajectory points available")
            
        if len(self.points) == 1:
            return self.points[0].position.copy()
            
        if self._position_interpolator is None:
            self._build_interpolators()
            
        # Extrapolate if outside time range
        if time <= self.points[0].time:
            return self.points[0].position.copy()
        if time >= self.points[-1].time:
            return self.points[-1].position.copy()
            
        # Interpolate
        return np.array([interp(time) for interp in self._position_interpolator])
    
    def get_velocity(self, time: float) -> np.ndarray:
        """Get interpolated velocity at given time"""
        if len(self.points) == 0:
            raise ValueError("No trajectory points available")
            
        if len(self.points) == 1:
            return self.points[0].velocity.copy()
            
        if self._velocity_interpolator is None:
            self._build_interpolators()
            
        # Extrapolate if outside time range
        if time <= self.points[0].time:
            return self.points[0].velocity.copy()
        if time >= self.points[-1].time:
            return self.points[-1].velocity.copy()
            
        # Interpolate
        return np.array([interp(time) for interp in self._velocity_interpolator])
    
    def get_state(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get both position and velocity at given time"""
        return self.get_position(time), self.get_velocity(time)
    
    def calculate_orbital_elements(self, central_mass: float, G: float = 6.67430e-11) -> Dict:
        """Calculate orbital elements from trajectory"""
        if len(self.points) == 0:
            return {}
            
        # Use middle point for best accuracy
        mid_point = self.points[len(self.points) // 2]
        r = mid_point.position
        v = mid_point.velocity
        
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Specific orbital energy
        energy = v_mag**2 / 2 - G * central_mass / r_mag
        
        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        mu = G * central_mass
        e_vec = np.cross(v, h) / mu - r / r_mag
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        if abs(energy) > 1e-10:  # Not parabolic
            a = -mu / (2 * energy)
        else:
            a = float('inf')
            
        # Inclination
        i = np.arccos(h[2] / h_mag) if h_mag > 0 else 0
        
        # Calculate periapsis and apoapsis
        if e < 1:  # Elliptical orbit
            periapsis = a * (1 - e)
            apoapsis = a * (1 + e)
            period = 2 * np.pi * np.sqrt(a**3 / mu) if a > 0 else float('inf')
        else:
            periapsis = h_mag**2 / (mu * (1 + e))
            apoapsis = float('inf')
            period = float('inf')
            
        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'inclination': np.degrees(i),
            'periapsis': periapsis,
            'apoapsis': apoapsis,
            'period': period,
            'specific_energy': energy,
            'specific_angular_momentum': h_mag
        }
    
    def find_closest_approach(self, other_trajectory: 'Trajectory') -> Tuple[float, float]:
        """Find time and distance of closest approach to another trajectory"""
        min_distance = float('inf')
        closest_time = 0
        
        # Sample trajectories
        times = [p.time for p in self.points]
        if len(times) == 0:
            return 0, float('inf')
            
        for t in times:
            try:
                pos1 = self.get_position(t)
                pos2 = other_trajectory.get_position(t)
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_time = t
            except:
                continue
                
        return closest_time, min_distance
    
    def to_json(self) -> str:
        """Serialize trajectory to JSON"""
        data = {
            'body_name': self.body_name,
            'points': [
                {
                    'time': p.time,
                    'position': p.position.tolist(),
                    'velocity': p.velocity.tolist(),
                    'acceleration': p.acceleration.tolist() if p.acceleration is not None else None
                }
                for p in self.points
            ]
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Trajectory':
        """Deserialize trajectory from JSON"""
        data = json.loads(json_str)
        trajectory = cls(data['body_name'])
        
        for point_data in data['points']:
            trajectory.add_point(
                point_data['time'],
                np.array(point_data['position']),
                np.array(point_data['velocity']),
                np.array(point_data['acceleration']) if point_data['acceleration'] else None
            )
            
        return trajectory
    
    def simplify(self, tolerance: float = 0.01) -> 'Trajectory':
        """Create simplified trajectory with fewer points using Douglas-Peucker algorithm"""
        if len(self.points) <= 2:
            return self
            
        # Implement simplified version for trajectory
        # This is a basic implementation - could be improved
        simplified = Trajectory(self.body_name)
        simplified.add_point(self.points[0].time, self.points[0].position, 
                           self.points[0].velocity, self.points[0].acceleration)
        
        # Keep points where trajectory changes significantly
        for i in range(1, len(self.points) - 1):
            prev_vel = self.points[i-1].velocity
            curr_vel = self.points[i].velocity
            next_vel = self.points[i+1].velocity
            
            # Check if velocity direction changes significantly
            dot1 = np.dot(prev_vel, curr_vel) / (np.linalg.norm(prev_vel) * np.linalg.norm(curr_vel))
            dot2 = np.dot(curr_vel, next_vel) / (np.linalg.norm(curr_vel) * np.linalg.norm(next_vel))
            
            if abs(dot1 - dot2) > tolerance:
                simplified.add_point(self.points[i].time, self.points[i].position,
                                   self.points[i].velocity, self.points[i].acceleration)
                
        simplified.add_point(self.points[-1].time, self.points[-1].position,
                           self.points[-1].velocity, self.points[-1].acceleration)
        
        return simplified