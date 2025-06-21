import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import scipy.special as special

@dataclass
class GravityField:
    """Represents a gravitational field in space"""
    
    def __init__(self, bodies: List['CelestialBody']):
        self.bodies = bodies
        self.G = 6.67430e-11
        
    def potential_at_point(self, position: np.ndarray) -> float:
        """Calculate gravitational potential at a point"""
        potential = 0.0
        
        for body in self.bodies:
            r = np.linalg.norm(position - body.position)
            if r > 0:
                potential -= self.G * body.mass / r
                
        return potential
    
    def field_strength_at_point(self, position: np.ndarray) -> np.ndarray:
        """Calculate gravitational field strength (acceleration) at a point"""
        field = np.zeros(3 if len(position) == 3 else 2)
        
        for body in self.bodies:
            r_vec = position - body.position
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                field -= self.G * body.mass * r_vec / r**3
                
        return field
    
    def gradient_at_point(self, position: np.ndarray) -> np.ndarray:
        """Calculate gradient of gravitational field (tidal tensor)"""
        if len(position) == 2:
            gradient = np.zeros((2, 2))
        else:
            gradient = np.zeros((3, 3))
            
        for body in self.bodies:
            r_vec = position - body.position
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                # Calculate second derivatives of potential
                for i in range(len(position)):
                    for j in range(len(position)):
                        if i == j:
                            gradient[i, j] += self.G * body.mass * (3 * r_vec[i]**2 / r**5 - 1 / r**3)
                        else:
                            gradient[i, j] += self.G * body.mass * 3 * r_vec[i] * r_vec[j] / r**5
                            
        return gradient
    
    def create_field_map(self, bounds: Tuple[float, float, float, float], 
                        resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a 2D map of the gravitational field"""
        x_min, x_max, y_min, y_max = bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate potential at each point
        potential = np.zeros_like(X)
        field_x = np.zeros_like(X)
        field_y = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                potential[i, j] = self.potential_at_point(pos)
                field = self.field_strength_at_point(pos)
                field_x[i, j] = field[0]
                field_y[i, j] = field[1]
                
        return potential, field_x, field_y
    
    def find_equilibrium_points(self, bounds: Tuple[float, float, float, float],
                               tolerance: float = 1e-6) -> List[np.ndarray]:
        """Find gravitational equilibrium points (like Lagrange points)"""
        equilibrium_points = []
        
        # Use gradient descent/ascent to find points where field is zero
        # This is a simplified implementation
        x_min, x_max, y_min, y_max = bounds
        
        # Start from a grid of initial guesses
        for x0 in np.linspace(x_min, x_max, 10):
            for y0 in np.linspace(y_min, y_max, 10):
                pos = np.array([x0, y0])
                
                # Newton-Raphson iteration
                for _ in range(100):
                    field = self.field_strength_at_point(pos)
                    if np.linalg.norm(field) < tolerance:
                        # Check if this is a new equilibrium point
                        is_new = True
                        for ep in equilibrium_points:
                            if np.linalg.norm(pos - ep) < 0.1:
                                is_new = False
                                break
                        if is_new:
                            equilibrium_points.append(pos.copy())
                        break
                        
                    # Update position using gradient
                    gradient = self.gradient_at_point(pos)
                    try:
                        delta = -np.linalg.solve(gradient, field)
                        pos += delta * 0.1  # Small step
                    except:
                        break  # Singular matrix, skip
                        
        return equilibrium_points
    
    def effective_potential(self, position: np.ndarray, angular_momentum: float) -> float:
        """Calculate effective potential including centrifugal term"""
        r = np.linalg.norm(position)
        gravitational = self.potential_at_point(position)
        centrifugal = angular_momentum**2 / (2 * r**2) if r > 0 else 0
        return gravitational + centrifugal
    
    def hill_sphere_radius(self, body_index: int, primary_index: int) -> float:
        """Calculate Hill sphere radius for a body orbiting a primary"""
        if body_index == primary_index:
            return float('inf')
            
        body = self.bodies[body_index]
        primary = self.bodies[primary_index]
        
        r = np.linalg.norm(body.position - primary.position)
        return r * (body.mass / (3 * primary.mass)) ** (1/3)
    
    def roche_limit(self, primary_index: int, satellite_density: float,
                   fluid: bool = True) -> float:
        """Calculate Roche limit for a given primary and satellite density"""
        primary = self.bodies[primary_index]
        primary_density = primary.mass / (4/3 * np.pi * primary.radius**3)
        
        if fluid:
            return 2.456 * primary.radius * (primary_density / satellite_density) ** (1/3)
        else:
            return 2.455 * primary.radius * (primary_density / satellite_density) ** (1/3)