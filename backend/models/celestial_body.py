import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CelestialBody:
    """Represents a celestial body with physical properties"""
    name: str
    mass: float  # kg
    position: np.ndarray  # [x, y] in meters
    velocity: np.ndarray  # [vx, vy] in m/s
    radius: float = 1e6  # meters
    color: str = "blue"
    trail: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.trail is None:
            self.trail = []
        # Ensure numpy arrays
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
    
    def update_position(self, dt: float, acceleration: np.ndarray):
        """Update position and velocity using Verlet integration"""
        # Update velocity
        self.velocity += acceleration * dt
        # Update position
        self.position += self.velocity * dt
        # Store trail
        self.trail.append((self.position[0], self.position[1]))
        # Limit trail length
        if len(self.trail) > 1000:
            self.trail.pop(0)
    
    def distance_to(self, other: 'CelestialBody') -> float:
        """Calculate distance to another body"""
        return np.linalg.norm(self.position - other.position)
    
    def gravitational_force_from(self, other: 'CelestialBody', G: float = 6.67430e-11) -> np.ndarray:
        """Calculate gravitational force from another body"""
        r_vec = other.position - self.position
        r = np.linalg.norm(r_vec)
        
        if r < (self.radius + other.radius):
            # Collision
            return np.zeros(2)
        
        # F = G * m1 * m2 / r^2 * r_hat
        force_magnitude = G * self.mass * other.mass / (r ** 2)
        force_direction = r_vec / r
        
        return force_magnitude * force_direction
    
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
    
    def potential_energy_with(self, other: 'CelestialBody', G: float = 6.67430e-11) -> float:
        """Calculate gravitational potential energy with another body"""
        r = self.distance_to(other)
        if r == 0:
            return 0
        return -G * self.mass * other.mass / r
    
    def escape_velocity_from(self, other: 'CelestialBody', G: float = 6.67430e-11) -> float:
        """Calculate escape velocity from another body"""
        r = self.distance_to(other)
        if r == 0:
            return float('inf')
        return np.sqrt(2 * G * other.mass / r)
    
    def orbital_velocity_around(self, other: 'CelestialBody', G: float = 6.67430e-11) -> float:
        """Calculate circular orbital velocity around another body"""
        r = self.distance_to(other)
        if r == 0:
            return 0
        return np.sqrt(G * other.mass / r)
    
    def hill_sphere_radius(self, primary: 'CelestialBody', a: float) -> float:
        """Calculate Hill sphere radius (sphere of gravitational influence)"""
        return a * (self.mass / (3 * primary.mass)) ** (1/3)
    
    def schwarzschild_radius(self, c: float = 299792458) -> float:
        """Calculate Schwarzschild radius (for black hole threshold)"""
        G = 6.67430e-11
        return 2 * G * self.mass / (c ** 2)
    
    def copy(self) -> 'CelestialBody':
        """Create a deep copy of the celestial body"""
        return CelestialBody(
            name=self.name,
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            radius=self.radius,
            color=self.color,
            trail=self.trail.copy() if self.trail else []
        )

class CelestialSystem:
    """Manages a collection of celestial bodies"""
    
    def __init__(self, bodies: List[CelestialBody]):
        self.bodies = bodies
        self.G = 6.67430e-11
        self.time = 0.0
    
    def add_body(self, body: CelestialBody):
        """Add a body to the system"""
        self.bodies.append(body)
    
    def remove_body(self, name: str):
        """Remove a body by name"""
        self.bodies = [b for b in self.bodies if b.name != name]
    
    def get_body(self, name: str) -> Optional[CelestialBody]:
        """Get a body by name"""
        for body in self.bodies:
            if body.name == name:
                return body
        return None
    
    def center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the system"""
        total_mass = sum(body.mass for body in self.bodies)
        if total_mass == 0:
            return np.zeros(2)
        
        com = np.zeros(2)
        for body in self.bodies:
            com += body.mass * body.position
        
        return com / total_mass
    
    def total_energy(self) -> float:
        """Calculate total energy of the system"""
        kinetic = sum(body.kinetic_energy() for body in self.bodies)
        
        potential = 0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                potential += body1.potential_energy_with(body2, self.G)
        
        return kinetic + potential
    
    def angular_momentum(self) -> float:
        """Calculate total angular momentum about center of mass"""
        com = self.center_of_mass()
        L = 0
        
        for body in self.bodies:
            r = body.position - com
            L += body.mass * np.cross(r, body.velocity)
        
        return L
    
    def find_lagrange_points(self, primary: str, secondary: str) -> List[np.ndarray]:
        """Find the 5 Lagrange points for two bodies"""
        body1 = self.get_body(primary)
        body2 = self.get_body(secondary)
        
        if not body1 or not body2:
            return []
        
        # Simplified calculation for L1, L2, L3, L4, L5
        r = body2.position - body1.position
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag
        
        # Mass ratio
        mu = body2.mass / (body1.mass + body2.mass)
        
        # L1 (between bodies)
        L1_dist = r_mag * (1 - (mu / 3) ** (1/3))
        L1 = body1.position + L1_dist * r_hat
        
        # L2 (beyond secondary)
        L2_dist = r_mag * (1 + (mu / 3) ** (1/3))
        L2 = body1.position + L2_dist * r_hat
        
        # L3 (opposite side of primary)
        L3 = body1.position - r_mag * r_hat
        
        # L4 and L5 (equilateral triangles)
        angle = np.pi / 3  # 60 degrees
        rotation_matrix_L4 = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotation_matrix_L5 = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])
        
        L4 = body1.position + rotation_matrix_L4 @ r
        L5 = body1.position + rotation_matrix_L5 @ r
        
        return [L1, L2, L3, L4, L5]