"""
CUDA acceleration utilities for high-performance physics computations
Provides optimized GPU kernels for gravitational calculations
"""

import numpy as np
import cupy as cp
import numba
from numba import cuda, float32, float64
import math
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# CUDA kernel for gravitational force computation
@cuda.jit
def gravity_force_kernel(positions, masses, forces, n_bodies, G):
    """
    CUDA kernel to compute gravitational forces between all bodies
    
    Args:
        positions: Array of body positions [n_bodies, 3]
        masses: Array of body masses [n_bodies]
        forces: Output array for forces [n_bodies, 3]
        n_bodies: Number of bodies
        G: Gravitational constant
    """
    i = cuda.grid(1)
    
    if i < n_bodies:
        fx, fy, fz = 0.0, 0.0, 0.0
        
        for j in range(n_bodies):
            if i != j:
                # Calculate distance vector
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                # Calculate distance magnitude
                r_sq = dx*dx + dy*dy + dz*dz + 1e-10  # Softening parameter
                r_mag = math.sqrt(r_sq)
                
                # Calculate force magnitude
                force_mag = G * masses[i] * masses[j] / r_sq
                
                # Add force components
                fx += force_mag * dx / r_mag
                fy += force_mag * dy / r_mag
                fz += force_mag * dz / r_mag
        
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz

@cuda.jit
def gravity_field_kernel(grid_x, grid_y, grid_z, positions, masses, field, n_bodies, n_grid, G):
    """
    CUDA kernel to compute gravitational field on a 3D grid
    
    Args:
        grid_x, grid_y, grid_z: Grid coordinate arrays
        positions: Body positions [n_bodies, 3]
        masses: Body masses [n_bodies]
        field: Output field array [n_grid, n_grid, n_grid]
        n_bodies: Number of bodies
        n_grid: Grid size
        G: Gravitational constant
    """
    i, j, k = cuda.grid(3)
    
    if i < n_grid and j < n_grid and k < n_grid:
        potential = 0.0
        
        for body in range(n_bodies):
            # Distance from grid point to body
            dx = grid_x[i] - positions[body, 0]
            dy = grid_y[j] - positions[body, 1]
            dz = grid_z[k] - positions[body, 2]
            
            r = math.sqrt(dx*dx + dy*dy + dz*dz + 1e-10)
            potential += -G * masses[body] / r
        
        field[i, j, k] = potential

@cuda.jit
def trajectory_integration_kernel(positions, velocities, masses, dt, n_bodies, n_steps, G):
    """
    CUDA kernel for trajectory integration using leapfrog method
    
    Args:
        positions: Position array [n_steps, n_bodies, 3]
        velocities: Velocity array [n_steps, n_bodies, 3]
        masses: Body masses [n_bodies]
        dt: Time step
        n_bodies: Number of bodies
        n_steps: Number of integration steps
        G: Gravitational constant
    """
    body_idx = cuda.grid(1)
    
    if body_idx < n_bodies:
        # Initialize positions and velocities
        x, y, z = positions[0, body_idx, 0], positions[0, body_idx, 1], positions[0, body_idx, 2]
        vx, vy, vz = velocities[0, body_idx, 0], velocities[0, body_idx, 1], velocities[0, body_idx, 2]
        
        for step in range(1, n_steps):
            # Compute forces
            fx, fy, fz = 0.0, 0.0, 0.0
            
            for other in range(n_bodies):
                if body_idx != other:
                    dx = positions[step-1, other, 0] - x
                    dy = positions[step-1, other, 1] - y
                    dz = positions[step-1, other, 2] - z
                    
                    r_sq = dx*dx + dy*dy + dz*dz + 1e-10
                    r_mag = math.sqrt(r_sq)
                    
                    force_mag = G * masses[body_idx] * masses[other] / r_sq
                    
                    fx += force_mag * dx / r_mag
                    fy += force_mag * dy / r_mag
                    fz += force_mag * dz / r_mag
            
            # Update velocity (leapfrog)
            ax = fx / masses[body_idx]
            ay = fy / masses[body_idx]
            az = fz / masses[body_idx]
            
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            z += vz * dt
            
            # Store results
            positions[step, body_idx, 0] = x
            positions[step, body_idx, 1] = y
            positions[step, body_idx, 2] = z
            velocities[step, body_idx, 0] = vx
            velocities[step, body_idx, 1] = vy
            velocities[step, body_idx, 2] = vz

class CUDAGravityAccelerator:
    """
    CUDA-accelerated gravity simulation engine
    """
    
    def __init__(self):
        self.device_available = self._check_cuda_availability()
        self.G = 6.67430e-11  # Gravitational constant
        
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available"""
        try:
            cuda.detect()
            logger.info("CUDA detected and available")
            return True
        except Exception as e:
            logger.warning(f"CUDA not available: {e}")
            return False
    
    def compute_forces_gpu(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Compute gravitational forces using GPU acceleration
        
        Args:
            positions: Body positions [n_bodies, 3]
            masses: Body masses [n_bodies]
            
        Returns:
            Forces array [n_bodies, 3]
        """
        if not self.device_available:
            return self._compute_forces_cpu(positions, masses)
        
        n_bodies = len(positions)
        
        # Allocate GPU memory
        d_positions = cuda.to_device(positions.astype(np.float64))
        d_masses = cuda.to_device(masses.astype(np.float64))
        d_forces = cuda.device_array((n_bodies, 3), dtype=np.float64)
        
        # Configure grid and block dimensions
        threads_per_block = 256
        blocks_per_grid = (n_bodies + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        gravity_force_kernel[blocks_per_grid, threads_per_block](
            d_positions, d_masses, d_forces, n_bodies, self.G
        )
        
        # Copy result back to host
        forces = d_forces.copy_to_host()
        
        return forces
    
    def compute_gravity_field_gpu(self, positions: np.ndarray, masses: np.ndarray, 
                                 grid_bounds: Tuple[float, float], grid_size: int) -> np.ndarray:
        """
        Compute gravitational field on 3D grid using GPU
        
        Args:
            positions: Body positions [n_bodies, 3]
            masses: Body masses [n_bodies]
            grid_bounds: (min_coord, max_coord) for grid
            grid_size: Number of grid points per dimension
            
        Returns:
            Gravitational potential field [grid_size, grid_size, grid_size]
        """
        if not self.device_available:
            return self._compute_gravity_field_cpu(positions, masses, grid_bounds, grid_size)
        
        # Create coordinate grids
        coords = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
        grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
        
        n_bodies = len(positions)
        
        # Allocate GPU memory
        d_grid_x = cuda.to_device(grid_x.astype(np.float64))
        d_grid_y = cuda.to_device(grid_y.astype(np.float64))
        d_grid_z = cuda.to_device(grid_z.astype(np.float64))
        d_positions = cuda.to_device(positions.astype(np.float64))
        d_masses = cuda.to_device(masses.astype(np.float64))
        d_field = cuda.device_array((grid_size, grid_size, grid_size), dtype=np.float64)
        
        # Configure 3D grid and block dimensions
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (grid_size + threads_per_block[0] - 1) // threads_per_block[0],
            (grid_size + threads_per_block[1] - 1) // threads_per_block[1],
            (grid_size + threads_per_block[2] - 1) // threads_per_block[2]
        )
        
        # Launch kernel
        gravity_field_kernel[blocks_per_grid, threads_per_block](
            d_grid_x, d_grid_y, d_grid_z, d_positions, d_masses, 
            d_field, n_bodies, grid_size, self.G
        )
        
        # Copy result back to host
        field = d_field.copy_to_host()
        
        return field
    
    def integrate_trajectories_gpu(self, initial_positions: np.ndarray, 
                                  initial_velocities: np.ndarray, masses: np.ndarray,
                                  dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate trajectories using GPU acceleration
        
        Args:
            initial_positions: Initial positions [n_bodies, 3]
            initial_velocities: Initial velocities [n_bodies, 3]
            masses: Body masses [n_bodies]
            dt: Time step
            n_steps: Number of integration steps
            
        Returns:
            Tuple of (positions, velocities) arrays [n_steps, n_bodies, 3]
        """
        if not self.device_available:
            return self._integrate_trajectories_cpu(initial_positions, initial_velocities, masses, dt, n_steps)
        
        n_bodies = len(initial_positions)
        
        # Allocate arrays for trajectory data
        positions = np.zeros((n_steps, n_bodies, 3), dtype=np.float64)
        velocities = np.zeros((n_steps, n_bodies, 3), dtype=np.float64)
        
        # Set initial conditions
        positions[0] = initial_positions
        velocities[0] = initial_velocities
        
        # Allocate GPU memory
        d_positions = cuda.to_device(positions)
        d_velocities = cuda.to_device(velocities)
        d_masses = cuda.to_device(masses.astype(np.float64))
        
        # Configure grid and block dimensions
        threads_per_block = 256
        blocks_per_grid = (n_bodies + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        trajectory_integration_kernel[blocks_per_grid, threads_per_block](
            d_positions, d_velocities, d_masses, dt, n_bodies, n_steps, self.G
        )
        
        # Copy results back to host
        positions = d_positions.copy_to_host()
        velocities = d_velocities.copy_to_host()
        
        return positions, velocities
    
    def _compute_forces_cpu(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """CPU fallback for force computation"""
        n_bodies = len(positions)
        forces = np.zeros_like(positions)
        
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec) + 1e-10
                    force_mag = self.G * masses[i] * masses[j] / (r_mag ** 2)
                    forces[i] += force_mag * r_vec / r_mag
        
        return forces
    
    def _compute_gravity_field_cpu(self, positions: np.ndarray, masses: np.ndarray,
                                  grid_bounds: Tuple[float, float], grid_size: int) -> np.ndarray:
        """CPU fallback for gravity field computation"""
        coords = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
        field = np.zeros((grid_size, grid_size, grid_size))
        
        for i, x in enumerate(coords):
            for j, y in enumerate(coords):
                for k, z in enumerate(coords):
                    potential = 0.0
                    for pos, mass in zip(positions, masses):
                        r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2 + 1e-10)
                        potential += -self.G * mass / r
                    field[i, j, k] = potential
        
        return field
    
    def _integrate_trajectories_cpu(self, initial_positions: np.ndarray, 
                                   initial_velocities: np.ndarray, masses: np.ndarray,
                                   dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for trajectory integration"""
        n_bodies = len(initial_positions)
        positions = np.zeros((n_steps, n_bodies, 3))
        velocities = np.zeros((n_steps, n_bodies, 3))
        
        positions[0] = initial_positions
        velocities[0] = initial_velocities
        
        for step in range(1, n_steps):
            forces = self._compute_forces_cpu(positions[step-1], masses)
            accelerations = forces / masses[:, np.newaxis]
            
            velocities[step] = velocities[step-1] + accelerations * dt
            positions[step] = positions[step-1] + velocities[step] * dt
        
        return positions, velocities

# Example usage
if __name__ == "__main__":
    # Test CUDA acceleration
    accelerator = CUDAGravityAccelerator()
    
    # Test data: Earth-Moon system
    positions = np.array([[0.0, 0.0, 0.0], [384400000.0, 0.0, 0.0]])
    masses = np.array([5.972e24, 7.35e22])
    
    # Test force computation
    forces = accelerator.compute_forces_gpu(positions, masses)
    print(f"Computed forces: {forces}")
    
    # Test gravity field
    field = accelerator.compute_gravity_field_gpu(positions, masses, (-1e9, 1e9), 64)
    print(f"Gravity field shape: {field.shape}")
    
    # Test trajectory integration
    initial_velocities = np.array([[0.0, 0.0, 0.0], [0.0, 1022.0, 0.0]])
    traj_pos, traj_vel = accelerator.integrate_trajectories_gpu(
        positions, initial_velocities, masses, 3600.0, 100
    )
    print(f"Trajectory computed: positions {traj_pos.shape}, velocities {traj_vel.shape}")
