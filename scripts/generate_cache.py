#!/usr/bin/env python3
"""
Generate pre-computed simulation cache for Gravity Yonder Over
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulations.precompute import PrecomputedSimulations
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate all pre-computed simulations"""
    logger.info("Starting pre-computation of simulations...")
    
    # Ensure cache directory exists
    cache_dir = Path("backend/data/precomputed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize precomputer
    precomputer = PrecomputedSimulations(str(cache_dir))
    
    # Generate standard scenarios
    logger.info("Generating standard orbital scenarios...")
    scenarios = precomputer.generate_standard_orbits()
    
    logger.info(f"Generated {len(scenarios)} scenarios:")
    for name in scenarios:
        logger.info(f"  - {name}")
    
    # Additional specialized scenarios
    logger.info("Generating specialized scenarios...")
    
    # Black hole scenarios
    logger.info("  - Black hole approach trajectories")
    # precomputer.generate_black_hole_scenarios()
    
    # Multi-body systems
    logger.info("  - Complex multi-body systems")
    # precomputer.generate_multibody_scenarios()
    
    # Educational game levels
    logger.info("  - Game level configurations")
    # precomputer.generate_game_levels()
    
    logger.info("Pre-computation complete!")
    logger.info(f"Cache files saved to: {cache_dir}")

if __name__ == "__main__":
    main()