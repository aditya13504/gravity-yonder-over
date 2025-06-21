/**
 * Newtonian physics calculations
 */

export const GRAVITATIONAL_CONSTANT = 6.67430e-11; // m³/kg·s²

/**
 * Calculate gravitational force between two bodies
 * F = G * (m1 * m2) / r²
 */
export function gravitationalForce(m1, m2, r) {
  if (r === 0) return 0;
  return GRAVITATIONAL_CONSTANT * m1 * m2 / (r * r);
}

/**
 * Calculate escape velocity
 * v = √(2GM/r)
 */
export function escapeVelocity(mass, radius) {
  return Math.sqrt(2 * GRAVITATIONAL_CONSTANT * mass / radius);
}

/**
 * Calculate orbital velocity
 * v = √(GM/r)
 */
export function orbitalVelocity(centralMass, radius) {
  return Math.sqrt(GRAVITATIONAL_CONSTANT * centralMass / radius);
}

/**
 * Calculate orbital period
 * T = 2π√(r³/GM)
 */
export function orbitalPeriod(radius, centralMass) {
  return 2 * Math.PI * Math.sqrt(Math.pow(radius, 3) / (GRAVITATIONAL_CONSTANT * centralMass));
}

/**
 * Calculate gravitational potential energy
 * U = -GMm/r
 */
export function gravitationalPotentialEnergy(m1, m2, r) {
  if (r === 0) return -Infinity;
  return -GRAVITATIONAL_CONSTANT * m1 * m2 / r;
}

/**
 * Calculate kinetic energy
 * KE = ½mv²
 */
export function kineticEnergy(mass, velocity) {
  return 0.5 * mass * velocity * velocity;
}

/**
 * Calculate free fall time
 * t = √(2h/g)
 */
export function freeFallTime(height, gravity = 9.81) {
  return Math.sqrt(2 * height / gravity);
}

/**
 * Calculate final velocity after free fall
 * v = √(2gh)
 */
export function freeFallVelocity(height, gravity = 9.81) {
  return Math.sqrt(2 * gravity * height);
}

/**
 * Calculate centripetal acceleration
 * a = v²/r
 */
export function centripetalAcceleration(velocity, radius) {
  return velocity * velocity / radius;
}

/**
 * Kepler's third law
 * T² = (4π²/GM) * r³
 */
export function keplerThirdLaw(radius, centralMass) {
  const constant = 4 * Math.PI * Math.PI / (GRAVITATIONAL_CONSTANT * centralMass);
  return Math.sqrt(constant * Math.pow(radius, 3));
}

/**
 * Calculate tidal force
 * F_tidal = 2GMmr/d³
 */
export function tidalForce(primaryMass, satelliteMass, satelliteRadius, distance) {
  return 2 * GRAVITATIONAL_CONSTANT * primaryMass * satelliteMass * satelliteRadius / Math.pow(distance, 3);
}

/**
 * Calculate Roche limit
 * d = 2.456 * R_primary * (ρ_primary/ρ_satellite)^(1/3)
 */
export function rocheLimit(primaryRadius, primaryDensity, satelliteDensity) {
  return 2.456 * primaryRadius * Math.pow(primaryDensity / satelliteDensity, 1/3);
}

/**
 * Calculate Hill sphere radius
 * r_H = a * (m_secondary / 3m_primary)^(1/3)
 */
export function hillSphere(semiMajorAxis, secondaryMass, primaryMass) {
  return semiMajorAxis * Math.pow(secondaryMass / (3 * primaryMass), 1/3);
}

/**
 * Calculate gravitational time dilation (simplified)
 * Δt' = Δt * √(1 - 2GM/rc²)
 */
export function gravitationalTimeDilation(mass, radius, time, c = 299792458) {
  const factor = Math.sqrt(1 - 2 * GRAVITATIONAL_CONSTANT * mass / (radius * c * c));
  return time * factor;
}

/**
 * Two-body problem solver
 */
export class TwoBodyProblem {
  constructor(m1, m2, r0, v0) {
    this.m1 = m1;
    this.m2 = m2;
    this.mu = GRAVITATIONAL_CONSTANT * (m1 + m2);
    this.r0 = r0;
    this.v0 = v0;
    
    // Calculate orbital elements
    this.calculateOrbitalElements();
  }
  
  calculateOrbitalElements() {
    // Specific orbital energy
    this.energy = this.v0 * this.v0 / 2 - this.mu / this.r0;
    
    // Semi-major axis
    this.a = -this.mu / (2 * this.energy);
    
    // Eccentricity
    const h = this.r0 * this.v0; // Specific angular momentum (simplified)
    this.e = Math.sqrt(1 + 2 * this.energy * h * h / (this.mu * this.mu));
    
    // Period
    this.period = 2 * Math.PI * Math.sqrt(Math.pow(this.a, 3) / this.mu);
    
    // Periapsis and apoapsis
    this.periapsis = this.a * (1 - this.e);
    this.apoapsis = this.a * (1 + this.e);
  }
  
  positionAtTime(t) {
    // Simplified circular orbit assumption
    const n = 2 * Math.PI / this.period; // Mean motion
    const theta = n * t;
    
    return {
      x: this.a * Math.cos(theta),
      y: this.a * Math.sin(theta)
    };
  }
}