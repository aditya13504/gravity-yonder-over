/**
 * Einstein's Relativity Physics Module
 * Comprehensive relativistic effects and calculations
 */

// Physical constants
export const CONSTANTS = {
  c: 299792458, // Speed of light (m/s)
  G: 6.67430e-11, // Gravitational constant (m³/kg⋅s²)
  h: 6.62607015e-34, // Planck constant (J⋅s)
  k: 1.380649e-23, // Boltzmann constant (J/K)
};

const c = CONSTANTS.c;
const G = CONSTANTS.G;

/**
 * Calculate Schwarzschild radius
 * r_s = 2GM/c²
 */
export function schwarzschildRadius(mass) {
  return 2 * G * mass / (c * c);
}

/**
 * Calculate gravitational time dilation
 * t' = t√(1 - r_s/r)
 */
export function timeDilation(time, mass, radius) {
  const rs = schwarzschildRadius(mass);
  if (radius <= rs) return Infinity; // Inside event horizon
  return time * Math.sqrt(1 - rs / radius);
}

/**
 * Calculate gravitational redshift
 * z = √(1 - r_s/r_e) / √(1 - r_s/r_o) - 1
 */
export function gravitationalRedshift(mass, emitterRadius, observerRadius) {
  const rs = schwarzschildRadius(mass);
  const emitterFactor = Math.sqrt(1 - rs / emitterRadius);
  const observerFactor = Math.sqrt(1 - rs / observerRadius);
  return emitterFactor / observerFactor - 1;
}

/**
 * Calculate photon sphere radius (1.5 times Schwarzschild radius)
 * @param {number} mass - Mass in kg
 * @returns {number} Photon sphere radius in meters
 */
export function photonSphere(mass) {
  return 1.5 * schwarzschildRadius(mass);
}

/**
 * Calculate special relativistic time dilation factor
 * @param {number} velocity - Velocity in m/s
 * @returns {number} Time dilation factor (gamma)
 */
export function specialTimeDilation(velocity) {
  const beta = velocity / c;
  if (beta >= 1) return Infinity;
  return 1 / Math.sqrt(1 - beta * beta);
}

/**
 * Calculate relativistic orbital velocity
 * @param {number} mass - Central mass in kg
 * @param {number} distance - Orbital distance in meters
 * @returns {number} Relativistic orbital velocity in m/s
 */
export function relativisticOrbitalVelocity(mass, distance) {
  const rs = schwarzschildRadius(mass);
  const classicalV = Math.sqrt(G * mass / distance);
  
  // Post-Newtonian correction (first order)
  const correction = 1 - rs / (2 * distance);
  return classicalV * Math.sqrt(Math.max(0, correction));
}

/**
 * Calculate orbital precession rate (perihelion precession)
 * @param {number} mass - Central mass in kg
 * @param {number} semiMajorAxis - Semi-major axis in meters
 * @param {number} eccentricity - Orbital eccentricity
 * @returns {number} Precession rate in radians per orbit
 */
export function orbitalPrecession(mass, semiMajorAxis, eccentricity) {
  const rs = schwarzschildRadius(mass);
  return (6 * Math.PI * rs) / (semiMajorAxis * (1 - eccentricity * eccentricity));
}

/**
 * Calculate escape velocity considering relativistic effects
 * @param {number} mass - Mass in kg
 * @param {number} distance - Distance from center in meters
 * @returns {number} Relativistic escape velocity in m/s
 */
export function relativisticEscapeVelocity(mass, distance) {
  const rs = schwarzschildRadius(mass);
  
  if (distance <= rs) {
    return c; // At or inside event horizon
  }
  
  // Post-Newtonian correction
  const classical = Math.sqrt(2 * G * mass / distance);
  const correction = 1 + rs / (4 * distance);
  
  return Math.min(classical * correction, 0.99 * c);
}

/**
 * Calculate tidal acceleration at distance from black hole
 * @param {number} mass - Black hole mass in kg
 * @param {number} distance - Distance from center in meters
 * @param {number} objectHeight - Height of object in meters
 * @returns {number} Tidal acceleration in m/s²
 */
export function tidalAcceleration(mass, distance, objectHeight) {
  return (2 * G * mass * objectHeight) / Math.pow(distance, 3);
}

/**
 * Check if orbit is stable in Schwarzschild metric
 * @param {number} mass - Central mass in kg
 * @param {number} distance - Orbital distance in meters
 * @returns {object} Stability information
 */
export function orbitStability(mass, distance) {
  const rs = schwarzschildRadius(mass);
  const photonR = photonSphere(mass);
  
  // Innermost stable circular orbit (ISCO)
  const isco = 3 * rs;
  
  return {
    isStable: distance > isco,
    isInsidePhotonSphere: distance < photonR,
    isInsideEventHorizon: distance < rs,
    distanceToISCO: isco,
    distanceToPhotonSphere: photonR,
    distanceToEventHorizon: rs,
    safeDistance: Math.max(isco * 2, photonR * 2)
  };
}

/**
 * Calculate Hawking radiation temperature
 * @param {number} mass - Black hole mass in kg
 * @returns {number} Hawking temperature in Kelvin
 */
export function hawkingTemperature(mass) {
  return (CONSTANTS.h * c * c * c) / 
         (8 * Math.PI * CONSTANTS.k * G * mass);
}

/**
 * Calculate gravitational wave frequency for binary system
 * @param {number} mass1 - First mass in kg
 * @param {number} mass2 - Second mass in kg
 * @param {number} separation - Orbital separation in meters
 * @returns {number} Gravitational wave frequency in Hz
 */
export function gravitationalWaveFrequency(mass1, mass2, separation) {
  const totalMass = mass1 + mass2;
  
  // Orbital frequency
  const orbitalFreq = Math.sqrt(G * totalMass / Math.pow(separation, 3)) / (2 * Math.PI);
  
  // GW frequency is twice orbital frequency
  return 2 * orbitalFreq;
}

/**
 * Calculate frame dragging effect (Lense-Thirring precession)
 * @param {number} mass - Mass in kg
 * @param {number} spin - Dimensionless spin parameter (0 to 1)
 * @param {number} distance - Distance from center in meters
 * @returns {number} Frame dragging angular velocity in rad/s
 */
export function frameDragging(mass, spin, distance) {
  const rs = schwarzschildRadius(mass);
  const a = spin * rs / 2; // Spin parameter
  
  return (2 * G * mass * a) / (c * distance * distance * distance);
}

/**
 * Calculate energy at infinity for relativistic particle
 * @param {number} mass - Central mass in kg
 * @param {number} velocity - Particle velocity in m/s
 * @param {number} distance - Distance from center in meters
 * @returns {number} Energy at infinity (normalized)
 */
export function energyAtInfinity(mass, velocity, distance) {
  const rs = schwarzschildRadius(mass);
  const gamma = specialTimeDilation(velocity);
  
  return gamma * (1 - rs / distance) - 1;
}

/**
 * Calculate wormhole traversal conditions
 * @param {number} mass - Wormhole mass in kg
 * @param {number} velocity - Traversal velocity in m/s
 * @returns {object} Traversal analysis
 */
export function wormholeTraversal(mass, velocity) {
  const rs = schwarzschildRadius(mass);
  const gamma = specialTimeDilation(velocity);
  
  return {
    isTraversable: velocity < 0.9 * c,
    timeDilation: gamma,
    properTime: 1 / gamma,
    exoticMatterRequired: mass < 0, // Negative mass needed
    throatRadius: rs / 2 // Minimum throat radius
  };
}

/**
 * Calculate relativistic rocket equation
 * @param {number} initialMass - Initial mass in kg
 * @param {number} finalMass - Final mass in kg
 * @param {number} exhaustVelocity - Exhaust velocity in m/s
 * @returns {number} Final velocity in m/s
 */
export function relativisticRocket(initialMass, finalMass, exhaustVelocity) {
  const massRatio = initialMass / finalMass;
  const beta_e = exhaustVelocity / c;
  
  const term1 = (massRatio * massRatio - 1) / (massRatio * massRatio + 1);
  const term2 = beta_e * beta_e;
  
  return c * Math.sqrt(term1 * term2);
}

// Export default object with all functions
export default {
  CONSTANTS,
  schwarzschildRadius,
  photonSphere,
  timeDilation,
  gravitationalRedshift,
  specialTimeDilation,
  relativisticOrbitalVelocity,
  orbitalPrecession,
  relativisticEscapeVelocity,
  tidalAcceleration,
  orbitStability,
  hawkingTemperature,
  gravitationalWaveFrequency,
  frameDragging,
  energyAtInfinity,
  wormholeTraversal,
  relativisticRocket
};