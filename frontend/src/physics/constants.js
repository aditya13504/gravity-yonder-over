// Physical constants used throughout the application
export const PHYSICS_CONSTANTS = {
  // Universal constants
  G: 6.67430e-11,           // Gravitational constant (m³/kg·s²)
  c: 299792458,             // Speed of light in vacuum (m/s)
  h: 6.62607015e-34,        // Planck constant (J⋅s)
  hbar: 1.054571817e-34,    // Reduced Planck constant (J⋅s)
  k_B: 1.380649e-23,        // Boltzmann constant (J/K)
  
  // Astronomical constants
  AU: 1.495978707e11,       // Astronomical unit (m)
  PARSEC: 3.0857e16,        // Parsec (m)
  LIGHT_YEAR: 9.4607e15,    // Light year (m)
  
  // Solar system bodies
  SUN: {
    mass: 1.98847e30,       // kg
    radius: 6.96e8,         // m
    luminosity: 3.828e26    // W
  },
  
  EARTH: {
    mass: 5.9722e24,        // kg
    radius: 6.3781e6,       // m
    orbitalRadius: 1.496e11, // m (1 AU)
    rotationPeriod: 86400,   // s (1 day)
    orbitalPeriod: 31557600  // s (1 year)
  },
  
  MOON: {
    mass: 7.342e22,         // kg
    radius: 1.737e6,        // m
    orbitalRadius: 3.844e8, // m
    orbitalPeriod: 2.36e6   // s (~27.3 days)
  },
  
  JUPITER: {
    mass: 1.8982e27,        // kg
    radius: 6.9911e7,       // m
    orbitalRadius: 7.785e11, // m
    orbitalPeriod: 3.74e8   // s (~11.86 years)
  },

  // Common black holes (approximate)
  BLACK_HOLES: {
    STELLAR: {
      mass: 3 * 1.98847e30,  // 3 solar masses
      name: "Typical Stellar Black Hole"
    },
    SAGITTARIUS_A_STAR: {
      mass: 4.154e36,        // ~4.15 million solar masses
      name: "Sagittarius A*"
    }
  },

  // Conversion factors
  CONVERSIONS: {
    METERS_TO_KM: 1e-3,
    KM_TO_METERS: 1e3,
    SECONDS_TO_DAYS: 1/86400,
    DAYS_TO_SECONDS: 86400,
    YEARS_TO_SECONDS: 31557600,
    SECONDS_TO_YEARS: 1/31557600,
    KG_TO_SOLAR_MASSES: 1/1.98847e30,
    SOLAR_MASSES_TO_KG: 1.98847e30
  }
};

// Derived constants calculated from base constants
export const DERIVED_CONSTANTS = {
  // Schwarzschild radius of Sun
  SUN_SCHWARZSCHILD_RADIUS: 2 * PHYSICS_CONSTANTS.G * PHYSICS_CONSTANTS.SUN.mass / (PHYSICS_CONSTANTS.c ** 2),
  
  // Earth's escape velocity
  EARTH_ESCAPE_VELOCITY: Math.sqrt(2 * PHYSICS_CONSTANTS.G * PHYSICS_CONSTANTS.EARTH.mass / PHYSICS_CONSTANTS.EARTH.radius),
  
  // Moon's escape velocity
  MOON_ESCAPE_VELOCITY: Math.sqrt(2 * PHYSICS_CONSTANTS.G * PHYSICS_CONSTANTS.MOON.mass / PHYSICS_CONSTANTS.MOON.radius),
  
  // Earth's surface gravity
  EARTH_SURFACE_GRAVITY: PHYSICS_CONSTANTS.G * PHYSICS_CONSTANTS.EARTH.mass / (PHYSICS_CONSTANTS.EARTH.radius ** 2),
  
  // Moon's surface gravity
  MOON_SURFACE_GRAVITY: PHYSICS_CONSTANTS.G * PHYSICS_CONSTANTS.MOON.mass / (PHYSICS_CONSTANTS.MOON.radius ** 2)
};

// Unit conversion utilities
export const convertUnits = {
  metersToKm: (m) => m * PHYSICS_CONSTANTS.CONVERSIONS.METERS_TO_KM,
  kmToMeters: (km) => km * PHYSICS_CONSTANTS.CONVERSIONS.KM_TO_METERS,
  secondsToDays: (s) => s * PHYSICS_CONSTANTS.CONVERSIONS.SECONDS_TO_DAYS,
  daysToSeconds: (d) => d * PHYSICS_CONSTANTS.CONVERSIONS.DAYS_TO_SECONDS,
  secondsToYears: (s) => s * PHYSICS_CONSTANTS.CONVERSIONS.SECONDS_TO_YEARS,
  yearsToSeconds: (y) => y * PHYSICS_CONSTANTS.CONVERSIONS.YEARS_TO_SECONDS,
  kgToSolarMasses: (kg) => kg * PHYSICS_CONSTANTS.CONVERSIONS.KG_TO_SOLAR_MASSES,
  solarMassesToKg: (sm) => sm * PHYSICS_CONSTANTS.CONVERSIONS.SOLAR_MASSES_TO_KG,
  
  // Angular conversions
  radiansTodegrees: (rad) => rad * 180 / Math.PI,
  degreesToRadians: (deg) => deg * Math.PI / 180,
  
  // Velocity conversions
  msToKmh: (ms) => ms * 3.6,
  kmhToMs: (kmh) => kmh / 3.6,
  
  // Energy conversions
  joulesToElectronVolts: (j) => j / 1.602176634e-19,
  electronVoltsToJoules: (ev) => ev * 1.602176634e-19
};

// Useful physical scales for UI display
export const DISPLAY_SCALES = {
  DISTANCE: {
    ATOMIC: { min: 1e-15, max: 1e-9, unit: 'pm', name: 'Atomic' },
    MICROSCOPIC: { min: 1e-9, max: 1e-3, unit: 'nm', name: 'Microscopic' },
    HUMAN: { min: 1e-3, max: 1e6, unit: 'm', name: 'Human Scale' },
    PLANETARY: { min: 1e6, max: 1e12, unit: 'km', name: 'Planetary' },
    STELLAR: { min: 1e12, max: 1e18, unit: 'AU', name: 'Stellar' },
    GALACTIC: { min: 1e18, max: 1e24, unit: 'ly', name: 'Galactic' }
  },
  
  MASS: {
    PARTICLE: { min: 1e-30, max: 1e-20, unit: 'u', name: 'Particle' },
    MOLECULAR: { min: 1e-20, max: 1e-10, unit: 'u', name: 'Molecular' },
    HUMAN: { min: 1e-10, max: 1e10, unit: 'kg', name: 'Human Scale' },
    PLANETARY: { min: 1e10, max: 1e30, unit: 'Earth masses', name: 'Planetary' },
    STELLAR: { min: 1e30, max: 1e40, unit: 'Solar masses', name: 'Stellar' }
  },
  
  TIME: {
    QUANTUM: { min: 1e-24, max: 1e-12, unit: 'fs', name: 'Quantum' },
    MOLECULAR: { min: 1e-12, max: 1e-6, unit: 'ps', name: 'Molecular' },
    HUMAN: { min: 1e-6, max: 1e8, unit: 's', name: 'Human Scale' },
    GEOLOGICAL: { min: 1e8, max: 1e15, unit: 'years', name: 'Geological' },
    COSMOLOGICAL: { min: 1e15, max: 1e18, unit: 'Gyr', name: 'Cosmological' }
  }
};
