// General utility functions for the gravity application

/**
 * Mathematical utility functions
 */
export const math = {
  /**
   * Clamp a value between min and max
   * @param {number} value - Value to clamp
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @returns {number} Clamped value
   */
  clamp: (value, min, max) => Math.min(Math.max(value, min), max),

  /**
   * Linear interpolation between two values
   * @param {number} a - Start value
   * @param {number} b - End value
   * @param {number} t - Interpolation factor (0-1)
   * @returns {number} Interpolated value
   */
  lerp: (a, b, t) => a + (b - a) * t,

  /**
   * Map a value from one range to another
   * @param {number} value - Input value
   * @param {number} inMin - Input range minimum
   * @param {number} inMax - Input range maximum
   * @param {number} outMin - Output range minimum
   * @param {number} outMax - Output range maximum
   * @returns {number} Mapped value
   */
  map: (value, inMin, inMax, outMin, outMax) => {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
  },

  /**
   * Calculate distance between two points
   * @param {Object} p1 - Point 1 {x, y}
   * @param {Object} p2 - Point 2 {x, y}
   * @returns {number} Distance
   */
  distance: (p1, p2) => Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2),

  /**
   * Calculate 3D distance between two points
   * @param {Object} p1 - Point 1 {x, y, z}
   * @param {Object} p2 - Point 2 {x, y, z}
   * @returns {number} Distance
   */
  distance3D: (p1, p2) => Math.sqrt(
    (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2
  ),

  /**
   * Normalize a vector
   * @param {Object} vector - Vector {x, y}
   * @returns {Object} Normalized vector
   */
  normalize: (vector) => {
    const magnitude = Math.sqrt(vector.x ** 2 + vector.y ** 2);
    if (magnitude === 0) return { x: 0, y: 0 };
    return { x: vector.x / magnitude, y: vector.y / magnitude };
  },

  /**
   * Calculate angle between two vectors
   * @param {Object} v1 - Vector 1 {x, y}
   * @param {Object} v2 - Vector 2 {x, y}
   * @returns {number} Angle in radians
   */
  angleBetween: (v1, v2) => {
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x ** 2 + v1.y ** 2);
    const mag2 = Math.sqrt(v2.x ** 2 + v2.y ** 2);
    return Math.acos(dot / (mag1 * mag2));
  },

  /**
   * Generate random number between min and max
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @returns {number} Random number
   */
  random: (min, max) => Math.random() * (max - min) + min,

  /**
   * Convert degrees to radians
   * @param {number} degrees - Angle in degrees
   * @returns {number} Angle in radians
   */
  toRadians: (degrees) => degrees * Math.PI / 180,

  /**
   * Convert radians to degrees
   * @param {number} radians - Angle in radians
   * @returns {number} Angle in degrees
   */
  toDegrees: (radians) => radians * 180 / Math.PI
};

/**
 * Formatting utilities
 */
export const format = {
  /**
   * Format a number with appropriate scientific notation
   * @param {number} value - Number to format
   * @param {number} precision - Decimal places
   * @returns {string} Formatted number
   */
  scientific: (value, precision = 2) => {
    if (value === 0) return '0';
    const exponent = Math.floor(Math.log10(Math.abs(value)));
    if (exponent >= -2 && exponent <= 4) {
      return value.toFixed(precision);
    }
    const mantissa = value / Math.pow(10, exponent);
    return `${mantissa.toFixed(precision)}×10${exponent >= 0 ? '⁺' : '⁻'}${Math.abs(exponent)}`;
  },

  /**
   * Format distance with appropriate units
   * @param {number} meters - Distance in meters
   * @returns {string} Formatted distance
   */
  distance: (meters) => {
    const abs = Math.abs(meters);
    if (abs < 1e-9) return `${(meters * 1e12).toFixed(2)} pm`;
    if (abs < 1e-6) return `${(meters * 1e9).toFixed(2)} nm`;
    if (abs < 1e-3) return `${(meters * 1e6).toFixed(2)} μm`;
    if (abs < 1) return `${(meters * 1e3).toFixed(2)} mm`;
    if (abs < 1e3) return `${meters.toFixed(2)} m`;
    if (abs < 1e6) return `${(meters / 1e3).toFixed(2)} km`;
    if (abs < 1.496e11) return `${(meters / 1e6).toFixed(2)} Mm`;
    if (abs < 9.461e15) return `${(meters / 1.496e11).toFixed(2)} AU`;
    return `${(meters / 9.461e15).toFixed(2)} ly`;
  },

  /**
   * Format mass with appropriate units
   * @param {number} kg - Mass in kilograms
   * @returns {string} Formatted mass
   */
  mass: (kg) => {
    const abs = Math.abs(kg);
    if (abs < 1e-15) return `${(kg * 1e18).toFixed(2)} ag`;
    if (abs < 1e-12) return `${(kg * 1e15).toFixed(2)} fg`;
    if (abs < 1e-9) return `${(kg * 1e12).toFixed(2)} pg`;
    if (abs < 1e-6) return `${(kg * 1e9).toFixed(2)} ng`;
    if (abs < 1e-3) return `${(kg * 1e6).toFixed(2)} mg`;
    if (abs < 1) return `${(kg * 1e3).toFixed(2)} g`;
    if (abs < 1e3) return `${kg.toFixed(2)} kg`;
    if (abs < 1.989e30) return `${(kg / 1e3).toFixed(2)} t`;
    return `${(kg / 1.989e30).toFixed(2)} M☉`;
  },

  /**
   * Format time with appropriate units
   * @param {number} seconds - Time in seconds
   * @returns {string} Formatted time
   */
  time: (seconds) => {
    const abs = Math.abs(seconds);
    if (abs < 1e-12) return `${(seconds * 1e15).toFixed(2)} fs`;
    if (abs < 1e-9) return `${(seconds * 1e12).toFixed(2)} ps`;
    if (abs < 1e-6) return `${(seconds * 1e9).toFixed(2)} ns`;
    if (abs < 1e-3) return `${(seconds * 1e6).toFixed(2)} μs`;
    if (abs < 1) return `${(seconds * 1e3).toFixed(2)} ms`;
    if (abs < 60) return `${seconds.toFixed(2)} s`;
    if (abs < 3600) return `${(seconds / 60).toFixed(2)} min`;
    if (abs < 86400) return `${(seconds / 3600).toFixed(2)} h`;
    if (abs < 31557600) return `${(seconds / 86400).toFixed(2)} days`;
    return `${(seconds / 31557600).toFixed(2)} years`;
  },

  /**
   * Format velocity with appropriate units
   * @param {number} ms - Velocity in m/s
   * @returns {string} Formatted velocity
   */
  velocity: (ms) => {
    const abs = Math.abs(ms);
    const c = 299792458; // Speed of light
    if (abs < 1) return `${(ms * 1000).toFixed(2)} mm/s`;
    if (abs < 1000) return `${ms.toFixed(2)} m/s`;
    if (abs < c / 10) return `${(ms / 1000).toFixed(2)} km/s`;
    return `${(ms / c).toFixed(4)}c`;
  },

  /**
   * Format energy with appropriate units
   * @param {number} joules - Energy in joules
   * @returns {string} Formatted energy
   */
  energy: (joules) => {
    const abs = Math.abs(joules);
    if (abs < 1e-18) return `${(joules / 1.602e-19).toFixed(2)} eV`;
    if (abs < 1e-15) return `${(joules / 1.602e-16).toFixed(2)} keV`;
    if (abs < 1e-12) return `${(joules / 1.602e-13).toFixed(2)} MeV`;
    if (abs < 1e-9) return `${(joules / 1.602e-10).toFixed(2)} GeV`;
    if (abs < 1) return `${(joules * 1e9).toFixed(2)} nJ`;
    if (abs < 1e3) return `${joules.toFixed(2)} J`;
    if (abs < 1e6) return `${(joules / 1e3).toFixed(2)} kJ`;
    return `${(joules / 1e6).toFixed(2)} MJ`;
  }
};

/**
 * Animation utilities
 */
export const animation = {
  /**
   * Ease-in-out cubic function
   * @param {number} t - Time parameter (0-1)
   * @returns {number} Eased value
   */
  easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,

  /**
   * Ease-in quadratic function
   * @param {number} t - Time parameter (0-1)
   * @returns {number} Eased value
   */
  easeInQuad: (t) => t * t,

  /**
   * Ease-out quadratic function
   * @param {number} t - Time parameter (0-1)
   * @returns {number} Eased value
   */
  easeOutQuad: (t) => t * (2 - t),

  /**
   * Simple linear interpolation for animations
   * @param {number} start - Start value
   * @param {number} end - End value
   * @param {number} progress - Progress (0-1)
   * @param {Function} easingFn - Easing function
   * @returns {number} Animated value
   */
  animate: (start, end, progress, easingFn = (t) => t) => {
    const easedProgress = easingFn(math.clamp(progress, 0, 1));
    return math.lerp(start, end, easedProgress);
  }
};

/**
 * Color utilities
 */
export const color = {
  /**
   * Interpolate between two colors
   * @param {string} color1 - Start color (hex)
   * @param {string} color2 - End color (hex)
   * @param {number} factor - Interpolation factor (0-1)
   * @returns {string} Interpolated color (hex)
   */
  interpolate: (color1, color2, factor) => {
    const c1 = parseInt(color1.slice(1), 16);
    const c2 = parseInt(color2.slice(1), 16);
    
    const r1 = (c1 >> 16) & 255;
    const g1 = (c1 >> 8) & 255;
    const b1 = c1 & 255;
    
    const r2 = (c2 >> 16) & 255;
    const g2 = (c2 >> 8) & 255;
    const b2 = c2 & 255;
    
    const r = Math.round(r1 + factor * (r2 - r1));
    const g = Math.round(g1 + factor * (g2 - g1));
    const b = Math.round(b1 + factor * (b2 - b1));
    
    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
  },

  /**
   * Generate color based on temperature (for stellar visualization)
   * @param {number} temperature - Temperature in Kelvin
   * @returns {string} Color (hex)
   */
  temperatureToColor: (temperature) => {
    // Simplified blackbody radiation color approximation
    if (temperature < 3500) return '#ff4500'; // Red giant
    if (temperature < 5000) return '#ffa500'; // Orange
    if (temperature < 6000) return '#ffff00'; // Yellow (Sun-like)
    if (temperature < 7500) return '#ffffff'; // White
    if (temperature < 10000) return '#87ceeb'; // Blue-white
    return '#4169e1'; // Blue
  }
};

/**
 * Performance utilities
 */
export const performance = {
  /**
   * Debounce function calls
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in ms
   * @returns {Function} Debounced function
   */
  debounce: (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  /**
   * Throttle function calls
   * @param {Function} func - Function to throttle
   * @param {number} limit - Limit in ms
   * @returns {Function} Throttled function
   */
  throttle: (func, limit) => {
    let inThrottle;
    return function executedFunction(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  /**
   * Request animation frame wrapper
   * @param {Function} callback - Callback function
   * @returns {number} Request ID
   */
  raf: (callback) => {
    if (typeof requestAnimationFrame !== 'undefined') {
      return requestAnimationFrame(callback);
    }
    return setTimeout(callback, 16); // Fallback for 60fps
  }
};

/**
 * Local storage utilities
 */
export const storage = {
  /**
   * Get item from localStorage with JSON parsing
   * @param {string} key - Storage key
   * @param {*} defaultValue - Default value if not found
   * @returns {*} Parsed value
   */
  get: (key, defaultValue = null) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return defaultValue;
    }
  },

  /**
   * Set item in localStorage with JSON stringification
   * @param {string} key - Storage key
   * @param {*} value - Value to store
   */
  set: (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error writing to localStorage:', error);
    }
  },

  /**
   * Remove item from localStorage
   * @param {string} key - Storage key
   */
  remove: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from localStorage:', error);
    }
  },

  /**
   * Clear all localStorage
   */
  clear: () => {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  }
};
