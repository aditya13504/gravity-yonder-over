// Validation utilities for physics parameters and user input

/**
 * Physics parameter validators
 */
export const physicsValidators = {
  /**
   * Validate mass value
   * @param {number} mass - Mass in kg
   * @returns {Object} {isValid, error, warnings}
   */
  validateMass: (mass) => {
    const result = { isValid: true, error: null, warnings: [] };
    
    if (typeof mass !== 'number' || isNaN(mass)) {
      result.isValid = false;
      result.error = 'Mass must be a valid number';
      return result;
    }
    
    if (mass <= 0) {
      result.isValid = false;
      result.error = 'Mass must be positive';
      return result;
    }
    
    if (mass > 1e50) {
      result.warnings.push('Mass is extremely large - may cause numerical instability');
    }
    
    if (mass < 1e-30) {
      result.warnings.push('Mass is extremely small - may not be physically meaningful');
    }
    
    return result;
  },

  /**
   * Validate distance value
   * @param {number} distance - Distance in meters
   * @returns {Object} {isValid, error, warnings}
   */
  validateDistance: (distance) => {
    const result = { isValid: true, error: null, warnings: [] };
    
    if (typeof distance !== 'number' || isNaN(distance)) {
      result.isValid = false;
      result.error = 'Distance must be a valid number';
      return result;
    }
    
    if (distance <= 0) {
      result.isValid = false;
      result.error = 'Distance must be positive';
      return result;
    }
    
    if (distance > 1e26) {
      result.warnings.push('Distance is larger than observable universe');
    }
    
    if (distance < 1e-15) {
      result.warnings.push('Distance is smaller than atomic scale');
    }
    
    return result;
  },

  /**
   * Validate velocity value
   * @param {number} velocity - Velocity in m/s
   * @returns {Object} {isValid, error, warnings}
   */
  validateVelocity: (velocity) => {
    const result = { isValid: true, error: null, warnings: [] };
    const c = 299792458; // Speed of light
    
    if (typeof velocity !== 'number' || isNaN(velocity)) {
      result.isValid = false;
      result.error = 'Velocity must be a valid number';
      return result;
    }
    
    if (Math.abs(velocity) >= c) {
      result.isValid = false;
      result.error = 'Velocity cannot exceed speed of light';
      return result;
    }
    
    if (Math.abs(velocity) > 0.1 * c) {
      result.warnings.push('Velocity is relativistic - consider using relativistic calculations');
    }
    
    return result;
  },

  /**
   * Validate time step for numerical integration
   * @param {number} timeStep - Time step in seconds
   * @param {Object} systemParams - System parameters for stability check
   * @returns {Object} {isValid, error, warnings}
   */
  validateTimeStep: (timeStep, systemParams = {}) => {
    const result = { isValid: true, error: null, warnings: [] };
    
    if (typeof timeStep !== 'number' || isNaN(timeStep)) {
      result.isValid = false;
      result.error = 'Time step must be a valid number';
      return result;
    }
    
    if (timeStep <= 0) {
      result.isValid = false;
      result.error = 'Time step must be positive';
      return result;
    }
    
    if (timeStep > 86400) { // 1 day
      result.warnings.push('Large time step may reduce accuracy');
    }
    
    if (timeStep < 1e-6) {
      result.warnings.push('Very small time step may cause performance issues');
    }
    
    // Check stability for orbital systems
    if (systemParams.orbitalPeriod) {
      const stepsPerOrbit = systemParams.orbitalPeriod / timeStep;
      if (stepsPerOrbit < 100) {
        result.warnings.push('Time step may be too large for stable orbit calculation');
      }
    }
    
    return result;
  },

  /**
   * Validate orbital parameters
   * @param {Object} params - Orbital parameters
   * @returns {Object} {isValid, error, warnings}
   */
  validateOrbitParameters: (params) => {
    const result = { isValid: true, error: null, warnings: [] };
    const { semiMajorAxis, eccentricity, inclination, centralMass } = params;
    
    // Validate semi-major axis
    const axisValidation = physicsValidators.validateDistance(semiMajorAxis);
    if (!axisValidation.isValid) {
      result.isValid = false;
      result.error = `Semi-major axis: ${axisValidation.error}`;
      return result;
    }
    
    // Validate eccentricity
    if (typeof eccentricity !== 'number' || isNaN(eccentricity)) {
      result.isValid = false;
      result.error = 'Eccentricity must be a valid number';
      return result;
    }
    
    if (eccentricity < 0) {
      result.isValid = false;
      result.error = 'Eccentricity cannot be negative';
      return result;
    }
    
    if (eccentricity >= 1) {
      result.warnings.push('Eccentricity ≥ 1 indicates unbound orbit');
    }
    
    // Validate inclination
    if (inclination !== undefined) {
      if (typeof inclination !== 'number' || isNaN(inclination)) {
        result.isValid = false;
        result.error = 'Inclination must be a valid number';
        return result;
      }
      
      if (inclination < 0 || inclination > 180) {
        result.warnings.push('Inclination should be between 0° and 180°');
      }
    }
    
    // Validate central mass
    if (centralMass !== undefined) {
      const massValidation = physicsValidators.validateMass(centralMass);
      if (!massValidation.isValid) {
        result.isValid = false;
        result.error = `Central mass: ${massValidation.error}`;
        return result;
      }
    }
    
    return result;
  },

  /**
   * Validate black hole parameters
   * @param {Object} params - Black hole parameters
   * @returns {Object} {isValid, error, warnings}
   */
  validateBlackHoleParameters: (params) => {
    const result = { isValid: true, error: null, warnings: [] };
    const { mass, spinParameter, charge } = params;
    const G = 6.67430e-11;
    const c = 299792458;
    
    // Validate mass
    const massValidation = physicsValidators.validateMass(mass);
    if (!massValidation.isValid) {
      result.isValid = false;
      result.error = `Black hole mass: ${massValidation.error}`;
      return result;
    }
    
    // Check if mass is sufficient for black hole formation
    const solarMass = 1.989e30;
    if (mass < 3 * solarMass) {
      result.warnings.push('Mass is below typical stellar black hole threshold');
    }
    
    // Validate spin parameter (a/M)
    if (spinParameter !== undefined) {
      if (typeof spinParameter !== 'number' || isNaN(spinParameter)) {
        result.isValid = false;
        result.error = 'Spin parameter must be a valid number';
        return result;
      }
      
      if (Math.abs(spinParameter) > 1) {
        result.isValid = false;
        result.error = 'Spin parameter magnitude cannot exceed 1 (extremal limit)';
        return result;
      }
      
      if (Math.abs(spinParameter) > 0.998) {
        result.warnings.push('Near-extremal black hole - numerical calculations may be unstable');
      }
    }
    
    // Validate charge (for Reissner-Nordström black holes)
    if (charge !== undefined) {
      if (typeof charge !== 'number' || isNaN(charge)) {
        result.isValid = false;
        result.error = 'Charge must be a valid number';
        return result;
      }
      
      // Check extremal charge limit
      const extremalCharge = Math.sqrt(G * mass / (4 * Math.PI * 8.854e-12 * c * c));
      if (Math.abs(charge) > extremalCharge) {
        result.isValid = false;
        result.error = 'Charge exceeds extremal limit - would create naked singularity';
        return result;
      }
    }
    
    return result;
  }
};

/**
 * User input validators
 */
export const inputValidators = {
  /**
   * Validate email format
   * @param {string} email - Email address
   * @returns {Object} {isValid, error}
   */
  validateEmail: (email) => {
    const result = { isValid: true, error: null };
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (!email || typeof email !== 'string') {
      result.isValid = false;
      result.error = 'Email is required';
      return result;
    }
    
    if (!emailRegex.test(email)) {
      result.isValid = false;
      result.error = 'Please enter a valid email address';
      return result;
    }
    
    return result;
  },

  /**
   * Validate username
   * @param {string} username - Username
   * @returns {Object} {isValid, error}
   */
  validateUsername: (username) => {
    const result = { isValid: true, error: null };
    
    if (!username || typeof username !== 'string') {
      result.isValid = false;
      result.error = 'Username is required';
      return result;
    }
    
    if (username.length < 3) {
      result.isValid = false;
      result.error = 'Username must be at least 3 characters long';
      return result;
    }
    
    if (username.length > 20) {
      result.isValid = false;
      result.error = 'Username cannot exceed 20 characters';
      return result;
    }
    
    if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
      result.isValid = false;
      result.error = 'Username can only contain letters, numbers, hyphens, and underscores';
      return result;
    }
    
    return result;
  },

  /**
   * Validate simulation name
   * @param {string} name - Simulation name
   * @returns {Object} {isValid, error}
   */
  validateSimulationName: (name) => {
    const result = { isValid: true, error: null };
    
    if (!name || typeof name !== 'string') {
      result.isValid = false;
      result.error = 'Simulation name is required';
      return result;
    }
    
    if (name.trim().length === 0) {
      result.isValid = false;
      result.error = 'Simulation name cannot be empty';
      return result;
    }
    
    if (name.length > 50) {
      result.isValid = false;
      result.error = 'Simulation name cannot exceed 50 characters';
      return result;
    }
    
    return result;
  },

  /**
   * Validate numeric input with range
   * @param {*} value - Input value
   * @param {Object} options - Validation options
   * @returns {Object} {isValid, error}
   */
  validateNumericInput: (value, options = {}) => {
    const { min, max, required = true, label = 'Value' } = options;
    const result = { isValid: true, error: null };
    
    if (required && (value === null || value === undefined || value === '')) {
      result.isValid = false;
      result.error = `${label} is required`;
      return result;
    }
    
    if (value !== null && value !== undefined && value !== '') {
      const numValue = Number(value);
      
      if (isNaN(numValue)) {
        result.isValid = false;
        result.error = `${label} must be a valid number`;
        return result;
      }
      
      if (min !== undefined && numValue < min) {
        result.isValid = false;
        result.error = `${label} must be at least ${min}`;
        return result;
      }
      
      if (max !== undefined && numValue > max) {
        result.isValid = false;
        result.error = `${label} cannot exceed ${max}`;
        return result;
      }
    }
    
    return result;
  }
};

/**
 * Form validation utilities
 */
export const formValidators = {
  /**
   * Validate entire simulation form
   * @param {Object} formData - Form data object
   * @returns {Object} {isValid, errors, warnings}
   */
  validateSimulationForm: (formData) => {
    const result = { isValid: true, errors: {}, warnings: {} };
    
    // Validate each field
    const fields = [
      { key: 'mass1', validator: physicsValidators.validateMass },
      { key: 'mass2', validator: physicsValidators.validateMass },
      { key: 'distance', validator: physicsValidators.validateDistance },
      { key: 'velocity', validator: physicsValidators.validateVelocity },
      { key: 'timeStep', validator: physicsValidators.validateTimeStep }
    ];
    
    fields.forEach(({ key, validator }) => {
      if (formData[key] !== undefined) {
        const validation = validator(formData[key]);
        if (!validation.isValid) {
          result.isValid = false;
          result.errors[key] = validation.error;
        }
        if (validation.warnings && validation.warnings.length > 0) {
          result.warnings[key] = validation.warnings;
        }
      }
    });
    
    return result;
  },

  /**
   * Validate user registration form
   * @param {Object} formData - Registration form data
   * @returns {Object} {isValid, errors}
   */
  validateRegistrationForm: (formData) => {
    const result = { isValid: true, errors: {} };
    
    // Validate username
    const usernameValidation = inputValidators.validateUsername(formData.username);
    if (!usernameValidation.isValid) {
      result.isValid = false;
      result.errors.username = usernameValidation.error;
    }
    
    // Validate email
    const emailValidation = inputValidators.validateEmail(formData.email);
    if (!emailValidation.isValid) {
      result.isValid = false;
      result.errors.email = emailValidation.error;
    }
    
    // Validate password
    if (!formData.password) {
      result.isValid = false;
      result.errors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      result.isValid = false;
      result.errors.password = 'Password must be at least 6 characters long';
    }
    
    // Validate password confirmation
    if (formData.password !== formData.confirmPassword) {
      result.isValid = false;
      result.errors.confirmPassword = 'Passwords do not match';
    }
    
    return result;
  }
};

/**
 * Real-time validation utilities
 */
export const realtimeValidators = {
  /**
   * Create debounced validator
   * @param {Function} validator - Validation function
   * @param {number} delay - Debounce delay in ms
   * @returns {Function} Debounced validator
   */
  createDebouncedValidator: (validator, delay = 300) => {
    let timeout;
    return (value, callback) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        const result = validator(value);
        callback(result);
      }, delay);
    };
  },

  /**
   * Create real-time physics parameter validator
   * @param {Function} onValidation - Callback for validation results
   * @returns {Function} Validator function
   */
  createPhysicsValidator: (onValidation) => {
    return realtimeValidators.createDebouncedValidator((value, paramType) => {
      let validation;
      
      switch (paramType) {
        case 'mass':
          validation = physicsValidators.validateMass(value);
          break;
        case 'distance':
          validation = physicsValidators.validateDistance(value);
          break;
        case 'velocity':
          validation = physicsValidators.validateVelocity(value);
          break;
        default:
          validation = { isValid: true, error: null, warnings: [] };
      }
      
      onValidation(paramType, validation);
    });
  }
};
