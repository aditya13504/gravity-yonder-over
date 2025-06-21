// Interpolation utilities for pre-computed physics data
export class PhysicsInterpolator {
  constructor() {
    this.cache = new Map();
  }

  /**
   * Linear interpolation between two points
   * @param {number} x - Input value
   * @param {number} x0 - Lower bound x
   * @param {number} x1 - Upper bound x
   * @param {number} y0 - Value at x0
   * @param {number} y1 - Value at x1
   * @returns {number} Interpolated value
   */
  linearInterpolate(x, x0, x1, y0, y1) {
    if (x1 === x0) return y0;
    const t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  }

  /**
   * Bilinear interpolation for 2D data
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {Array} data - 2D array of values
   * @param {Array} xRange - [xMin, xMax]
   * @param {Array} yRange - [yMin, yMax]
   * @returns {number} Interpolated value
   */
  bilinearInterpolate(x, y, data, xRange, yRange) {
    const [xMin, xMax] = xRange;
    const [yMin, yMax] = yRange;
    const rows = data.length;
    const cols = data[0].length;

    // Normalize coordinates to array indices
    const xNorm = (x - xMin) / (xMax - xMin) * (cols - 1);
    const yNorm = (y - yMin) / (yMax - yMin) * (rows - 1);

    // Get surrounding grid points
    const x0 = Math.floor(xNorm);
    const x1 = Math.min(x0 + 1, cols - 1);
    const y0 = Math.floor(yNorm);
    const y1 = Math.min(y0 + 1, rows - 1);

    // Get values at corners
    const f00 = data[y0][x0];
    const f01 = data[y1][x0];
    const f10 = data[y0][x1];
    const f11 = data[y1][x1];

    // Interpolate
    const fx0 = this.linearInterpolate(xNorm, x0, x1, f00, f10);
    const fx1 = this.linearInterpolate(xNorm, x0, x1, f01, f11);
    
    return this.linearInterpolate(yNorm, y0, y1, fx0, fx1);
  }

  /**
   * Cubic spline interpolation
   * @param {number} x - Input value
   * @param {Array} xData - Array of x values
   * @param {Array} yData - Array of y values
   * @returns {number} Interpolated value
   */
  cubicSplineInterpolate(x, xData, yData) {
    const n = xData.length;
    if (n !== yData.length) {
      throw new Error('X and Y data arrays must have the same length');
    }

    // Find the interval containing x
    let i = 0;
    for (i = 0; i < n - 1; i++) {
      if (x <= xData[i + 1]) break;
    }

    // Handle edge cases
    if (x < xData[0]) return yData[0];
    if (x > xData[n - 1]) return yData[n - 1];

    // Simplified cubic interpolation using finite differences
    const h = xData[i + 1] - xData[i];
    const t = (x - xData[i]) / h;
    
    let y0 = yData[i];
    let y1 = yData[i + 1];
    let dy0 = i > 0 ? (yData[i] - yData[i - 1]) / (xData[i] - xData[i - 1]) : 0;
    let dy1 = i < n - 2 ? (yData[i + 2] - yData[i + 1]) / (xData[i + 2] - xData[i + 1]) : 0;

    // Hermite interpolation
    const h00 = (1 + 2 * t) * (1 - t) * (1 - t);
    const h10 = t * (1 - t) * (1 - t);
    const h01 = t * t * (3 - 2 * t);
    const h11 = t * t * (t - 1);

    return h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1;
  }

  /**
   * Interpolate gravitational field strength from pre-computed data
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {Object} fieldData - Pre-computed field data
   * @returns {Object} {fx, fy} - Force components
   */
  interpolateGravityField(x, y, fieldData) {
    const cacheKey = `field_${x}_${y}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const fx = this.bilinearInterpolate(
      x, y, 
      fieldData.fx, 
      fieldData.xRange, 
      fieldData.yRange
    );

    const fy = this.bilinearInterpolate(
      x, y, 
      fieldData.fy, 
      fieldData.xRange, 
      fieldData.yRange
    );

    const result = { fx, fy };
    this.cache.set(cacheKey, result);
    return result;
  }

  /**
   * Interpolate orbital trajectory from pre-computed data
   * @param {number} time - Time parameter
   * @param {Array} trajectoryData - Pre-computed trajectory points
   * @returns {Object} {x, y, vx, vy} - Position and velocity
   */
  interpolateTrajectory(time, trajectoryData) {
    if (trajectoryData.length < 2) {
      return trajectoryData[0] || { x: 0, y: 0, vx: 0, vy: 0 };
    }

    // Find the time interval
    let i = 0;
    for (i = 0; i < trajectoryData.length - 1; i++) {
      if (time <= trajectoryData[i + 1].t) break;
    }

    const p0 = trajectoryData[i];
    const p1 = trajectoryData[Math.min(i + 1, trajectoryData.length - 1)];

    if (p0.t === p1.t) return p0;

    const t = (time - p0.t) / (p1.t - p0.t);

    return {
      x: this.linearInterpolate(t, 0, 1, p0.x, p1.x),
      y: this.linearInterpolate(t, 0, 1, p0.y, p1.y),
      vx: this.linearInterpolate(t, 0, 1, p0.vx, p1.vx),
      vy: this.linearInterpolate(t, 0, 1, p0.vy, p1.vy)
    };
  }

  /**
   * Interpolate potential energy field
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {Object} potentialData - Pre-computed potential data
   * @returns {number} Potential energy
   */
  interpolatePotential(x, y, potentialData) {
    return this.bilinearInterpolate(
      x, y,
      potentialData.values,
      potentialData.xRange,
      potentialData.yRange
    );
  }

  /**
   * Fast lookup for commonly used values using hash tables
   * @param {string} key - Lookup key
   * @param {*} value - Value to store
   */
  cacheLookup(key, value = null) {
    if (value !== null) {
      this.cache.set(key, value);
      return value;
    }
    return this.cache.get(key);
  }

  /**
   * Clear interpolation cache
   */
  clearCache() {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache stats
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }

  /**
   * Adaptive interpolation that chooses method based on data characteristics
   * @param {number} x - Input value
   * @param {Array} xData - X data points
   * @param {Array} yData - Y data points
   * @param {string} method - 'auto', 'linear', 'cubic'
   * @returns {number} Interpolated value
   */
  adaptiveInterpolate(x, xData, yData, method = 'auto') {
    if (method === 'auto') {
      // Choose method based on data size and characteristics
      if (xData.length <= 3) {
        method = 'linear';
      } else {
        // Check for smoothness (simple heuristic)
        let maxCurvature = 0;
        for (let i = 1; i < xData.length - 1; i++) {
          const curvature = Math.abs(
            yData[i-1] - 2 * yData[i] + yData[i+1]
          );
          maxCurvature = Math.max(maxCurvature, curvature);
        }
        
        method = maxCurvature > 0.1 * Math.max(...yData) ? 'cubic' : 'linear';
      }
    }

    switch (method) {
      case 'cubic':
        return this.cubicSplineInterpolate(x, xData, yData);
      case 'linear':
      default:
        // Find surrounding points for linear interpolation
        let i = 0;
        for (i = 0; i < xData.length - 1; i++) {
          if (x <= xData[i + 1]) break;
        }
        return this.linearInterpolate(x, xData[i], xData[i + 1], yData[i], yData[i + 1]);
    }
  }
}
