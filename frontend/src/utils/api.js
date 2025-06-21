// API communication utilities
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };
  }

  /**
   * Generic request method
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Request options
   * @returns {Promise} API response
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: this.headers,
      ...options
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  /**
   * GET request
   * @param {string} endpoint - API endpoint
   * @param {Object} params - Query parameters
   * @returns {Promise} API response
   */
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    
    return this.request(url, {
      method: 'GET'
    });
  }

  /**
   * POST request
   * @param {string} endpoint - API endpoint
   * @param {Object} data - Request body data
   * @returns {Promise} API response
   */
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  /**
   * PUT request
   * @param {string} endpoint - API endpoint
   * @param {Object} data - Request body data
   * @returns {Promise} API response
   */
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  /**
   * DELETE request
   * @param {string} endpoint - API endpoint
   * @returns {Promise} API response
   */
  async delete(endpoint) {
    return this.request(endpoint, {
      method: 'DELETE'
    });
  }
}

// Create singleton instance
const apiClient = new ApiClient();

// Specific API functions for gravity simulation
export const gravityAPI = {
  /**
   * Run gravity simulation
   * @param {Object} params - Simulation parameters
   * @returns {Promise} Simulation results
   */
  runSimulation: async (params) => {
    return apiClient.post('/api/simulate', params);
  },

  /**
   * Get pre-computed trajectory data
   * @param {string} scenarioId - Scenario identifier
   * @returns {Promise} Trajectory data
   */
  getTrajectory: async (scenarioId) => {
    return apiClient.get(`/api/trajectories/${scenarioId}`);
  },

  /**
   * Get gravity field data
   * @param {Object} params - Field parameters
   * @returns {Promise} Field data
   */
  getGravityField: async (params) => {
    return apiClient.get('/api/gravity-field', params);
  },

  /**
   * Calculate orbital parameters
   * @param {Object} params - Initial conditions
   * @returns {Promise} Orbital parameters
   */
  calculateOrbit: async (params) => {
    return apiClient.post('/api/orbit-calculation', params);
  },

  /**
   * Get relativistic calculations
   * @param {Object} params - Relativistic parameters
   * @returns {Promise} Relativistic effects data
   */
  getRelativisticEffects: async (params) => {
    return apiClient.post('/api/relativistic-effects', params);
  },

  /**
   * Validate physics parameters
   * @param {Object} params - Parameters to validate
   * @returns {Promise} Validation results
   */
  validateParameters: async (params) => {
    return apiClient.post('/api/validate-parameters', params);
  },

  /**
   * Get educational content
   * @param {string} topicId - Topic identifier
   * @returns {Promise} Educational content
   */
  getEducationalContent: async (topicId) => {
    return apiClient.get(`/api/education/${topicId}`);
  },

  /**
   * Submit game score
   * @param {Object} scoreData - Score and game data
   * @returns {Promise} Submission result
   */
  submitScore: async (scoreData) => {
    return apiClient.post('/api/scores', scoreData);
  },

  /**
   * Get leaderboard
   * @param {string} gameId - Game identifier
   * @returns {Promise} Leaderboard data
   */
  getLeaderboard: async (gameId) => {
    return apiClient.get(`/api/leaderboard/${gameId}`);
  },

  /**
   * Save user progress
   * @param {Object} progressData - User progress data
   * @returns {Promise} Save result
   */
  saveProgress: async (progressData) => {
    return apiClient.post('/api/progress', progressData);
  },

  /**
   * Load user progress
   * @param {string} userId - User identifier
   * @returns {Promise} User progress data
   */
  loadProgress: async (userId) => {
    return apiClient.get(`/api/progress/${userId}`);
  },

  /**
   * Get simulation presets
   * @returns {Promise} Available simulation presets
   */
  getSimulationPresets: async () => {
    return apiClient.get('/api/presets');
  },

  /**
   * Save custom simulation
   * @param {Object} simulationData - Custom simulation parameters
   * @returns {Promise} Save result
   */
  saveCustomSimulation: async (simulationData) => {
    return apiClient.post('/api/custom-simulations', simulationData);
  },

  /**
   * Load custom simulations
   * @param {string} userId - User identifier
   * @returns {Promise} User's custom simulations
   */
  loadCustomSimulations: async (userId) => {
    return apiClient.get(`/api/custom-simulations/${userId}`);
  }
};

// Error handling wrapper
export const withErrorHandling = (apiFunction) => {
  return async (...args) => {
    try {
      return await apiFunction(...args);
    } catch (error) {
      console.error('API Error:', error);
      
      // Handle specific error types
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Network error. Please check your connection.');
      } else if (error.message.includes('404')) {
        throw new Error('Resource not found.');
      } else if (error.message.includes('500')) {
        throw new Error('Server error. Please try again later.');
      } else {
        throw error;
      }
    }
  };
};

// Request caching for performance
class RequestCache {
  constructor(maxSize = 100) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key) {
    if (this.cache.has(key)) {
      // Move to end (LRU)
      const value = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, value);
      return value;
    }
    return null;
  }

  set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  clear() {
    this.cache.clear();
  }
}

const requestCache = new RequestCache();

// Cached API wrapper
export const withCaching = (apiFunction, ttl = 300000) => { // 5 minutes default TTL
  return async (...args) => {
    const cacheKey = JSON.stringify([apiFunction.name, ...args]);
    const cached = requestCache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.data;
    }
    
    const result = await apiFunction(...args);
    requestCache.set(cacheKey, {
      data: result,
      timestamp: Date.now()
    });
    
    return result;
  };
};

export default apiClient;
