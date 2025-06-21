import os
from typing import Dict, Any

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///gravity_yonder.db')
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # API settings
    API_VERSION = os.environ.get('API_VERSION', 'v1')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # ML Model settings
    PINN_MODEL_PATH = os.environ.get('PINN_MODEL_PATH', 'ml_models/trained_models/pinn_gravity_v1.pth')
    TRAJECTORY_MODEL_PATH = os.environ.get('TRAJECTORY_MODEL_PATH', 'ml_models/trained_models/trajectory_predictor_v1.pth')
    ML_DEVICE = os.environ.get('ML_DEVICE', 'cpu')
    ML_BATCH_SIZE = int(os.environ.get('ML_BATCH_SIZE', '32'))
    
    # Physics Engine settings
    CUDA_ENABLED = os.environ.get('CUDA_ENABLED', 'False').lower() == 'true'
    CUDA_DEVICE = int(os.environ.get('CUDA_DEVICE', '0'))
    PHYSICS_PRECISION = os.environ.get('PHYSICS_PRECISION', 'float64')
    MAX_SIMULATION_TIME = int(os.environ.get('MAX_SIMULATION_TIME', '3600'))
    MAX_BODIES_PER_SIMULATION = int(os.environ.get('MAX_BODIES_PER_SIMULATION', '100'))
    
    # Caching settings
    CACHE_SIMULATION_RESULTS = os.environ.get('CACHE_SIMULATION_RESULTS', 'True').lower() == 'true'
    CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))
    CACHE_MAX_SIZE = int(os.environ.get('CACHE_MAX_SIZE', '1000'))
    
    # Rate limiting
    RATE_LIMIT_SIMULATIONS = os.environ.get('RATE_LIMIT_SIMULATIONS', '100 per minute')
    RATE_LIMIT_API = os.environ.get('RATE_LIMIT_API', '1000 per minute')
    RATE_LIMIT_ML = os.environ.get('RATE_LIMIT_ML', '50 per minute')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/gravity_yonder.log')
    LOG_MAX_SIZE = os.environ.get('LOG_MAX_SIZE', '10MB')
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', '5'))
    
    # Security
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', '1800'))
    BCRYPT_ROUNDS = int(os.environ.get('BCRYPT_ROUNDS', '12'))
    
    # Game settings
    LEADERBOARD_SIZE = int(os.environ.get('LEADERBOARD_SIZE', '100'))
    SCORE_VALIDATION = os.environ.get('SCORE_VALIDATION', 'True').lower() == 'true'
    ANTI_CHEAT_ENABLED = os.environ.get('ANTI_CHEAT_ENABLED', 'True').lower() == 'true'
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            'url': cls.DATABASE_URL,
            'echo': cls.DEBUG,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            'url': cls.REDIS_URL,
            'decode_responses': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'retry_on_timeout': True
        }
    
    @classmethod
    def get_ml_config(cls) -> Dict[str, Any]:
        """Get ML model configuration dictionary."""
        return {
            'pinn_model_path': cls.PINN_MODEL_PATH,
            'trajectory_model_path': cls.TRAJECTORY_MODEL_PATH,
            'device': cls.ML_DEVICE,
            'batch_size': cls.ML_BATCH_SIZE,
            'precision': cls.PHYSICS_PRECISION
        }
    
    @classmethod
    def get_physics_config(cls) -> Dict[str, Any]:
        """Get physics engine configuration dictionary."""
        return {
            'cuda_enabled': cls.CUDA_ENABLED,
            'cuda_device': cls.CUDA_DEVICE,
            'precision': cls.PHYSICS_PRECISION,
            'max_simulation_time': cls.MAX_SIMULATION_TIME,
            'max_bodies': cls.MAX_BODIES_PER_SIMULATION
        }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    FLASK_ENV = 'development'
    LOG_LEVEL = 'DEBUG'
    CACHE_SIMULATION_RESULTS = False  # Disable caching in development
    MOCK_ML_MODELS = True  # Use mock models for faster development

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'
    REDIS_URL = 'redis://localhost:6379/1'  # Use different Redis DB
    CACHE_SIMULATION_RESULTS = False
    MOCK_ML_MODELS = True
    RATE_LIMIT_SIMULATIONS = '1000 per minute'  # Relaxed for testing
    RATE_LIMIT_API = '10000 per minute'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'
    LOG_LEVEL = 'WARNING'
    
    # Production-specific overrides
    CACHE_SIMULATION_RESULTS = True
    MOCK_ML_MODELS = False
    SCORE_VALIDATION = True
    ANTI_CHEAT_ENABLED = True
    
    # Security hardening
    BCRYPT_ROUNDS = 14
    JWT_ACCESS_TOKEN_EXPIRES = 900  # 15 minutes

# Configuration dictionary
config_dict = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment name."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config_dict.get(config_name, config_dict['default'])
