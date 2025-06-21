# Backend API initialization
from .routes import api_router
from .middleware import setup_middleware

__all__ = ['api_router', 'setup_middleware']
