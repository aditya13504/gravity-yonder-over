from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from typing import Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI):
    """
    Set up middleware for the FastAPI application
    """
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React development server
            "http://localhost:5173",  # Vite development server
            "https://gravity-yonder-over.vercel.app",  # Production frontend
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Gzip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware for request logging and timing
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable):
        """
        Log requests and measure response time
        """
        start_time = time.time()
        
        # Log request details
        logger.info(f"Request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Add response time header
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response details
            logger.info(
                f"Response: {response.status_code} - "
                f"Time: {process_time:.3f}s - "
                f"Path: {request.url.path}"
            )
            
            return response
            
        except Exception as e:
            # Log errors
            process_time = time.time() - start_time
            logger.error(
                f"Error processing request: {request.method} {request.url.path} - "
                f"Time: {process_time:.3f}s - "
                f"Error: {str(e)}"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    # Rate limiting middleware (basic implementation)
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable):
        """
        Basic rate limiting middleware
        """
        client_ip = request.client.host
        
        # In production, you would use Redis or similar for rate limiting
        # For now, this is a placeholder
        
        response = await call_next(request)
        return response
    
    # Security headers middleware
    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable):
        """
        Add security headers to responses
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Custom exception handlers
async def validation_exception_handler(request: Request, exc: Exception):
    """
    Handle validation errors
    """
    logger.warning(f"Validation error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "message": str(exc),
            "type": "validation_error"
        }
    )

async def http_exception_handler(request: Request, exc: Exception):
    """
    Handle HTTP exceptions
    """
    logger.error(f"HTTP error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content={
            "detail": str(exc),
            "type": "http_error"
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle generic exceptions
    """
    logger.error(f"Unhandled error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )

def setup_exception_handlers(app: FastAPI):
    """
    Set up exception handlers for the FastAPI application
    """
    from fastapi.exceptions import RequestValidationError, HTTPException
    
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

# Request/Response models for middleware
class RequestMetrics:
    """
    Store request metrics for monitoring
    """
    def __init__(self):
        self.total_requests = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.endpoint_stats = {}
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """
        Record request metrics
        """
        self.total_requests += 1
        self.total_response_time += response_time
        
        if status_code >= 400:
            self.error_count += 1
        
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0.0,
                'errors': 0
            }
        
        self.endpoint_stats[endpoint]['count'] += 1
        self.endpoint_stats[endpoint]['total_time'] += response_time
        
        if status_code >= 400:
            self.endpoint_stats[endpoint]['errors'] += 1
    
    def get_average_response_time(self):
        """
        Get average response time
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    def get_error_rate(self):
        """
        Get error rate percentage
        """
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100

# Global metrics instance
request_metrics = RequestMetrics()

def setup_middleware(app):
    """Setup all middleware for the FastAPI app"""
    
    # Middleware for API versioning
    @app.middleware("http")
    async def api_versioning_middleware(request: Request, call_next: Callable):
        """
        Handle API versioning
        """
        # Check for API version in headers
        api_version = request.headers.get("X-API-Version", "v1")
        
        # Add version to request state
        request.state.api_version = api_version
        
        response = await call_next(request)
        
        # Add version to response headers
        response.headers["X-API-Version"] = api_version
        
        return response

    # Health check middleware
    @app.middleware("http")
    async def health_check_middleware(request: Request, call_next: Callable):
        """
        Basic health check functionality
        """
        if request.url.path == "/health":
            # Quick health check without processing through full stack
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "timestamp": time.time(),
                    "metrics": {
                        "total_requests": request_metrics.total_requests,
                        "average_response_time": request_metrics.get_average_response_time(),
                        "error_rate": request_metrics.get_error_rate()
                    }
                }
            )
        
        return await call_next(request)
