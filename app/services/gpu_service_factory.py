# app/services/gpu_service_factory.py

import os
import logging
from typing import Union
from .mock_gpu_service import MockGPUService
from .real_gpu_service import RealGPUService

logger = logging.getLogger(__name__)


def get_gpu_service() -> Union[MockGPUService, RealGPUService]:
    """Get the appropriate GPU service based on environment configuration."""
    
    # Check if RunPod API key is available
    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    environment = os.getenv("ENVIRONMENT", "development")
    force_mock = os.getenv("FORCE_MOCK_GPU", "false").lower() == "true"
    
    # Force mock mode for testing
    if force_mock:
        logger.info("Using mock GPU service (forced)")
        return MockGPUService()
    
    # Use real service if API key is available and in production
    if runpod_api_key and environment == "production":
        logger.info("Using real RunPod GPU service")
        return RealGPUService(runpod_api_key)
    
    # Use real service if API key is available and explicitly enabled
    if runpod_api_key and os.getenv("ENABLE_REAL_GPU", "false").lower() == "true":
        logger.info("Using real RunPod GPU service (explicitly enabled)")
        return RealGPUService(runpod_api_key)
    
    # Default to mock service
    logger.info("Using mock GPU service (default)")
    return MockGPUService()


# Global instance - will be initialized when first imported
_gpu_service = None


def get_gpu_service_instance() -> Union[MockGPUService, RealGPUService]:
    """Get the global GPU service instance."""
    global _gpu_service
    if _gpu_service is None:
        _gpu_service = get_gpu_service()
    return _gpu_service


def reset_gpu_service():
    """Reset the global GPU service instance (useful for testing)."""
    global _gpu_service
    _gpu_service = None
    logger.info("GPU service instance reset")


# For backward compatibility
gpu_service = get_gpu_service_instance()
