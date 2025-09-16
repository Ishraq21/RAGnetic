# app/services/gpu_providers/base.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProvisionConfig:
    """Configuration for GPU provisioning."""
    gpu_type: str
    image: str = "ragnetic/training:latest"
    request_data: Optional[Dict[str, Any]] = None
    volume_mount: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None
    max_hours: Optional[float] = None


@dataclass
class ProvisionResult:
    """Result of GPU provisioning."""
    instance_id: str
    hourly_price: float
    status: str = "provisioning"
    message: Optional[str] = None


@dataclass
class InstanceStatus:
    """GPU instance status information."""
    status: str  # 'running', 'stopped', 'failed', 'provisioning'
    message: Optional[str] = None
    uptime_minutes: Optional[int] = None
    cost_so_far: Optional[float] = None


class GPUProviderClient(ABC):
    """Abstract base class for GPU provider clients."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def provision(self, config: ProvisionConfig) -> ProvisionResult:
        """
        Provision a new GPU instance.
        
        Args:
            config: Provisioning configuration
            
        Returns:
            ProvisionResult with instance details
        """
        pass
    
    @abstractmethod
    async def status(self, instance_id: str) -> InstanceStatus:
        """
        Get the status of a GPU instance.
        
        Args:
            instance_id: The instance identifier
            
        Returns:
            InstanceStatus with current state
        """
        pass
    
    @abstractmethod
    async def stop(self, instance_id: str) -> bool:
        """
        Stop a GPU instance.
        
        Args:
            instance_id: The instance identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def logs(self, instance_id: str, tail_kb: int = 100) -> str:
        """
        Get logs from a GPU instance.
        
        Args:
            instance_id: The instance identifier
            tail_kb: Number of KB of logs to retrieve from the end
            
        Returns:
            Log content as string
        """
        pass
    
    @abstractmethod
    async def get_available_gpus(self) -> Dict[str, Any]:
        """
        Get available GPU types and their pricing.
        
        Returns:
            Dictionary mapping GPU types to pricing information
        """
        pass
    
    def validate_config(self, config: ProvisionConfig) -> None:
        """Validate provisioning configuration."""
        if not config.gpu_type:
            raise ValueError("GPU type is required")
        
        if not config.image:
            raise ValueError("Docker image is required")
        
        if config.max_hours and config.max_hours <= 0:
            raise ValueError("Max hours must be positive")
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.replace("Client", "").lower()
