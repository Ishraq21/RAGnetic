# app/services/mock_gpu_service.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random

logger = logging.getLogger(__name__)


class MockGPUService:
    """Mock GPU service that provides RunPod-style GPU data for testing and development."""
    
    def __init__(self):
        self.mock_gpus = self._initialize_mock_gpus()
        self.mock_providers = self._initialize_mock_providers()
        self.mock_instances = []
        self.instance_counter = 1
    
    def _initialize_mock_gpus(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock GPU types based on RunPod's actual GPU offerings."""
        return {
            "NVIDIA A100 80GB PCIe": {
                "id": "NVIDIA A100 80GB PCIe",
                "displayName": "A100 80GB",
                "memoryInGb": 80,
                "manufacturer": "Nvidia",
                "cudaCores": 6912,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 1.89,
                "communityPrice": 1.59,
                "communitySpotPrice": 0.89,
                "secureSpotPrice": 1.20,
                "lowestPrice": {
                    "minimumBidPrice": 0.89,
                    "uninterruptablePrice": 1.59
                }
            },
            "NVIDIA A100-SXM4-80GB": {
                "id": "NVIDIA A100-SXM4-80GB",
                "displayName": "A100 SXM 80GB",
                "memoryInGb": 80,
                "manufacturer": "Nvidia",
                "cudaCores": 6912,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 2.19,
                "communityPrice": 1.89,
                "communitySpotPrice": 1.09,
                "secureSpotPrice": 1.50,
                "lowestPrice": {
                    "minimumBidPrice": 1.09,
                    "uninterruptablePrice": 1.89
                }
            },
            "NVIDIA H100 PCIe": {
                "id": "NVIDIA H100 PCIe",
                "displayName": "H100 PCIe",
                "memoryInGb": 80,
                "manufacturer": "Nvidia",
                "cudaCores": 16896,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 2.89,
                "communityPrice": 2.49,
                "communitySpotPrice": 1.49,
                "secureSpotPrice": 2.00,
                "lowestPrice": {
                    "minimumBidPrice": 1.49,
                    "uninterruptablePrice": 2.49
                }
            },
            "NVIDIA H100 SXM": {
                "id": "NVIDIA H100 SXM",
                "displayName": "H100 SXM",
                "memoryInGb": 80,
                "manufacturer": "Nvidia",
                "cudaCores": 16896,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 3.29,
                "communityPrice": 2.89,
                "communitySpotPrice": 1.79,
                "secureSpotPrice": 2.40,
                "lowestPrice": {
                    "minimumBidPrice": 1.79,
                    "uninterruptablePrice": 2.89
                }
            },
            "NVIDIA RTX 4090": {
                "id": "NVIDIA RTX 4090",
                "displayName": "RTX 4090",
                "memoryInGb": 24,
                "manufacturer": "Nvidia",
                "cudaCores": 16384,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.34,
                "communityPrice": 0.29,
                "communitySpotPrice": 0.19,
                "secureSpotPrice": 0.24,
                "lowestPrice": {
                    "minimumBidPrice": 0.19,
                    "uninterruptablePrice": 0.29
                }
            },
            "NVIDIA RTX 4080": {
                "id": "NVIDIA RTX 4080",
                "displayName": "RTX 4080",
                "memoryInGb": 16,
                "manufacturer": "Nvidia",
                "cudaCores": 9728,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.24,
                "communityPrice": 0.19,
                "communitySpotPrice": 0.14,
                "secureSpotPrice": 0.16,
                "lowestPrice": {
                    "minimumBidPrice": 0.14,
                    "uninterruptablePrice": 0.19
                }
            },
            "NVIDIA RTX 3090": {
                "id": "NVIDIA RTX 3090",
                "displayName": "RTX 3090",
                "memoryInGb": 24,
                "manufacturer": "Nvidia",
                "cudaCores": 10496,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.29,
                "communityPrice": 0.24,
                "communitySpotPrice": 0.17,
                "secureSpotPrice": 0.20,
                "lowestPrice": {
                    "minimumBidPrice": 0.17,
                    "uninterruptablePrice": 0.24
                }
            },
            "NVIDIA RTX 3080": {
                "id": "NVIDIA RTX 3080",
                "displayName": "RTX 3080",
                "memoryInGb": 10,
                "manufacturer": "Nvidia",
                "cudaCores": 8704,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.19,
                "communityPrice": 0.14,
                "communitySpotPrice": 0.10,
                "secureSpotPrice": 0.12,
                "lowestPrice": {
                    "minimumBidPrice": 0.10,
                    "uninterruptablePrice": 0.14
                }
            },
            "NVIDIA Tesla V100": {
                "id": "NVIDIA Tesla V100",
                "displayName": "Tesla V100",
                "memoryInGb": 16,
                "manufacturer": "Nvidia",
                "cudaCores": 5120,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.69,
                "communityPrice": 0.59,
                "communitySpotPrice": 0.39,
                "secureSpotPrice": 0.49,
                "lowestPrice": {
                    "minimumBidPrice": 0.39,
                    "uninterruptablePrice": 0.59
                }
            },
            "NVIDIA T4": {
                "id": "NVIDIA T4",
                "displayName": "T4",
                "memoryInGb": 16,
                "manufacturer": "Nvidia",
                "cudaCores": 2560,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.20,
                "communityPrice": 0.15,
                "communitySpotPrice": 0.10,
                "secureSpotPrice": 0.12,
                "lowestPrice": {
                    "minimumBidPrice": 0.10,
                    "uninterruptablePrice": 0.15
                }
            },
            "NVIDIA Tesla P100": {
                "id": "NVIDIA Tesla P100",
                "displayName": "Tesla P100",
                "memoryInGb": 16,
                "manufacturer": "Nvidia",
                "cudaCores": 3584,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.40,
                "communityPrice": 0.35,
                "communitySpotPrice": 0.25,
                "secureSpotPrice": 0.30,
                "lowestPrice": {
                    "minimumBidPrice": 0.25,
                    "uninterruptablePrice": 0.35
                }
            }
        }
    
    def _initialize_mock_providers(self) -> List[Dict[str, Any]]:
        """Initialize mock GPU providers based on the GPU types."""
        providers = []
        
        for gpu_id, gpu_info in self.mock_gpus.items():
            # Create multiple provider entries for different pricing tiers
            providers.extend([
                {
                    "name": "RunPod",
                    "gpu_type": gpu_info["displayName"],
                    "cost_per_hour": gpu_info["securePrice"],
                    "availability": True,
                    "provider_type": "secure"
                },
                {
                    "name": "RunPod Community",
                    "gpu_type": gpu_info["displayName"],
                    "cost_per_hour": gpu_info["communityPrice"],
                    "availability": True,
                    "provider_type": "community"
                },
                {
                    "name": "RunPod Spot",
                    "gpu_type": gpu_info["displayName"],
                    "cost_per_hour": gpu_info["communitySpotPrice"],
                    "availability": random.choice([True, True, True, False]),  # 75% availability
                    "provider_type": "spot"
                }
            ])
        
        return providers
    
    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """Get all available GPU types in RunPod format."""
        return list(self.mock_gpus.values())
    
    async def get_gpu_by_id(self, gpu_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific GPU by its ID."""
        return self.mock_gpus.get(gpu_id)
    
    async def get_gpu_providers(self) -> List[Dict[str, Any]]:
        """Get all available GPU providers."""
        return [p for p in self.mock_providers if p["availability"]]
    
    async def get_gpu_pricing(self) -> List[Dict[str, Any]]:
        """Get GPU pricing information."""
        return [
            {
                "gpu_type": provider["gpu_type"],
                "provider": provider["name"],
                "cost_per_hour": provider["cost_per_hour"]
            }
            for provider in await self.get_gpu_providers()
        ]
    
    async def create_instance(self, gpu_type: str, provider: str, user_id: int, project_id: int, **kwargs) -> Dict[str, Any]:
        """Create a mock GPU instance."""
        # Find the provider info
        provider_info = None
        for p in self.mock_providers:
            if p["name"] == provider and p["gpu_type"] == gpu_type:
                provider_info = p
                break
        
        if not provider_info:
            raise ValueError(f"Provider {provider} with GPU {gpu_type} not found")
        
        instance_id = self.instance_counter
        self.instance_counter += 1
        
        # Create mock instance
        instance = {
            "id": instance_id,
            "project_id": project_id,
            "user_id": user_id,
            "gpu_type": gpu_type,
            "provider": provider,
            "status": "pending",
            "instance_id": f"mock-instance-{instance_id}",
            "cost_per_hour": provider_info["cost_per_hour"],
            "total_cost": 0.0,
            "started_at": None,
            "stopped_at": None,
            "created_at": datetime.utcnow()
        }
        
        self.mock_instances.append(instance)
        
        # Simulate instance starting after a short delay
        self._simulate_instance_start(instance_id)
        
        return instance
    
    def _simulate_instance_start(self, instance_id: int):
        """Simulate an instance starting up."""
        def update_status():
            # Find the instance
            instance = None
            for inst in self.mock_instances:
                if inst["id"] == instance_id:
                    instance = inst
                    break
            
            if instance:
                # Update to running status
                instance["status"] = "running"
                instance["started_at"] = datetime.utcnow()
                logger.info(f"Mock instance {instance_id} started")
        
        # Simulate startup delay (in real implementation, this would be async)
        import threading
        import time
        timer = threading.Timer(2.0, update_status)  # 2 second delay
        timer.start()
    
    async def get_user_instances(self, user_id: int, project_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user's GPU instances."""
        instances = [inst for inst in self.mock_instances if inst["user_id"] == user_id]
        
        if project_id:
            instances = [inst for inst in instances if inst["project_id"] == project_id]
        
        if status:
            instances = [inst for inst in instances if inst["status"] == status]
        
        return sorted(instances, key=lambda x: x["created_at"], reverse=True)
    
    async def stop_instance(self, instance_id: int, user_id: int) -> bool:
        """Stop a GPU instance."""
        for instance in self.mock_instances:
            if instance["id"] == instance_id and instance["user_id"] == user_id:
                if instance["status"] in ["running", "pending"]:
                    instance["status"] = "stopped"
                    instance["stopped_at"] = datetime.utcnow()
                    
                    # Calculate total cost
                    if instance["started_at"]:
                        duration = instance["stopped_at"] - instance["started_at"]
                        hours = duration.total_seconds() / 3600
                        instance["total_cost"] = hours * instance["cost_per_hour"]
                    
                    logger.info(f"Mock instance {instance_id} stopped")
                    return True
                break
        
        return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status (for RunPod compatibility)."""
        # Find instance by instance_id
        instance = None
        for inst in self.mock_instances:
            if inst["instance_id"] == instance_id:
                instance = inst
                break
        
        if not instance:
            return {
                "status": "failed",
                "message": "Instance not found"
            }
        
        # Map our status to RunPod status
        status_mapping = {
            "pending": "PROVISIONING",
            "running": "RUNNING",
            "stopped": "STOPPED",
            "failed": "UNKNOWN"
        }
        
        container_status = status_mapping.get(instance["status"], "UNKNOWN")
        uptime_seconds = 0
        
        if instance["started_at"]:
            end_time = instance["stopped_at"] or datetime.utcnow()
            uptime_seconds = int((end_time - instance["started_at"]).total_seconds())
        
        return {
            "status": instance["status"],
            "uptime_minutes": uptime_seconds // 60 if uptime_seconds else None,
            "message": f"Container status: {container_status}",
            "container_status": container_status,
            "uptime_seconds": uptime_seconds
        }
    
    async def create_usage_entries(self, instance_id: int, usage_type: str = "training") -> List[Dict[str, Any]]:
        """Create mock usage entries for an instance."""
        # Find the instance
        instance = None
        for inst in self.mock_instances:
            if inst["id"] == instance_id:
                instance = inst
                break
        
        if not instance or not instance["started_at"]:
            return []
        
        # Create usage entries
        usage_entries = []
        start_time = instance["started_at"]
        end_time = instance["stopped_at"] or datetime.utcnow()
        
        # Create hourly usage entries
        current_time = start_time
        while current_time < end_time:
            next_time = min(current_time + timedelta(hours=1), end_time)
            duration_minutes = int((next_time - current_time).total_seconds() / 60)
            cost = (duration_minutes / 60) * instance["cost_per_hour"]
            
            usage_entries.append({
                "instance_id": instance_id,
                "usage_type": usage_type,
                "duration_minutes": duration_minutes,
                "cost": cost,
                "created_at": current_time
            })
            
            current_time = next_time
        
        return usage_entries


# Global mock service instance
mock_gpu_service = MockGPUService()
