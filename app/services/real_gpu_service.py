# app/services/real_gpu_service.py

import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class RealGPUService:
    """Real RunPod GPU service integration."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """Get available GPU types from RunPod API."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    f"{self.base_url}/gpu-types",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Transform RunPod API response to our format
                        gpus = []
                        for gpu in data.get("data", []):
                            gpus.append({
                                "id": gpu.get("id"),
                                "displayName": gpu.get("displayName"),
                                "memoryInGb": gpu.get("memoryInGb"),
                                "manufacturer": gpu.get("manufacturer"),
                                "cudaCores": gpu.get("cudaCores", 0),
                                "secureCloud": gpu.get("secureCloud", False),
                                "communityCloud": gpu.get("communityCloud", False),
                                "securePrice": gpu.get("securePrice", 0.0),
                                "communityPrice": gpu.get("communityPrice", 0.0),
                                "communitySpotPrice": gpu.get("communitySpotPrice", 0.0),
                                "secureSpotPrice": gpu.get("secureSpotPrice", 0.0),
                                "lowestPrice": gpu.get("lowestPrice", {})
                            })
                        return gpus
                    else:
                        logger.error(f"RunPod API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching GPU types from RunPod: {e}")
            return []
    
    async def get_gpu_providers(self) -> List[Dict[str, Any]]:
        """Get GPU providers with pricing information."""
        try:
            gpus = await self.get_available_gpus()
            providers = []
            
            for gpu in gpus:
                gpu_id = gpu["id"]
                
                # Add secure cloud provider
                if gpu.get("secureCloud"):
                    providers.append({
                        "name": "RunPod",
                        "gpu_type": gpu_id,
                        "cost_per_hour": gpu.get("securePrice", 0.0),
                        "availability": True,
                        "provider_type": "secure"
                    })
                
                # Add community cloud provider
                if gpu.get("communityCloud"):
                    providers.append({
                        "name": "RunPod Community",
                        "gpu_type": gpu_id,
                        "cost_per_hour": gpu.get("communityPrice", 0.0),
                        "availability": True,
                        "provider_type": "community"
                    })
                
                # Add spot pricing providers
                if gpu.get("communitySpotPrice", 0) > 0:
                    providers.append({
                        "name": "RunPod Spot",
                        "gpu_type": gpu_id,
                        "cost_per_hour": gpu.get("communitySpotPrice", 0.0),
                        "availability": True,
                        "provider_type": "spot"
                    })
            
            return providers
        except Exception as e:
            logger.error(f"Error getting GPU providers: {e}")
            return []
    
    async def get_gpu_pricing(self) -> List[Dict[str, Any]]:
        """Get GPU pricing information."""
        try:
            providers = await self.get_gpu_providers()
            pricing = []
            
            for provider in providers:
                pricing.append({
                    "gpu_type": provider["gpu_type"],
                    "provider": provider["name"],
                    "cost_per_hour": provider["cost_per_hour"],
                    "availability": provider["availability"]
                })
            
            return pricing
        except Exception as e:
            logger.error(f"Error getting GPU pricing: {e}")
            return []
    
    async def get_gpu_by_id(self, gpu_id: str) -> Optional[Dict[str, Any]]:
        """Get specific GPU details by ID."""
        try:
            gpus = await self.get_available_gpus()
            for gpu in gpus:
                if gpu["id"] == gpu_id:
                    return gpu
            return None
        except Exception as e:
            logger.error(f"Error getting GPU by ID {gpu_id}: {e}")
            return None
    
    async def create_instance(
        self, 
        gpu_type: str, 
        provider: str, 
        user_id: int, 
        project_id: int,
        container_disk_gb: int = 50,
        volume_gb: int = 0,
        ports: str = "8000/http",
        environment_vars: Optional[Dict[str, str]] = None,
        docker_args: str = "",
        start_jupyter: bool = False,
        start_ssh: bool = True
    ) -> Dict[str, Any]:
        """Create a real RunPod instance."""
        try:
            # Map provider names to RunPod cloud types
            cloud_type_map = {
                "RunPod": "SECURE",
                "RunPod Community": "COMMUNITY",
                "RunPod Spot": "COMMUNITY"  # Spot uses community cloud
            }
            
            cloud_type = cloud_type_map.get(provider, "COMMUNITY")
            
            # Prepare container environment
            container_env = environment_vars or {}
            if start_jupyter:
                container_env["JUPYTER_ENABLED"] = "true"
            if start_ssh:
                container_env["SSH_ENABLED"] = "true"
            
            # Prepare payload
            payload = {
                "gpuTypeId": gpu_type,
                "cloudType": cloud_type,
                "containerDiskSizeGb": container_disk_gb,
                "volumeSizeGb": volume_gb,
                "ports": ports,
                "containerEnv": container_env,
                "dockerArgs": docker_args,
                "name": f"ragnetic-{user_id}-{project_id}-{int(datetime.now().timestamp())}"
            }
            
            # Add spot pricing if using RunPod Spot
            if provider == "RunPod Spot":
                payload["bidPerGpu"] = 0.01  # Minimum bid
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/pods",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        pod_data = result.get("data", {})
                        
                        # Get cost per hour
                        cost_per_hour = await self._get_gpu_cost(gpu_type, provider)
                        
                        return {
                            "id": pod_data.get("id"),
                            "gpu_type": gpu_type,
                            "provider": provider,
                            "status": "starting",
                            "cost_per_hour": cost_per_hour,
                            "created_at": datetime.now().isoformat(),
                            "user_id": user_id,
                            "project_id": project_id,
                            "runpod_pod_id": pod_data.get("id"),
                            "container_disk_gb": container_disk_gb,
                            "volume_gb": volume_gb,
                            "ports": ports,
                            "environment_vars": environment_vars,
                            "docker_args": docker_args,
                            "start_jupyter": start_jupyter,
                            "start_ssh": start_ssh
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"RunPod API error creating instance: {response.status} - {error_text}")
                        raise Exception(f"Failed to create RunPod instance: {error_text}")
        except Exception as e:
            logger.error(f"Error creating GPU instance: {e}")
            raise
    
    async def stop_instance(self, instance_id: str, user_id: int) -> bool:
        """Stop a RunPod instance."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/pods/{instance_id}/stop",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully stopped RunPod instance {instance_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"RunPod API error stopping instance: {response.status} - {error_text}")
                        return False
        except Exception as e:
            logger.error(f"Error stopping GPU instance {instance_id}: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status from RunPod."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    f"{self.base_url}/pods/{instance_id}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        pod_data = result.get("data", {})
                        
                        # Map RunPod status to our status
                        runpod_status = pod_data.get("desiredStatus", "UNKNOWN")
                        status_map = {
                            "RUNNING": "running",
                            "STOPPED": "stopped",
                            "STARTING": "starting",
                            "STOPPING": "stopping",
                            "UNKNOWN": "unknown"
                        }
                        
                        return {
                            "id": instance_id,
                            "status": status_map.get(runpod_status, "unknown"),
                            "runpod_status": runpod_status,
                            "runtime": pod_data.get("runtime", {}),
                            "machine": pod_data.get("machine", {}),
                            "last_updated": datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"RunPod API error getting status: {response.status} - {error_text}")
                        return {"status": "error", "message": error_text}
        except Exception as e:
            logger.error(f"Error getting instance status {instance_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_user_instances(
        self, 
        user_id: int, 
        project_id: Optional[int] = None, 
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's GPU instances from RunPod."""
        try:
            # Note: RunPod API doesn't have user-specific filtering
            # We'll need to track this in our database
            # For now, return empty list - this should be handled by database queries
            return []
        except Exception as e:
            logger.error(f"Error getting user instances: {e}")
            return []
    
    async def _get_gpu_cost(self, gpu_type: str, provider: str) -> float:
        """Get GPU cost from RunPod pricing."""
        try:
            providers = await self.get_gpu_providers()
            for p in providers:
                if p["gpu_type"] == gpu_type and p["name"] == provider:
                    return p["cost_per_hour"]
            return 1.89  # Default fallback
        except Exception as e:
            logger.error(f"Error getting GPU cost: {e}")
            return 1.89  # Default fallback
    
    async def create_usage_entries(self, instance_id: int, usage_type: str) -> List[Dict[str, Any]]:
        """Create usage entries for billing (handled by database)."""
        # This is handled by the database layer, not the RunPod API
        return []
