# app/services/gpu_providers/runpod.py

import json
import logging
from typing import Dict, Any, Optional
import httpx

from .base import GPUProviderClient, ProvisionConfig, ProvisionResult, InstanceStatus

logger = logging.getLogger(__name__)


class RunPodClient(GPUProviderClient):
    """RunPod GPU provider client implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.runpod.io"):
        super().__init__(api_key, base_url)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def provision(self, config: ProvisionConfig) -> ProvisionResult:
        """Provision a new GPU instance on RunPod."""
        try:
            self.validate_config(config)
            
            # Map GPU types to RunPod GPU IDs
            gpu_mapping = {
                "RTX 4090": "NVIDIA GeForce RTX 4090",
                "RTX 4080": "NVIDIA GeForce RTX 4080",
                "RTX 3090": "NVIDIA GeForce RTX 3090",
                "RTX 3080": "NVIDIA GeForce RTX 3080",
                "A100": "NVIDIA A100-SXM4-40GB",
                "V100": "NVIDIA Tesla V100",
                "H100": "NVIDIA H100 PCIe",
                "T4": "NVIDIA T4",
                "P100": "NVIDIA Tesla P100",
            }
            
            gpu_id = gpu_mapping.get(config.gpu_type, config.gpu_type)
            
            # Build pod configuration
            pod_config = {
                "name": f"ragnetic-{config.gpu_type.lower().replace(' ', '-')}",
                "imageName": config.image,
                "gpuIds": gpu_id,
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "volumeMountPath": "/workspace",
                "env": config.environment_vars or {},
                "ports": "8000/http",
                "isServerless": False,
                "startJupyter": False,
                "startSsh": True,
                "dockerArgs": "",
                "templateId": None,
                "networkVolumeId": None,
            }
            
            # Add request data if provided
            if config.request_data:
                pod_config.update(config.request_data)
            
            # Add volume mount if specified
            if config.volume_mount:
                pod_config["volumeInGb"] = 100  # Default volume size
                pod_config["volumeMountPath"] = config.volume_mount
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql",
                    headers=self.headers,
                    json={
                        "query": """
                        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
                            podFindAndDeployOnDemand(input: $input) {
                                id
                                imageName
                                env
                                machineId
                                machine {
                                    podHostId
                                }
                            }
                        }
                        """,
                        "variables": {
                            "input": pod_config
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"RunPod API error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    raise Exception(f"RunPod provisioning error: {error_msg}")
                
                pod_data = result["data"]["podFindAndDeployOnDemand"]
                instance_id = pod_data["id"]
                
                # Get pricing information
                hourly_price = await self._get_gpu_pricing(config.gpu_type)
                
                self.logger.info(f"Successfully provisioned RunPod instance {instance_id}")
                
                return ProvisionResult(
                    instance_id=instance_id,
                    hourly_price=hourly_price,
                    status="provisioning",
                    message="Instance is being provisioned"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to provision RunPod instance: {str(e)}")
            raise
    
    async def status(self, instance_id: str) -> InstanceStatus:
        """Get the status of a RunPod instance."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql",
                    headers=self.headers,
                    json={
                        "query": """
                        query Pod($input: PodQueryInput!) {
                            pod(input: $input) {
                                id
                                name
                                runtime {
                                    uptimeInSeconds
                                    ports {
                                        ip
                                        isIpPublic
                                        privatePort
                                        publicPort
                                        type
                                    }
                                    gpus {
                                        id
                                        gpuUtilPercent
                                        memoryUtilPercent
                                    }
                                    containerStatus
                                }
                                machine {
                                    podHostId
                                }
                            }
                        }
                        """,
                        "variables": {
                            "input": {
                                "podId": instance_id
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"RunPod API error: {response.status_code}")
                
                result = response.json()
                
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    return InstanceStatus(
                        status="failed",
                        message=f"Error: {error_msg}"
                    )
                
                pod_data = result["data"]["pod"]
                if not pod_data:
                    return InstanceStatus(
                        status="failed",
                        message="Instance not found"
                    )
                
                runtime = pod_data.get("runtime", {})
                container_status = runtime.get("containerStatus", "UNKNOWN")
                uptime_seconds = runtime.get("uptimeInSeconds", 0)
                
                # Map RunPod status to our status
                status_mapping = {
                    "RUNNING": "running",
                    "STOPPED": "stopped",
                    "UNKNOWN": "failed",
                    "PROVISIONING": "provisioning"
                }
                
                status = status_mapping.get(container_status, "failed")
                uptime_minutes = uptime_seconds // 60 if uptime_seconds else None
                
                return InstanceStatus(
                    status=status,
                    uptime_minutes=uptime_minutes,
                    message=f"Container status: {container_status}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get RunPod instance status: {str(e)}")
            return InstanceStatus(
                status="failed",
                message=f"Error: {str(e)}"
            )
    
    async def stop(self, instance_id: str) -> bool:
        """Stop a RunPod instance."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql",
                    headers=self.headers,
                    json={
                        "query": """
                        mutation PodStop($input: PodStopInput!) {
                            podStop(input: $input) {
                                id
                            }
                        }
                        """,
                        "variables": {
                            "input": {
                                "podId": instance_id
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    self.logger.error(f"RunPod stop API error: {response.status_code}")
                    return False
                
                result = response.json()
                
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    self.logger.error(f"RunPod stop error: {error_msg}")
                    return False
                
                self.logger.info(f"Successfully stopped RunPod instance {instance_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop RunPod instance: {str(e)}")
            return False
    
    async def logs(self, instance_id: str, tail_kb: int = 100) -> str:
        """Get logs from a RunPod instance."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql",
                    headers=self.headers,
                    json={
                        "query": """
                        query PodLogs($input: PodLogsInput!) {
                            podLogs(input: $input) {
                                logs
                            }
                        }
                        """,
                        "variables": {
                            "input": {
                                "podId": instance_id,
                                "tail": tail_kb * 1024  # Convert KB to bytes
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    return f"Error fetching logs: {response.status_code}"
                
                result = response.json()
                
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    return f"Error: {error_msg}"
                
                logs_data = result["data"]["podLogs"]
                return logs_data.get("logs", "No logs available")
                
        except Exception as e:
            self.logger.error(f"Failed to get RunPod logs: {str(e)}")
            return f"Error fetching logs: {str(e)}"
    
    async def get_available_gpus(self) -> Dict[str, Any]:
        """Get available GPU types and their pricing from RunPod."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql",
                    headers=self.headers,
                    json={
                        "query": """
                        query GpuTypes {
                            gpuTypes {
                                id
                                displayName
                                memoryInGb
                                secureCloud
                                communityCloud
                                lowestPrice {
                                    minimumBidPrice
                                    uninterruptablePrice
                                }
                            }
                        }
                        """
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    return {}
                
                result = response.json()
                
                if "errors" in result:
                    return {}
                
                gpu_types = result["data"]["gpuTypes"]
                
                # Format the response
                available_gpus = {}
                for gpu in gpu_types:
                    gpu_id = gpu["id"]
                    display_name = gpu["displayName"]
                    memory = gpu["memoryInGb"]
                    lowest_price = gpu["lowestPrice"]
                    
                    available_gpus[gpu_id] = {
                        "display_name": display_name,
                        "memory_gb": memory,
                        "price_per_hour": lowest_price.get("uninterruptablePrice", 0),
                        "bid_price": lowest_price.get("minimumBidPrice", 0)
                    }
                
                return available_gpus
                
        except Exception as e:
            self.logger.error(f"Failed to get RunPod GPU types: {str(e)}")
            return {}
    
    async def _get_gpu_pricing(self, gpu_type: str) -> float:
        """Get pricing for a specific GPU type."""
        try:
            gpu_types = await self.get_available_gpus()
            
            # Try to find exact match first
            if gpu_type in gpu_types:
                return gpu_types[gpu_type]["price_per_hour"]
            
            # Try to find by display name
            for gpu_id, gpu_info in gpu_types.items():
                if gpu_type.lower() in gpu_info["display_name"].lower():
                    return gpu_info["price_per_hour"]
            
            # Fallback to default pricing
            default_pricing = {
                "RTX 4090": 0.34,
                "RTX 4080": 0.24,
                "RTX 3090": 0.29,
                "RTX 3080": 0.19,
                "A100": 1.10,
                "V100": 0.69,
                "H100": 2.89,
                "T4": 0.20,
                "P100": 0.40,
            }
            
            return default_pricing.get(gpu_type, 0.50)
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU pricing: {str(e)}")
            return 0.50  # Safe fallback
