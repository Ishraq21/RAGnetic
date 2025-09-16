# app/api/invoke.py
import hashlib
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, and_

from app.core.security import verify_api_key
from app.core.rate_limit import check_api_rate_limit, get_rate_limit_config
from app.schemas.security import User
from app.db import get_db
from app.db.models import (
    api_keys_table, deployments_table, api_usage_table,
    user_credits_table, credit_transactions_table
)
from app.agents.config_manager import load_agent_config
from app.agents.agent_graph import get_agent_graph, AgentState
from app.services.credit_service import CreditService
from app.services.cost_service import CostService
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/api/v1/invoke", tags=["Invoke API"])
logger = logging.getLogger(__name__)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for verification."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def verify_api_key(api_key: str, db: AsyncSession) -> Optional[User]:
    """Verify API key and return associated user."""
    if not api_key:
        return None
    
    key_hash = hash_api_key(api_key)
    
    result = await db.execute(
        select(api_keys_table).where(
            and_(
                api_keys_table.c.key_hash == key_hash,
                api_keys_table.c.is_active == True
            )
        )
    )
    api_key_record = result.fetchone()
    
    if not api_key_record:
        return None
    
    # Update last used timestamp
    await db.execute(
        update(api_keys_table).where(
            api_keys_table.c.id == api_key_record.id
        ).values(last_used_at=datetime.utcnow())
    )
    
    # Create a minimal User object for the API key owner
    return User(
        id=api_key_record.user_id,
        user_id=str(api_key_record.user_id),
        email=None,
        first_name=None,
        last_name=None,
        is_active=True,
        is_superuser=False
    )


@router.post("/{deployment_id}")
async def invoke_deployment(
    deployment_id: int,
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db)
):
    """Stateless API invocation endpoint with proper verification and rate limiting."""
    try:
        # Verify API key using new verification function
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        verification_result = await verify_api_key(x_api_key, db)
        if not verification_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        api_key_id, user_id, scope = verification_result
        
        # Apply rate limiting
        rate_limit_config = get_rate_limit_config(scope)
        rate_limit_result = check_api_rate_limit(api_key_id, rate_limit_config)
        
        if not rate_limit_result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_limit_result.limit),
                    "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                    "X-RateLimit-Reset": str(rate_limit_result.reset_time),
                    "Retry-After": str(rate_limit_result.retry_after) if rate_limit_result.retry_after else "60"
                }
            )
        
        # Get request body
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in request body"
            )
        
        query = body.get("query", "")
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required"
            )
        
        # Verify deployment exists and is active
        deployment_result = await db.execute(
            select(deployments_table).where(
                and_(
                    deployments_table.c.id == deployment_id,
                    deployments_table.c.status == "active"
                )
            )
        )
        deployment = deployment_result.fetchone()
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found or inactive"
            )
        
        # Enforce scope restrictions
        if scope == "viewer" and deployment.deployment_type != "api":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this deployment type"
            )
        
        # Initialize services
        credit_service = CreditService(db)
        cost_service = CostService(db)
        
        # Estimate cost for API call
        estimated_cost = await cost_service.estimate_api_cost(1, "gpt-4o-mini")
        
        # Check user credits
        await credit_service.ensure_balance(user_id, estimated_cost)
        
        # Check agent status first
        from app.db.models import agents_table
        result = await db.execute(
            select(agents_table.c.status).where(agents_table.c.name == deployment.agent_id)
        )
        agent_status = result.scalar_one_or_none()
        
        if agent_status == "stopped":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{deployment.agent_id}' is currently stopped. Please deploy the agent before sending requests."
            )
        
        if agent_status == "error":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{deployment.agent_id}' is in an error state. Please retry deployment or contact support."
            )

        # Load agent configuration
        try:
            agent_config = load_agent_config(deployment.agent_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load agent configuration: {str(e)}"
            )
        
        # Create agent state and invoke
        try:
            agent_state = AgentState(
                messages=[HumanMessage(content=query)],
                thread_id=f"api_{deployment_id}_{datetime.utcnow().timestamp()}",
                user_id=user_id
            )
            
            agent_graph = get_agent_graph(agent_config)
            result = await agent_graph.ainvoke(agent_state)
            
            # Extract response
            if result.get("messages"):
                last_message = result["messages"][-1]
                response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                response_text = "No response generated"
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {str(e)}"
            )
        
        # Deduct credits and log usage
        try:
            # Deduct credits using credit service
            await credit_service.deduct(
                user_id,
                estimated_cost,
                f"API call to deployment {deployment_id}",
                gpu_instance_id=None
            )
            
            # Log API usage
            await db.execute(
                insert(api_usage_table).values(
                    api_key_id=api_key_id,
                    deployment_id=deployment_id,
                    request_count=1,
                    total_cost=estimated_cost,
                    last_request=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
            )
            
            await db.commit()
            
        except Exception as e:
            # Log the error but don't fail the request
            logger.error(f"Failed to update billing: {str(e)}")
            await db.rollback()
        
        return {
            "response": response_text,
            "deployment_id": deployment_id,
            "cost": estimated_cost,
            "timestamp": datetime.utcnow().isoformat(),
            "rate_limit": {
                "remaining": rate_limit_result.remaining,
                "reset_time": rate_limit_result.reset_time
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal server error in invoke_deployment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
