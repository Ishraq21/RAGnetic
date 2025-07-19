import logging
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import conversation_metrics_table

logger = logging.getLogger("ragnetic")

async def save_conversation_metrics(db: AsyncSession, metrics_data: dict):
    """Saves a conversation metrics record to the database."""
    try:
        stmt = insert(conversation_metrics_table).values(**metrics_data)
        await db.execute(stmt)
        await db.commit()
        logger.info(f"Saved conversation metrics for request_id: {metrics_data['request_id']}")
    except Exception as e:
        logger.error(f"Failed to save conversation metrics: {e}", exc_info=True)