# app/db/dao.py
import json
import logging
from sqlalchemy import insert, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as SyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound
from app.db.models import conversation_metrics_table, users_table, roles_table, user_api_keys_table, \
    role_permissions_table, user_organizations_table, organizations_table, document_chunks_table, citations_table, \
    chat_sessions_table, temporary_documents_table, benchmark_runs_table, benchmark_items_table
from app.schemas.security import UserCreate, UserUpdate, RoleCreate
from app.db.models import conversation_metrics_table

from typing import Optional, List, Dict, Any
import bcrypt
from datetime import datetime, timedelta
import secrets
from app.schemas.agent import DocumentMetadata
from app.db.models import chat_messages_table
from uuid import uuid4
from sqlalchemy import func
from app.schemas.security import RoleCreate
from app.db.models import lambda_runs
from sqlalchemy import literal_column


logger = logging.getLogger("ragnetic")


def _duration_seconds(expr_start, expr_end, bind) -> Any:
    """Return a SQLAlchemy expression for (end - start) in seconds across dialects."""
    dname = bind.dialect.name if bind is not None else "sqlite"
    if dname == "sqlite":
        # (days) * 86400
        return 86400 * (func.julianday(expr_end) - func.julianday(expr_start))
    # Postgres / others: extract epoch from interval
    return func.extract('epoch', expr_end - expr_start)


# --- Password Hashing Utilities ---
def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed password."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:  # Hashed password might be malformed
        return False


# --- User Management ---

async def create_user(db: AsyncSession, user_in: UserCreate) -> Optional[Dict[str, Any]]:
    """Creates a new user and optionally assigns roles."""
    try:
        hashed_pw = hash_password(user_in.password)
        stmt = insert(users_table).values(
            user_id=user_in.username,  # Use username as user_id in DB
            email=user_in.email,
            first_name=user_in.first_name,
            last_name=user_in.last_name,
            hashed_password=hashed_pw,
            is_active=user_in.is_active,
            is_superuser=user_in.is_superuser,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        ).returning(
            users_table.c.id,
            users_table.c.user_id,  # Return user_id from DB
            users_table.c.email,
            users_table.c.first_name,
            users_table.c.last_name,
            users_table.c.hashed_password,
            users_table.c.is_active,
            users_table.c.is_superuser,
            users_table.c.created_at,
            users_table.c.updated_at
        )

        result = await db.execute(stmt)
        new_user_row = result.mappings().first()  # Get the row as a mapping

        if not new_user_row:
            # This case should ideally not happen if returning() works as expected
            raise Exception("Failed to retrieve newly created user data.")

        # Convert the SQLAlchemy RowMapping to a dict that matches Pydantic User schema
        new_user_data_dict = dict(new_user_row)
        new_user_data_dict['username'] = new_user_data_dict['user_id']  # Map 'user_id' to 'username' for Pydantic

        # Assign roles if provided
        if user_in.roles:
            for role_name in user_in.roles:
                try:
                    await assign_role_to_user(db, new_user_data_dict['id'], role_name)
                except ValueError as ve:
                    logger.warning(
                        f"Could not assign role '{role_name}' to new user {new_user_data_dict['username']}: {ve}")

        await db.commit()  # Commit after all assignments

        # Fetch the complete user object with assigned roles for accurate return
        # This re-fetches to ensure all relationships (roles) are loaded correctly
        return await get_user_by_id(db, new_user_data_dict['id'])
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Attempted to create duplicate user or email: {user_in.username} / {user_in.email}")
        raise ValueError("User with this username or email already exists.")
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create user {user_in.username}: {e}", exc_info=True)
        raise


async def _format_user_data_for_pydantic(db_row: Any, db: AsyncSession) -> Dict[str, Any]:
    """Helper to format a raw SQLAlchemy user row into a dictionary suitable for the User Pydantic model."""
    user_dict = dict(db_row)
    user_dict['username'] = user_dict['user_id']  # Map user_id to username for Pydantic
    user_dict['roles'] = await get_user_roles_data(db, user_dict['id'])
    return user_dict


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their username, including roles and permissions."""
    stmt = select(users_table).where(users_table.c.user_id == username)
    result = await db.execute(stmt)
    user_row = result.mappings().first()
    if user_row:
        row_map = dict(user_row)
        user_dict = await _format_user_data_for_pydantic(row_map, db)
        # include the scope explicitly
        user_dict["api_key_scope"] = row_map.get("scope")
        return user_dict
    return None


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their ID, including roles and permissions."""
    stmt = select(users_table).where(users_table.c.id == user_id)
    result = await db.execute(stmt)
    user_row = result.mappings().first()
    if user_row:
        return await _format_user_data_for_pydantic(user_row, db)
    return None


async def get_user_roles_data(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    """Helper to fetch roles and their permissions for a given user ID."""

    stmt = select(
        roles_table.c.id,
        roles_table.c.name,
        roles_table.c.description
    ).select_from(
        user_organizations_table.join(roles_table)
    ).where(user_organizations_table.c.user_id == user_id)

    result = await db.execute(stmt)
    roles = []
    for row in result.mappings().all():
        role_dict = dict(row)
        # Fetch permissions for each role
        role_dict["permissions"] = await get_permissions_for_role(db, role_dict["id"])
        roles.append(role_dict)
    return roles


async def update_user(db: AsyncSession, user_id: int, user_update: UserUpdate) -> Optional[Dict[str, Any]]:
    """Updates an existing user's details and optionally their roles."""
    update_values = user_update.model_dump(exclude_unset=True)

    if "password" in update_values:
        update_values["hashed_password"] = hash_password(update_values.pop("password"))
    # 'username' in UserUpdate maps to 'user_id' column in db
    if "username" in update_values:
        update_values["user_id"] = update_values.pop("username")

    update_values["updated_at"] = datetime.utcnow()

    # Exclude roles from the direct update on users_table as they are handled separately
    roles_to_assign_or_remove = update_values.pop("roles", None)

    stmt = update(users_table).where(users_table.c.id == user_id).values(**update_values).returning(users_table.c.id)

    try:
        result = await db.execute(stmt)
        updated_id = result.scalar_one_or_none()

        if updated_id is None:
            raise NoResultFound("User not found.")

        # Handle role updates if provided in the UserUpdate model
        if roles_to_assign_or_remove is not None:
            # Get current roles for the user
            current_role_names_stmt = select(roles_table.c.name).select_from(
                user_organizations_table.join(roles_table)
            ).where(user_organizations_table.c.user_id == user_id)

            current_role_names = [r[0] for r in (await db.execute(current_role_names_stmt)).fetchall()]

            roles_to_add = set(roles_to_assign_or_remove) - set(current_role_names)
            roles_to_remove = set(current_role_names) - set(
                user_update.roles)

            for role_name in roles_to_add:
                try:
                    await assign_role_to_user(db, user_id, role_name)
                except ValueError as ve:
                    logger.warning(f"Failed to assign role '{role_name}' to user {user_id}: {ve}")
            for role_name in roles_to_remove:
                await remove_role_from_user(db, user_id, role_name)

        await db.commit()
        return await get_user_by_id(db, user_id)  # Fetch the complete updated user data
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Attempted to update user to duplicate username or email for user ID: {user_id}")
        raise ValueError("Another user with this username or email already exists.")
    except NoResultFound:
        await db.rollback()
        raise ValueError("User not found.")  # Re-raise as ValueError for consistent API error handling
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update user {user_id}: {e}", exc_info=True)
        raise


async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """Deletes a user by their ID."""
    stmt = delete(users_table).where(users_table.c.id == user_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


async def get_all_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves a paginated list of all users."""
    stmt = select(users_table).offset(skip).limit(limit)
    result = await db.execute(stmt)
    users_data = []
    for row in result.mappings().all():
        users_data.append(await _format_user_data_for_pydantic(row, db))
    return users_data


# --- Role Management ---

async def create_role(db: AsyncSession, role_in: RoleCreate) -> Dict[str, Any]:
    """Creates a new role."""
    try:
        stmt = insert(roles_table).values(
            name=role_in.name,
            description=role_in.description
        ).returning(roles_table.c.id, roles_table.c.name, roles_table.c.description)
        result = await db.execute(stmt)
        new_role = result.mappings().first()
        await db.commit()
        return dict(new_role)
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Attempted to create duplicate role: {role_in.name}")
        raise ValueError("Role with this name already exists.")
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create role {role_in.name}: {e}", exc_info=True)
        raise


async def get_role_by_name(db: AsyncSession, role_name: str) -> Optional[Dict[str, Any]]:

    stmt = select(roles_table).where(roles_table.c.name == role_name)
    result = await db.execute(stmt)
    role_row = result.mappings().first()
    if role_row:
        role_dict = dict(role_row)
        role_dict["permissions"] = await get_permissions_for_role(db, role_dict["id"])  # <-- fix
        return role_dict
    return None



async def get_role_by_id(db: AsyncSession, role_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a role by its ID, including its permissions."""
    stmt = select(roles_table).where(roles_table.c.id == role_id)
    result = await db.execute(stmt)
    role_row = result.mappings().first()
    if role_row:
        role_dict = dict(role_row)
        role_dict["permissions"] = await get_permissions_for_role(db, role_dict["id"])
        return role_dict
    return None


async def get_all_roles(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves a paginated list of all roles, including their permissions."""
    stmt = select(roles_table).offset(skip).limit(limit)
    result = await db.execute(stmt)
    roles_data = []
    for row in result.mappings().all():
        role_dict = dict(row)
        role_dict["permissions"] = await get_permissions_for_role(db, role_dict["id"])
        roles_data.append(role_dict)
    return roles_data


async def delete_role(db: AsyncSession, role_id: int) -> bool:
    """Deletes a role by its ID."""
    # Need to check for cascade on users_organizations_table if not set up in model
    # For now, rely on CASCADE defined in `user_organizations_table` ForeignKey
    stmt = delete(roles_table).where(roles_table.c.id == role_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# --- User-Role Assignment (via user_organizations_table) ---

async def assign_role_to_user(db: AsyncSession, user_id: int, role_name: str,
                              organization_name: Optional[str] = "default") -> bool:
    """Assigns a role to a user, optionally within a specified organization.
    Assumes a 'default' organization if not explicitly provided for global roles."""

    org_id = None
    if organization_name:
        org_stmt = select(organizations_table.c.id).where(organizations_table.c.name == organization_name)
        org_id = (await db.execute(org_stmt)).scalar_one_or_none()
        if not org_id:
            # Create the organization if it doesn't exist
            insert_org_stmt = insert(organizations_table).values(name=organization_name).returning(
                organizations_table.c.id)
            org_id = (await db.execute(insert_org_stmt)).scalar_one()
            await db.commit()  # Commit the org creation

    if not org_id:  # If after trying to get/create, it's still None (e.g., if organization_name was None and no 'default' could be created)
        raise ValueError(
            "Cannot assign role: No organization specified and default organization could not be found or created.")

    role_row = await db.execute(select(roles_table.c.id).where(roles_table.c.name == role_name))
    role_id = role_row.scalar_one_or_none()
    if not role_id:
        raise ValueError(f"Role '{role_name}' not found.")

    user_exists = await db.scalar(select(users_table.c.id).where(users_table.c.id == user_id))
    if not user_exists:
        raise ValueError(f"User with ID '{user_id}' not found.")

    stmt = insert(user_organizations_table).values(
        user_id=user_id,
        organization_id=org_id,
        role_id=role_id
    )
    try:
        await db.execute(stmt)
        await db.commit()
        return True
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Role '{role_name}' already assigned to user {user_id} for org {organization_name}.")
        return False  # Role already assigned
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to assign role '{role_name}' to user {user_id} in org {organization_name}: {e}",
                     exc_info=True)
        raise


async def remove_role_from_user(db: AsyncSession, user_id: int, role_name: str,
                                organization_name: Optional[str] = "default") -> bool:
    """Removes a role from a user, optionally within a specified organization."""

    org_id = None
    if organization_name:
        org_stmt = select(organizations_table.c.id).where(organizations_table.c.name == organization_name)
        org_id = (await db.execute(org_stmt)).scalar_one_or_none()
        # If organization_name was provided but not found, cannot remove specific role from that org.
        if not org_id:
            logger.warning(
                f"Organization '{organization_name}' not found when attempting to remove role '{role_name}' from user {user_id}.")
            return False

    role_row = await db.execute(select(roles_table.c.id).where(roles_table.c.name == role_name))
    role_id = role_row.scalar_one_or_none()
    if not role_id:
        logger.warning(f"Attempted to remove non-existent role '{role_name}' from user {user_id}.")
        return False

        # Build the delete statement dynamically based on whether organization_id is relevant
    where_clauses = [
        user_organizations_table.c.user_id == user_id,
        user_organizations_table.c.role_id == role_id,
    ]
    if org_id:  # Only add org_id clause if a specific org was requested and found
        where_clauses.append(user_organizations_table.c.organization_id == org_id)

    stmt = delete(user_organizations_table).where(*where_clauses)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# --- Permission Management for Roles ---

async def assign_permission_to_role(db: AsyncSession, role_id: int, permission: str) -> bool:
    """Assigns a permission to a role."""
    stmt = insert(role_permissions_table).values(
        role_id=role_id,
        permission=permission
    )
    try:
        await db.execute(stmt)
        await db.commit()
        return True
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Permission '{permission}' already assigned to role ID {role_id}.")
        return False  # Permission already assigned
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to assign permission '{permission}' to role ID {role_id}: {e}", exc_info=True)
        raise


async def remove_permission_from_role(db: AsyncSession, role_id: int, permission: str) -> bool:
    """Removes a permission from a role."""
    stmt = delete(role_permissions_table).where(
        role_permissions_table.c.role_id == role_id,
        role_permissions_table.c.permission == permission
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


async def get_permissions_for_role(db: AsyncSession, role_id: int) -> List[str]:
    """Retrieves all permissions for a given role ID."""
    stmt = select(role_permissions_table.c.permission).where(role_permissions_table.c.role_id == role_id)
    result = await db.execute(stmt)
    return result.scalars().all()


# --- User API Key Management ---

async def create_user_api_key(db: AsyncSession, user_id: int, scope: str) -> str:
    """Generates and stores a new API key for a user with a specific scope."""
    api_key = secrets.token_hex(32)
    stmt = insert(user_api_keys_table).values(
        user_id=user_id,
        api_key=api_key,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        revoked=False,
        scope=scope,
        last_used_at=None,
        request_count=0
    )
    try:
        await db.execute(stmt)
        await db.commit()
        return api_key
    except IntegrityError:
        await db.rollback()
        logger.error(f"Integrity error creating API key for user {user_id}. Key might already exist (rare).",
                     exc_info=True)
        raise ValueError("Failed to create unique API key.")
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create API key for user {user_id}: {e}", exc_info=True)
        raise


async def revoke_user_api_key(db: AsyncSession, api_key_str: str) -> bool:
    """Revokes a user's API key."""
    stmt = update(user_api_keys_table).where(user_api_keys_table.c.api_key == api_key_str).values(
        revoked=True,
        updated_at=datetime.utcnow()
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


async def get_user_by_api_key(db: AsyncSession, api_key_str: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user associated with a non-revoked API key, including roles and permissions, and the key's scope."""
    stmt = select(
        users_table,
        user_api_keys_table.c.scope,
    ).join(user_api_keys_table).where(
        user_api_keys_table.c.api_key == api_key_str,
        user_api_keys_table.c.revoked == False
    )
    result = await db.execute(stmt)
    user_row = result.mappings().first()
    if user_row:
        user_data = dict(user_row)
        return await _format_user_data_for_pydantic(user_data, db)
    return None

async def update_api_key_usage(db: AsyncSession, api_key_str: str) -> None:
    """Updates the last_used_at timestamp and increments the request_count for a given API key."""
    stmt = update(user_api_keys_table).where(
        user_api_keys_table.c.api_key == api_key_str
    ).values(
        last_used_at=datetime.utcnow(),
        request_count=user_api_keys_table.c.request_count + 1,
    )
    try:
        await db.execute(stmt)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update API key usage for key ending in ...{api_key_str[-4:]}: {e}", exc_info=True)


def save_conversation_metrics_sync(connection, metrics_data: dict):
    """Saves a conversation metrics record to the database using a synchronous connection."""
    try:
        connection.execute(conversation_metrics_table.insert().values(**metrics_data))
        connection.commit() # Commit the transaction for this single operation
        logger.info(f"Saved conversation metrics (sync) for request_id: {metrics_data['request_id']}")
    except Exception as e:
        logger.error(f"Failed to save conversation metrics (sync): {e}", exc_info=True)
        connection.rollback() # Rollback on error


def create_benchmark_run_sync(connection, run_data: dict) -> None:
    existing = connection.execute(
        select(benchmark_runs_table.c.run_id).where(benchmark_runs_table.c.run_id == run_data["run_id"])
    ).first()
    if existing:
        return
    connection.execute(benchmark_runs_table.insert().values(**run_data))

def update_benchmark_run_sync(connection, run_id: str, **fields) -> None:
    connection.execute(
        update(benchmark_runs_table).where(benchmark_runs_table.c.run_id == run_id).values(**fields)
    )

def increment_benchmark_progress_sync(connection, run_id: str, inc: int = 1) -> None:
    connection.execute(
        update(benchmark_runs_table)
        .where(benchmark_runs_table.c.run_id == run_id)
        .values(completed_items=benchmark_runs_table.c.completed_items + inc)
    )

def insert_benchmark_item_sync(connection, item: dict) -> None:
    connection.execute(benchmark_items_table.insert().values(**item))

# --- Document Chunk Management ---

async def create_document_chunk(
    db: AsyncSession,
    document_name: str,
    chunk_index: int,
    content: str,
    page_number: Optional[int] = None,
    row_number: Optional[int] = None,
    temp_document_id: Optional[int] = None,   # <<< NEW
) -> int:
    """Creates a new document chunk and returns its ID."""
    stmt = insert(document_chunks_table).values(
        document_name=document_name,
        chunk_index=chunk_index,
        content=content,
        page_number=page_number,
        row_number=row_number,
        temp_document_id=temp_document_id      # <<< NEW
    ).returning(document_chunks_table.c.id)
    result = await db.execute(stmt)
    await db.commit()
    return result.scalar_one()

async def get_document_chunk(db: AsyncSession, chunk_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a document chunk by its ID."""
    stmt = select(document_chunks_table).where(document_chunks_table.c.id == chunk_id)
    result = await db.execute(stmt)
    return result.mappings().first()


async def get_document_chunks(db: AsyncSession, chunk_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Retrieves a list of document chunks based on a list of chunk IDs.
    Returns a list of dictionaries, one for each found chunk.
    """
    if not chunk_ids:
        return []

    stmt = select(document_chunks_table).where(document_chunks_table.c.id.in_(chunk_ids))
    result = await db.execute(stmt)
    return [dict(row) for row in result.mappings().all()]

# --- Citation Management ---

async def create_citation(db: AsyncSession, message_id: int, chunk_id: int, marker_text: str, start_char: int, end_char: int) -> int:
    """Creates a new citation and returns its ID."""
    stmt = insert(citations_table).values(
        message_id=message_id,
        chunk_id=chunk_id,
        marker_text=marker_text,
        start_char=start_char,
        end_char=end_char
    ).returning(citations_table.c.id)
    result = await db.execute(stmt)
    await db.commit()
    return result.scalar_one()

async def get_citations_for_message(db: AsyncSession, message_id: int) -> List[Dict[str, Any]]:
    """Retrieves all citations for a given message."""
    stmt = select(citations_table).where(citations_table.c.message_id == message_id)
    result = await db.execute(stmt)
    return [dict(m) for m in result.mappings().all()]


async def create_chat_message(
        db: AsyncSession,
        session_id: int,
        sender: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None
) -> int:
    """
    Saves a new chat message and updates the parent session's timestamp.
    Returns the ID of the newly created chat message.
    """
    now = datetime.utcnow()

    # 1. Insert the new message
    msg_stmt = insert(chat_messages_table).values(
        session_id=session_id,
        sender=sender,
        content=content,
        meta=meta,
        timestamp=now
    ).returning(chat_messages_table.c.id)

    # 2. Update the session's 'updated_at' timestamp
    session_stmt = update(chat_sessions_table).where(
        chat_sessions_table.c.id == session_id
    ).values(updated_at=now)

    try:
        # Execute both statements
        result = await db.execute(msg_stmt)
        await db.execute(session_stmt)

        # Commit the transaction
        await db.commit()

        message_id = result.scalar_one()
        logger.info(f"Successfully created message {message_id} and updated session {session_id}.")
        return message_id
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create chat message or update session {session_id}: {e}", exc_info=True)
        raise



DEFAULT_TTL_SECONDS = 7 * 24 * 3600   # one week

async def create_temp_document(
    db: AsyncSession,
    user_id: int,
    thread_id: str,
    original_name: str,
    file_size: int,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Dict[str, Any]:
    """Insert a row in temporary_documents and return it as a dict."""
    temp_doc_id = str(uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

    stmt = (
        insert(temporary_documents_table)
        .values(
            temp_doc_id=temp_doc_id,
            user_id=user_id,
            thread_id=thread_id,
            original_name=original_name,
            file_size=file_size,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            cleaned_up=False,
        )
        .returning(*temporary_documents_table.c)
    )
    row = (await db.execute(stmt)).mappings().first()
    await db.commit()
    return dict(row)


async def get_temp_document(db: AsyncSession, temp_doc_id: str) -> Optional[Dict[str, Any]]:
    stmt = select(temporary_documents_table).where(
        temporary_documents_table.c.temp_doc_id == temp_doc_id,
        temporary_documents_table.c.cleaned_up == False,
    )
    row = (await db.execute(stmt)).mappings().first()
    return dict(row) if row else None


async def mark_temp_document_cleaned(db: AsyncSession, row_id: int) -> None:
    stmt = (
        update(temporary_documents_table)
        .where(temporary_documents_table.c.id == row_id)
        .values(cleaned_up=True)
    )
    await db.execute(stmt)
    await db.commit()


async def list_expired_temp_documents(db: AsyncSession) -> List[Dict[str, Any]]:
    """Return all rows from temporary_documents_table whose expires_at ≤ now AND not yet cleaned."""
    stmt = (
        select(temporary_documents_table)
        .where(
            temporary_documents_table.c.expires_at <= func.now(),
            temporary_documents_table.c.cleaned_up == False,
        )
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(r) for r in rows]

async def delete_temp_document_data(db: AsyncSession, temp_doc_id: str) -> bool:
    """
    Atomically deletes a temporary document record and all its associated
    chunks and citations. Returns True on success, False otherwise.
    (Note: This function should be called within a transaction context).
    """
    try:
        # Step 1: Find the temporary document row to get its ID
        temp_doc_row = (await db.execute(
            select(temporary_documents_table.c.id)
            .where(temporary_documents_table.c.temp_doc_id == temp_doc_id)
        )).scalar_one_or_none()
        if not temp_doc_row:
            logger.warning(f"Attempted to delete non-existent temp document with ID: {temp_doc_id}")
            return False
        # Step 2: Get all chunks associated with this temp document
        chunk_ids_to_delete = (await db.execute(
            select(document_chunks_table.c.id)
            .where(document_chunks_table.c.temp_document_id == temp_doc_row)
        )).scalars().all()
        if chunk_ids_to_delete:
            # Step 3: Delete citations linked to these chunks
            await db.execute(
                delete(citations_table)
                .where(citations_table.c.chunk_id.in_(chunk_ids_to_delete))
            )
            logger.info(f"Deleted citations for chunks of temp doc {temp_doc_id}.")
            # Step 4: Delete the chunks themselves
            await db.execute(
                delete(document_chunks_table)
                .where(document_chunks_table.c.id.in_(chunk_ids_to_delete))
            )
            logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for temp doc {temp_doc_id}.")
        # Step 5: Delete the temporary document record
        await db.execute(
            delete(temporary_documents_table)
            .where(temporary_documents_table.c.id == temp_doc_row)
        )
        logger.info(f"Deleted temporary document record for temp doc {temp_doc_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to atomically delete temp document data for {temp_doc_id}: {e}", exc_info=True)
        # Re-raise the exception to trigger a rollback in the calling function
        raise

def mark_temp_document_cleaned_sync(db: SyncSession, row_id: int) -> None:
    db.execute(
        update(temporary_documents_table)
        .where(temporary_documents_table.c.id == row_id)
        .values(cleaned_up=True)
    )

def delete_temp_document_data_sync(db: SyncSession, temp_doc_id: str) -> bool:
    """
    Sync version. Deletes citations -> chunks -> temp doc.
    No commit here; call inside an explicit transaction (e.g., with Session.begin()).
    """
    # Find the temp doc row id
    temp_doc_row_id = db.execute(
        select(temporary_documents_table.c.id)
        .where(temporary_documents_table.c.temp_doc_id == temp_doc_id)
    ).scalar_one_or_none()
    if not temp_doc_row_id:
        logger.warning(f"[sync] Attempted to delete non-existent temp document: {temp_doc_id}")
        return False

    # Collect chunk ids
    chunk_ids = db.execute(
        select(document_chunks_table.c.id)
        .where(document_chunks_table.c.temp_document_id == temp_doc_row_id)
    ).scalars().all()

    if chunk_ids:
        db.execute(
            delete(citations_table)
            .where(citations_table.c.chunk_id.in_(chunk_ids))
        )
        db.execute(
            delete(document_chunks_table)
            .where(document_chunks_table.c.id.in_(chunk_ids))
        )

    db.execute(
        delete(temporary_documents_table)
        .where(temporary_documents_table.c.id == temp_doc_row_id)
    )
    return True


async def get_temp_document_by_user_thread_id(
    db: AsyncSession,
    temp_doc_id: str,
    user_id: int,
    thread_id: str
) -> bool:
    """
    Returns True if a non-cleaned temp‐doc with (temp_doc_id, user_id, thread_id) exists.
    """
    stmt = select(temporary_documents_table.c.id).where(
        temporary_documents_table.c.temp_doc_id == temp_doc_id,
        temporary_documents_table.c.user_id      == user_id,
        temporary_documents_table.c.thread_id    == thread_id,
        temporary_documents_table.c.cleaned_up   == False,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None

async def save_conversation_metrics(
    db: AsyncSession,
    metrics_data: dict
) -> None:
    await db.execute(conversation_metrics_table.insert().values(**metrics_data))
    await db.commit()


async def create_default_roles_and_permissions(db: AsyncSession) -> None:
    """
    Creates predefined roles (admin, editor, viewer) and assigns default permissions.
    This function is idempotent and can be run safely multiple times.
    """
    roles_and_permissions = {
        "admin": [
            "read:workflows", "create:workflows", "update:workflows", "delete:workflows",
            "read:agents", "create:agents", "update:agents", "delete:agents",
            "read:users", "create:users", "update:users", "delete:users",
            "read:roles", "create:roles", "update:roles", "delete:roles",
            "read:api_keys", "create:api_keys", "revoke:api_keys",
            "fine_tune:initiate", "fine_tune:read_status", "fine_tune:list_models",
        ],
        "editor": [
            "read:workflows", "create:workflows", "update:workflows",
            "read:agents", "create:agents", "update:agents",
            "fine_tune:initiate", "fine_tune:read_status", "fine_tune:list_models",
        ],
        "viewer": [
            "read:workflows", "read:agents",
            "fine_tune:read_status", "fine_tune:list_models",
        ]
    }

    for role_name, permissions in roles_and_permissions.items():
        try:
            # Check if the role already exists to prevent IntegrityError
            role = await get_role_by_name(db, role_name)
            if not role:
                role_in = RoleCreate(name=role_name, description=f"{role_name.capitalize()} role with default permissions.")
                role = await create_role(db, role_in)
                logger.info(f"Created new role: {role_name}")

            # Assign permissions, skipping ones that already exist
            existing_permissions = await get_permissions_for_role(db, role["id"])
            for perm in permissions:
                if perm not in existing_permissions:
                    await assign_permission_to_role(db, role["id"], perm)
                    logger.info(f"Assigned permission '{perm}' to role '{role_name}'.")

        except Exception as e:
            logger.error(f"Failed to create or update role '{role_name}': {e}")
            await db.rollback() # Ensure rollback on failure

async def list_temp_documents_for_user_thread(
    db: AsyncSession,
    user_id: int,
    thread_id: str
) -> List[Dict[str, Any]]:
    """
    Returns all non-cleaned temporary documents for the given user/thread.
    """
    stmt = select(temporary_documents_table).where(
        temporary_documents_table.c.user_id == user_id,
        temporary_documents_table.c.thread_id == thread_id,
        temporary_documents_table.c.cleaned_up == False
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(r) for r in rows]


async def create_lambda_run(
    db: AsyncSession,
    user_id: int,
    thread_id: str,
    payload: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a new lambda run record. If run_id is provided (e.g. staged earlier),
    it will be used. Otherwise a new one is generated.
    """
    run_id = run_id or str(uuid4())

    stmt = insert(lambda_runs).values(
        run_id=run_id,
        user_id=user_id,
        thread_id=thread_id,
        initial_request=json.loads(payload),
        start_time=datetime.utcnow(),
        status="pending"
    ).returning(*lambda_runs.c)

    row = (await db.execute(stmt)).mappings().first()
    await db.commit()
    return dict(row)



async def get_lambda_run(db: AsyncSession, run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single Lambda run record and its artifacts."""
    stmt = select(lambda_runs).where(lambda_runs.c.run_id == run_id)
    run_row = (await db.execute(stmt)).mappings().first()
    if not run_row:
        return None

    run_dict = dict(run_row)
    return run_dict


async def update_lambda_run_status(db: AsyncSession, run_id: str, status: str,
                                   final_state: Optional[Dict[str, Any]] = None,
                                   error_message: Optional[str] = None) -> None:
    """Updates the status and final state of a Lambda run."""
    update_data = {"status": status, "end_time": datetime.utcnow()}
    if final_state is not None:
        update_data["final_state"] = final_state
    if error_message is not None:
        update_data["error_message"] = error_message

    stmt = update(lambda_runs).where(lambda_runs.c.run_id == run_id).values(**update_data)
    await db.execute(stmt)
    await db.commit()


def get_temp_docs_metadata(conn, temp_doc_ids: list[str]):
    query = (
        select(
            temporary_documents_table.c.temp_doc_id,
            temporary_documents_table.c.original_name.label("file_name"),
            temporary_documents_table.c.user_id,
            temporary_documents_table.c.thread_id,
        )
        .where(temporary_documents_table.c.temp_doc_id.in_(temp_doc_ids))
    )
    result = conn.execute(query)
    return [dict(row) for row in result]
