from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import   *
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Any, List, Union
import gc
import ctypes
import asyncio
import json
from datetime import datetime
import hashlib
import uuid
import os
import logging
from sqlalchemy import or_

from App.core.LoggingInit import get_core_logger
from App.core.settings import settings
from App.api.dependencies.sqlite_connector import SessionLocal
from App.api.databases.Tables import ChatSession, SystemPrompt
# from App.api.dependencies.graph_engine import graph_engine # Avoid circular import

logger = get_core_logger(__name__)


class LcConnector:
    """
    Central LLM Connector optimized for FastAPI with single model architecture.
    
    Features:
    - Thread-safe singleton pattern
    - Single central LLM (admin-controlled)
    - Prompt template management
    - Conversation history tracking with SQLite persistence
    - User preferences
    - Token counting and usage tracking
    """
    
    _instance: Optional['LcConnector'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Central model (only ONE active at a time)
        self._central_model: Optional[ChatLlamaCpp] = None
        self._central_model_config: Optional[Dict[str, Any]] = None
        self._central_model_name: Optional[str] = None
        self._model_lock = Lock()  # For thread-safe model inference
        
        # Prompt templates (can be per-user or system-wide)
        self.prompt_templates: Dict[str, Dict[str, Any]] = {}
        
        # User preferences cache (user_id -> preferred_prompt, temperature, etc.)
        self.user_preferences: Dict[int, Dict[str, Any]] = {}
        
        # SQLite connection for chat history
        self._setup_database()
        
        # Session store for tracking which SQL store belongs to which session
        self._session_stores: Dict[str, SQLChatMessageHistory] = {}
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "requests_by_user": {},
            "requests_by_prompt": {}
        }
        
        # Thread pool for async operations
        self._thread_pool = None
        
        logger.info(f"✅ LcConnector initialized with SQLite database: {self.db_path}")
    
    def _setup_database(self):
        """Setup SQLite database connection."""
        try:
            # Parse SQLite URL from settings
            # Assuming SQLITE_DATABASE_URL is like "sqlite:///./chat_history.db"
            db_url = settings.SQLITE_DATABASE_URL
            
            # Extract path for logging
            if db_url.startswith("sqlite:///"):
                self.db_path = db_url.replace("sqlite:///", "")
            else:
                self.db_path = "chat_history.db"
            
            # Ensure directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Test connection by creating a temporary store
            test_session_id = f"test_{uuid.uuid4()}"
            test_store = SQLChatMessageHistory(
                session_id=test_session_id,
                connection_string=db_url
            )
            test_store.clear()  # Clean up test data
            
            logger.info(f"✅ SQLite database connected: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            # Fallback to in-memory if database fails
            self.db_path = ":memory:"
            logger.warning("Falling back to in-memory database")
    
    def _get_session_store(self, session_id: str) -> SQLChatMessageHistory:
        """
        Get or create a SQLChatMessageHistory for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            SQLChatMessageHistory instance
        """
        if session_id not in self._session_stores:
            try:
                self._session_stores[session_id] = SQLChatMessageHistory(
                    session_id=session_id,
                    connection_string=settings.SQLITE_DATABASE_URL
                )
                logger.debug(f"Created new session store: {session_id}")
            except Exception as e:
                logger.error(f"Failed to create session store: {e}")
                # Return None or raise - depending on your error handling
                raise
        
        return self._session_stores[session_id]
    
    def validate_session_ownership(self, session_id: str, user_id: int) -> bool:
        """Verify if a session belongs to a user."""
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
                ChatSession.is_deleted == False
            ).first()
            return session is not None
        finally:
            db.close()
    
    @property
    def thread_pool(self):
        """Lazy initialize thread pool."""
        if self._thread_pool is None:
            import concurrent.futures
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="llm_inference"
            )
        return self._thread_pool
    
    # ==================== CENTRAL MODEL MANAGEMENT ====================
    
    def set_central_model(self, name: str, model_path: Path, **kwargs) -> bool:
        """
        Set the central LLM (admin function).
        
        This replaces the current central model with a new one.
        Only one central model exists at a time.
        
        Args:
            name: Model identifier (e.g., "llama2-7b")
            model_path: Path to model file
            **kwargs: Model configuration parameters
            
        Returns:
            bool: True if model was set successfully
        """
        try:
            with self._model_lock:
                # Clear existing model if any
                if self._central_model:
                    self._central_model = None
                    gc.collect()
                
                # Store configuration
                self._central_model_config = {
                    "name": name,
                    "model_path": model_path,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 40),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "n_ctx": kwargs.get("n_ctx", 4096),
                    "n_gpu_layers": kwargs.get("n_gpu_layers", -1),
                    "n_threads": kwargs.get("n_threads", 6),
                    "stop": kwargs.get("stop", ["<|endoftext|>", "Human:", "User:"]),
                }
                
                self._central_model_name = name
                
                # Model will be lazy-loaded when first used
                logger.info(f"✅ Central model set to '{name}'")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to set central model: {e}")
            return False
    
    def _load_central_model(self) -> Optional[ChatLlamaCpp]:
        """Lazy-load the central model."""
        if self._central_model is not None:
            return self._central_model
        
        if not self._central_model_config:
            logger.error("No central model configured")
            return None
        
        try:
            config = self._central_model_config.copy()
            model_path = config.pop("model_path")
            
            self._central_model = ChatLlamaCpp(
                model_path=str(model_path),
                **config
            )
            
            logger.info(f"✅ Central model '{self._central_model_name}' loaded")
            return self._central_model
            
        except Exception as e:
            logger.error(f"❌ Failed to load central model: {e}")
            return None
    
    def get_central_model(self) -> Optional[ChatLlamaCpp]:
        """Get the central model (lazy loads if needed)."""
        return self._load_central_model()
    
    def get_central_model_info(self) -> Dict[str, Any]:
        """Get information about the current central model."""
        if not self._central_model_config:
            return {"status": "No model configured"}
        
        return {
            "name": self._central_model_config["name"],
            "loaded": self._central_model is not None,
            "config": {k: v for k, v in self._central_model_config.items() 
                      if k != "model_path"}  # Hide full path for security
        }
    
    # ==================== PROMPT TEMPLATE MANAGEMENT ====================
    
    def add_prompt_template(self, template_id: str, content: str, 
                           role: str = "system", description: str = "") -> bool:
        """
        Add a prompt template (admin function).
        
        Args:
            template_id: Unique identifier for the template
            content: The prompt text
            role: Message role (system, user, assistant)
            description: Description of what this template is for
            
        Returns:
            bool: True if added successfully
        """
        try:
            self.prompt_templates[template_id] = {
                "id": template_id,
                "role": role,
                "content": content,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
            logger.info(f"✅ Prompt template '{template_id}' added")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add prompt template: {e}")
            return False
    
    def update_prompt_template(self, template_id: str, **updates) -> bool:
        """Update an existing prompt template."""
        if template_id not in self.prompt_templates:
            logger.warning(f"Template '{template_id}' not found")
            return False
        
        for key, value in updates.items():
            if key in self.prompt_templates[template_id]:
                self.prompt_templates[template_id][key] = value
        
        self.prompt_templates[template_id]["updated_at"] = datetime.now().isoformat()
        logger.info(f"✅ Prompt template '{template_id}' updated")
        return True
    
    def delete_prompt_template(self, template_id: str) -> bool:
        """Delete a prompt template."""
        if template_id in self.prompt_templates:
            del self.prompt_templates[template_id]
            logger.info(f"✅ Prompt template '{template_id}' deleted")
            return True
        return False
    
    def list_prompt_templates(self) -> Dict[str, Any]:
        """List all available prompt templates."""
        return self.prompt_templates
    
    def get_prompt_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt template."""
        return self.prompt_templates.get(template_id)
    
    # ==================== USER PREFERENCES ====================
    
    def set_user_preference(self, user_id: int, template_id: str = None,
                           temperature: float = None, max_tokens: int = None,
                           **custom_params) -> bool:
        """
        Set preferences for a specific user.
        
        Args:
            user_id: User ID
            template_id: Preferred prompt template
            temperature: Preferred temperature
            max_tokens: Preferred max tokens
            **custom_params: Any other custom parameters
        """
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat()
                }
            
            prefs = self.user_preferences[user_id]
            
            if template_id is not None:
                prefs["template_id"] = template_id
            if temperature is not None:
                prefs["temperature"] = temperature
            if max_tokens is not None:
                prefs["max_tokens"] = max_tokens
            
            prefs.update(custom_params)
            prefs["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"✅ Preferences set for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set user preferences: {e}")
            return False
    
    def get_user_preference(self, user_id: int, key: str, default=None):
        """Get a specific user preference."""
        if user_id in self.user_preferences:
            return self.user_preferences[user_id].get(key, default)
        return default
    
    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get all preferences for a user."""
        return self.user_preferences.get(user_id, {})
    
    # ==================== CONVERSATION HISTORY ====================
    
    async def start_conversation(self, user_id: int, template_id: str = None) -> str:
        """
        Start a new conversation session and persist ownership mapping.
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        
        def _persist():
            # Get user's preferred template if not provided
            nonlocal template_id
            if template_id is None:
                template_id = self.get_user_preference(user_id, "template_id")
            
            db = SessionLocal()
            try:
                p_id = None
                if template_id and str(template_id).isdigit():
                    p_id = int(template_id)
                
                new_session = ChatSession(
                    session_id=session_id,
                    user_id=user_id,
                    persona_id=p_id
                )
                db.add(new_session)
                db.commit()
                logger.info(f"✅ Persisted session {session_id} for user {user_id}")
            except Exception as e:
                logger.error(f"❌ Failed to persist session mapping: {e}")
                db.rollback()
            finally:
                db.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _persist)

        # Get or create session store (sync initialization, but DB access later is async)
        store = self._get_session_store(session_id)
        
        # Add system prompt if template exists
        if template_id and template_id in self.prompt_templates:
            template = self.prompt_templates[template_id]
            await self.add_message(session_id, "system", template["content"])
            
            # Track template usage
            self.prompt_templates[template_id]["usage_count"] = \
                self.prompt_templates[template_id].get("usage_count", 0) + 1
        
        # Store session metadata (you might want to create a separate table for this)
        session_metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "template_id": template_id,
            "created_at": datetime.now().isoformat()
        }
        
        # You could store this in a separate SQLite table or a JSON file
        # For now, we'll keep it in memory
        if not hasattr(self, '_session_metadata'):
            self._session_metadata = {}
        self._session_metadata[session_id] = session_metadata
        
        logger.info(f"✅ New conversation started: {session_id} for user {user_id}")
        return session_id
    
    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a message to conversation history (Async).
        """
        def _add():
            try:
                store = self._get_session_store(session_id)
                if role == "user":
                    store.add_message(HumanMessage(content=content))
                elif role == "assistant":
                    store.add_message(AIMessage(content=content))
                elif role == "system":
                    store.add_message(SystemMessage(content=content))
                else:
                    return False
                return True
            except Exception as e:
                logger.error(f"Failed to add message: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _add)
    
    async def get_conversation_history(self, session_id: str, as_dict: bool = False, limit: Optional[int] = None) -> Optional[List]:
        """
        Get conversation history for a session (Async).
        """
        def _get():
            try:
                store = self._get_session_store(session_id)
                messages = store.messages
                
                if limit:
                    if messages and isinstance(messages[0], SystemMessage):
                        messages = [messages[0]] + messages[-(limit-1):] if limit > 1 else [messages[0]]
                    else:
                        messages = messages[-limit:]
                return messages
            except Exception as e:
                logger.error(f"Failed to get conversation history: {e}")
                return None

        loop = asyncio.get_event_loop()
        messages = await loop.run_in_executor(self.thread_pool, _get)
        
        if messages and as_dict:
            return [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else
                            "assistant" if isinstance(msg, AIMessage) else
                            "system",
                    "content": msg.content,
                    "type": msg.type
                }
                for msg in messages
            ]
        return messages
    
    def clear_conversation_history(self, session_id: str = None):
        """
        Clear conversation history for a session or all sessions.
        
        Args:
            session_id: Session ID to clear, or None to clear all
        """
        try:
            if session_id:
                if session_id in self._session_stores:
                    store = self._session_stores[session_id]
                    store.clear()
                    del self._session_stores[session_id]
                    
                    # Clear metadata if exists
                    if hasattr(self, '_session_metadata') and session_id in self._session_metadata:
                        del self._session_metadata[session_id]
                    
                    logger.info(f"✅ Conversation {session_id} cleared")
            else:
                # Clear all sessions
                for store in self._session_stores.values():
                    store.clear()
                self._session_stores.clear()
                if hasattr(self, '_session_metadata'):
                    self._session_metadata.clear()
                logger.info("✅ All conversation history cleared")
                
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
    
    async def get_user_conversations(self, user_id: int, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user from database (Async).
        """
        def _get():
            db = SessionLocal()
            try:
                query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
                if not include_deleted:
                    query = query.filter(ChatSession.is_deleted == False)
                
                sessions = query.all()
                return [
                    {
                        "session_id": s.session_id,
                        "persona_id": s.persona_id,
                        "is_deleted": s.is_deleted,
                        "created_at": s.created_at.isoformat() if s.created_at else None
                    }
                    for s in sessions
                ]
            except Exception as e:
                logger.error(f"Error fetching user conversations: {e}")
                return []
            finally:
                db.close()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _get)

    async def get_formatted_history(self, user_id: int, is_admin: bool = False, target_user_id: Optional[int] = None) -> Dict:
        """
        Get chat history in the requested nested format (Async).
        """
        def _get_sessions():
            db = SessionLocal()
            try:
                if is_admin and target_user_id:
                    return db.query(ChatSession).filter(ChatSession.user_id == target_user_id, ChatSession.is_deleted == False).all()
                elif is_admin:
                    return db.query(ChatSession).filter(ChatSession.is_deleted == False).all()
                else:
                    return db.query(ChatSession).filter(ChatSession.user_id == user_id, ChatSession.is_deleted == False).all()
            finally:
                db.close()

        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(self.thread_pool, _get_sessions)

        result = {}
        for s in sessions:
            history = await self.get_conversation_history(s.session_id, as_dict=True)
            session_data = {f"msg_{idx}": msg for idx, msg in enumerate(history)}
            
            if is_admin and not target_user_id:
                u_id = str(s.user_id)
                if u_id not in result: result[u_id] = {}
                result[u_id][s.session_id] = session_data
            else:
                result[s.session_id] = session_data
        
        return result

    def soft_delete_session(self, session_id: str, user_id: int, is_admin: bool = False) -> bool:
        """Soft delete a chat session."""
        db = SessionLocal()
        try:
            query = db.query(ChatSession).filter(ChatSession.session_id == session_id)
            if not is_admin:
                query = query.filter(ChatSession.user_id == user_id)
            
            session = query.first()
            if session:
                session.is_deleted = True
                session.deleted_at = datetime.now()
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Soft delete error: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def restore_session(self, session_id: str, user_id: int, is_admin: bool = False) -> bool:
        """Restore a soft-deleted chat session."""
        db = SessionLocal()
        try:
            query = db.query(ChatSession).filter(ChatSession.session_id == session_id)
            if not is_admin:
                query = query.filter(ChatSession.user_id == user_id)
            
            session = query.first()
            if session:
                session.is_deleted = False
                session.deleted_at = None
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Restore session error: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    # ==================== CHAT COMPLETION ====================
    
    async def chat(self, 
                   messages: List[Dict[str, str]],
                   user_id: Optional[int] = None,
                   session_id: Optional[str] = None,
                   template_id: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   stream: bool = False,
                   save_history: bool = True,
                   **kwargs) -> Union[Dict[str, Any], Any]:
        """
        Main chat method - handles everything with database persistence.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            user_id: User ID for preferences and tracking
            session_id: Session ID for conversation continuity
            template_id: Override prompt template
            temperature: Override temperature
            max_tokens: Override max tokens
            stream: Whether to stream the response
            save_history: Whether to save messages to database
            **kwargs: Additional parameters
            
        Returns:
            Response dict or stream generator
        """
        # 1. Get central model
        model = self.get_central_model()
        if not model:
            return {"error": "No central model configured"}
        
        # 2. Apply user preferences if available
        if user_id and user_id in self.user_preferences:
            prefs = self.user_preferences[user_id]
            if template_id is None:
                template_id = prefs.get("template_id")
            if temperature is None:
                temperature = prefs.get("temperature")
            if max_tokens is None:
                max_tokens = prefs.get("max_tokens")
        
        # 3. Build message list for inference
        inference_messages = []
        
        # Add system prompt from template if available
        if template_id and template_id in self.prompt_templates:
            template = self.prompt_templates[template_id]
            inference_messages.append({
                "role": template["role"],
                "content": template["content"]
            })
            # Track template usage
            self.prompt_templates[template_id]["usage_count"] = \
                self.prompt_templates[template_id].get("usage_count", 0) + 1
        
        # Add conversation history if session exists
        if session_id:
            # Enforce ownership check
            if user_id and not self.validate_session_ownership(session_id, user_id):
                return {"error": "Unauthorized: This session does not belong to you or has been deleted."}

            if save_history:
                history = self.get_conversation_history(session_id, as_dict=True)
                if history:
                    # Filter out system messages if we're adding a new one
                    if template_id:
                        history = [m for m in history if m["role"] != "system"]
                    inference_messages.extend(history)
        
        # Add current messages
        inference_messages.extend(messages)
        
        # 4. Prepare parameters
        params = {
            "temperature": temperature or self._central_model_config.get("temperature", 0.7),
            "max_tokens": max_tokens or self._central_model_config.get("max_tokens", 1024),
        }
        params.update(kwargs)
        
        # 5. Track usage
        self.usage_stats["total_requests"] += 1
        if user_id:
            self.usage_stats["requests_by_user"][user_id] = \
                self.usage_stats["requests_by_user"].get(user_id, 0) + 1
        
        # 6. Generate response
        try:
            if stream:
                return self._stream_response(model, inference_messages, session_id, user_id, save_history, **params)
            else:
                return await self._generate_response(model, inference_messages, session_id, user_id, save_history, **params)
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": str(e)}

    async def chat_with_graph(self, 
                             message: str,
                             user_id: int,
                             session_id: str,
                             persona_id: Optional[int] = None,
                             **kwargs):
        """Execute chat via LangGraph orchestration."""
        from App.api.dependencies.graph_engine import graph_engine
        try:
            response = await graph_engine.run(session_id, user_id, message, persona_id)
            return {"content": response}
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            return {"error": str(e)}
    
    async def _generate_response(self, model, messages, session_id, user_id, save_history, **params):
        """Generate non-streaming response and save to database."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool
        response = await loop.run_in_executor(
            self.thread_pool,
            self._run_inference,
            model,
            messages,
            params
        )
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Save to database if session exists
        if session_id and save_history:
            # Save user messages (only the new ones)
            for msg in messages:
                if msg["role"] == "user":
                    self.add_message(session_id, "user", msg["content"])
            
            # Save assistant response
            self.add_message(session_id, "assistant", response_content)
        
        # Track token usage (estimate)
        input_tokens = self._estimate_tokens(str(messages))
        output_tokens = self._estimate_tokens(response_content)
        self.usage_stats["total_tokens"] += input_tokens + output_tokens
        
        return {
            "content": response_content,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "model": self._central_model_name,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_inference(self, model, messages, params):
        """Run inference in thread pool."""
        with self._model_lock:  # Ensure thread safety for model
            return model.invoke(messages, **params)
    
    async def _stream_response(self, model, messages, session_id, user_id, save_history, **params):
        """Stream response and save to database."""
        # For streaming, you'd need to collect chunks and save at the end
        # This is a placeholder - implement based on your needs
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
    
    # ==================== UTILITY METHODS ====================
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_stats
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            stats = {
                "database_path": self.db_path,
                "tables": [],
                "total_sessions": len(self._session_stores),
                "active_sessions": len(self._session_stores)
            }
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                stats["tables"].append({
                    "name": table_name,
                    "row_count": count
                })
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear in-memory caches only (not database)."""
        self.user_preferences.clear()
        # Don't clear session stores as they're tied to database
        logger.info("✅ In-memory caches cleared")
    
    def cleanup(self):
        """Clean up all resources."""
        # Shutdown thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        # Clear model
        self._central_model = None
        
        # Clear caches
        self.clear_cache()
        self.prompt_templates.clear()
        
        # Close all session stores
        self._session_stores.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ LcConnector cleanup complete")


# Global singleton instance
lc_connector = LcConnector()


# FastAPI dependency
def get_llm_connector() -> LcConnector:
    return lc_connector