from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import get_current_user
from App.repository.llmRegistryRepo import LLM_RegistryRepo
from App.repository.userRepository import UserRepository
from App.repository.systemPromptREP import SystemPromptRepo
from App.core.settings import settings
from App.schemas.llmSchemas import LLMRegister, LLMUpdate
from App.api.dependencies.lcConnector import get_llm_connector, LcConnector
from App.api.databases.Tables import SystemPrompt, UserPersona, RegisterLLM

chain_route = APIRouter(prefix="/chain", tags=["LLM & Chat"])

# Note: LLM management endpoints (register, my-models, enable, disable, delete, etc.)
# are now consolidated in the /v1 router (LlmReg.py)
# This router now only contains chat and persona-related endpoints



@chain_route.post("/admin/load-central/{llm_id}")
async def load_central_model(
    llm_id: int,
    temperature: float = Query(0.7, ge=0.0, le=2.0, description="Controls randomness"),
    top_p: float = Query(0.9, ge=0.0, le=1.0, description="Nucleus sampling"),
    top_k: int = Query(40, ge=1, description="Top-k sampling"),
    max_tokens: int = Query(1024, ge=1, description="Max tokens to generate"),
    n_ctx: int = Query(4096, ge=256, description="Context window size"),
    n_threads: int = Query(6, ge=1, description="CPU threads"),
    n_gpu_layers: int = Query(-1, description="GPU layers (-1 for all)"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin: Load central LLM into memory.
    
    Optional query parameters for model configuration:
    - temperature: Controls randomness (0.0-2.0, default 0.7)
    - top_p: Nucleus sampling (0.0-1.0, default 0.9)
    - top_k: Top-k sampling (default 40)
    - max_tokens: Max tokens to generate (default 1024)
    - n_ctx: Context window size (default 4096)
    - n_threads: CPU threads (default 6)
    - n_gpu_layers: GPU layers (-1 for all, default -1)
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can load central model")
    
    repo = LLM_RegistryRepo(db)
    llm = repo.get_llm_by_id(llm_id, current_user["id"], True)
    
    if not llm:
        raise HTTPException(status_code=404, detail="LLM not found or deleted")
    
    if not llm.get("is_enabled"):
        raise HTTPException(status_code=400, detail="LLM is not enabled. Enable it first.")
    
    # Check if model file exists
    model_path = Path(llm["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {model_path}")
    
    connector = get_llm_connector()
    
    # Check if model is already loaded
    if connector._central_model is not None:
        current_model_name = connector._central_model_name
        if current_model_name == llm["model_name"]:
            return {
                "message": f"Model '{llm['model_name']}' is already loaded",
                "status": "already_loaded",
                "model_path": str(model_path),
                "config": connector._central_model_config
            }
        else:
            # Different model is loaded, proceed with replacement
            logger.info(f"Replacing currently loaded model '{current_model_name}' with '{llm['model_name']}'")
    
    # Configuration parameters
    config_params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_gpu_layers": n_gpu_layers,
    }
    
    loop = asyncio.get_event_loop()
    
    def do_load():
        try:
            # Setting central model and forcing it to load
            success = connector.set_central_model(
                name=llm["model_name"],
                model_path=model_path,
                **config_params
            )
            if not success:
                return False, "Failed to configure central model"
                
            # Actually load the model into memory (synchronous, may take time)
            model = connector.get_central_model()
            if not model:
                return False, "Failed to load model into memory. Check logs."
            
            return True, "Success"
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False, str(e)
            
    # Run heavy loading in a thread executor
    success, message = await loop.run_in_executor(connector.thread_pool, do_load)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    # Build response with all settings
    response_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_gpu_layers": n_gpu_layers,
    }
    
    return {
        "message": f"Central model '{llm['model_name']}' loaded successfully",
        "model_name": llm["model_name"],
        "model_path": str(model_path),
        "config": response_config,
        "status": "success"
    }


@chain_route.post("/admin/unload-central")
async def unload_central_model(
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Unload central LLM."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can unload model")
    
    connector = get_llm_connector()
    
    # Check if model is already unloaded
    if connector._central_model is None:
        return {
            "message": "No central model is currently loaded",
            "status": "already_unloaded"
        }
    
    # Store model name before unloading for the response
    unloaded_model_name = connector._central_model_name or "unknown"
    
    loop = asyncio.get_event_loop()
    
    def do_unload():
        try:
            with connector._model_lock:
                connector._central_model = None
                connector._central_model_name = None
                connector._central_model_config = None
                import gc
                gc.collect()
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            
    # Run garbage collection in thread executor to not block async loop
    await loop.run_in_executor(connector.thread_pool, do_unload)
    
    return {
        "message": f"Central model '{unloaded_model_name}' unloaded",
        "status": "success"
    }


@chain_route.get("/central-model-info")
async def get_central_model_info(
    current_user: Dict = Depends(get_current_user)
):
    """Get central model info - shows if model is loaded in memory."""
    connector = get_llm_connector()
    info = connector.get_central_model_info()
    
    # Check if model is actually loaded in memory
    info["loaded_in_memory"] = connector._central_model is not None
    
    return info


# ==================== PERSONAS ====================

@chain_route.get("/personas")
async def get_personas(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get all available personas (not deleted)."""
    prompts = db.query(SystemPrompt).filter(
        SystemPrompt.is_deleted == False,
        SystemPrompt.is_active == True
    ).all()
    
    return {
        "personas": [
            {"id": p.id, "role": p.role, "prompt": p.prompt[:100] + "..." if len(p.prompt) > 100 else p.prompt}
            for p in prompts
        ]
    }


@chain_route.get("/my-personas")
async def get_my_personas(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get personas assigned to or owned by current user."""
    # Personas assigned via UserPersona
    user_personas = db.query(UserPersona).filter(
        UserPersona.user_id == current_user["id"]
    ).all()
    
    assigned_ids = {up.persona_id for up in user_personas}
    
    # Personas owned by the user directly
    owned_prompts = db.query(SystemPrompt).filter(
        SystemPrompt.user_id == current_user["id"],
        SystemPrompt.is_deleted == False
    ).all()
    
    for p in owned_prompts:
        assigned_ids.add(p.id)
        
    result = []
    for pid in assigned_ids:
        prompt = db.query(SystemPrompt).filter(
            SystemPrompt.id == pid,
            SystemPrompt.is_deleted == False
        ).first()
        if prompt:
            result.append({"id": prompt.id, "role": prompt.role, "prompt": prompt.prompt})
    
    return {"my_personas": result}


# ==================== ADMIN PERSONA MANAGEMENT ====================

@chain_route.post("/admin/create-persona")
async def create_persona(
    role: str = Query(..., description="Role name"),
    prompt: str = Query(..., description="Prompt text"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Create a new persona."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create personas")
    
    new_prompt = SystemPrompt(
        user_id=current_user["id"],
        role=role,
        prompt=prompt,
        is_active=True
    )
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)
    
    return {"message": "Persona created", "persona_id": new_prompt.id}


@chain_route.post("/admin/assign-persona/{persona_id}")
async def assign_persona(
    persona_id: int,
    target_user_id: int = Query(..., description="User ID"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Assign persona to user."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can assign personas")
    
    persona = db.query(SystemPrompt).filter(
        SystemPrompt.id == persona_id,
        SystemPrompt.is_deleted == False
    ).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found or deleted")
    
    existing = db.query(UserPersona).filter(
        UserPersona.user_id == target_user_id,
        UserPersona.persona_id == persona_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Persona already assigned")
    
    assignment = UserPersona(user_id=target_user_id, persona_id=persona_id)
    db.add(assignment)
    db.commit()
    
    return {"message": f"Persona '{persona.role}' assigned to user {target_user_id}"}


@chain_route.delete("/admin/remove-persona/{persona_id}")
async def remove_persona(
    persona_id: int,
    target_user_id: int = Query(..., description="User ID"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Remove persona from user."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can remove personas")
    
    assignment = db.query(UserPersona).filter(
        UserPersona.user_id == target_user_id,
        UserPersona.persona_id == persona_id
    ).first()
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    db.delete(assignment)
    db.commit()
    
    return {"message": "Persona removed"}


# ==================== CHAT ====================

@chain_route.post("/chat")
async def chat(
    persona_id: int = Query(..., description="Persona ID"),
    message: str = Query(..., description="User message"),
    session_id: str = Query(None, description="Session ID"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Chat with the LLM using a persona."""
    persona = db.query(SystemPrompt).filter(
        SystemPrompt.id == persona_id,
        SystemPrompt.is_deleted == False
    ).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found or deleted")
    
    # Check access (Admin, Owner, or Assigned)
    is_admin = str(current_user.get("role", "")).lower() == "admin"
    is_owner = int(persona.user_id) == int(current_user["id"])
    
    user_assignment = db.query(UserPersona).filter(
        UserPersona.user_id == int(current_user["id"]),
        UserPersona.persona_id == int(persona_id)
    ).first()
    
    has_access = is_admin or is_owner or (user_assignment is not None)
        
    if not has_access:
        logger.warning(f"Access denied: user_id={current_user['id']}, role={current_user.get('role')}, persona_id={persona_id}")
        raise HTTPException(
            status_code=403, 
            detail=f"No access to this persona (ID: {persona_id}). Admin must assign it to you."
        )
    
    connector = get_llm_connector()
    
    if not connector._central_model:
        raise HTTPException(status_code=400, detail="No central model loaded. Admin must load a model first.")
    
    if not session_id:
        session_id = await connector.start_conversation(current_user["id"], template_id=str(persona_id))
    
    result = await connector.chat_with_graph(
        message=message,
        user_id=current_user["id"],
        session_id=session_id,
        persona_id=persona_id
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {
        "response": result["content"],
        "session_id": session_id
    }


@chain_route.get("/my-conversations")
async def get_my_conversations(
    current_user: Dict = Depends(get_current_user)
):
    """Get all chat sessions for the current user."""
    connector = get_llm_connector()
    sessions = await connector.get_user_conversations(current_user["id"])
    return {"conversations": sessions}


@chain_route.get("/history")
async def get_formatted_history(
    target_user_id: Optional[int] = Query(None, description="Target User ID (Admin only)"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get full chat history in nested format.
    - User: Gets their own history.
    - Admin: Can get all (target_user_id=None) or specific user's history.
    """
    connector = get_llm_connector()
    is_admin = str(current_user.get("role", "")).lower() == "admin"
    history = await connector.get_formatted_history(
        user_id=current_user["id"],
        is_admin=is_admin,
        target_user_id=target_user_id
    )
    return history


@chain_route.delete("/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Soft delete a chat session (Normal users only their own, Admins any)."""
    connector = get_llm_connector()
    is_admin = str(current_user.get("role", "")).lower() == "admin"
    success = connector.soft_delete_session(session_id, current_user["id"], is_admin)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already deleted")
    return {"message": "Session soft-deleted successfully"}


@chain_route.post("/session/{session_id}/restore")
async def restore_chat_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Restore a soft-deleted session."""
    if str(current_user.get("role", "")).lower() != "admin":
        raise HTTPException(status_code=403, detail="Only admins can restore deleted sessions")
    
    connector = get_llm_connector()
    success = connector.restore_session(session_id, current_user["id"], is_admin=True)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session restored successfully"}


@chain_route.post("/chat/stream")
async def chat_stream(
    persona_id: int = Query(..., description="Persona ID"),
    message: str = Query(..., description="User message"),
    session_id: str = Query(None, description="Session ID"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Chat with streaming response."""
    persona = db.query(SystemPrompt).filter(
        SystemPrompt.id == persona_id,
        SystemPrompt.is_deleted == False
    ).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    # Check access (Admin, Owner, or Assigned)
    is_admin = str(current_user.get("role", "")).lower() == "admin"
    is_owner = int(persona.user_id) == int(current_user["id"])
    
    user_assignment = db.query(UserPersona).filter(
        UserPersona.user_id == int(current_user["id"]),
        UserPersona.persona_id == int(persona_id)
    ).first()
    
    has_access = is_admin or is_owner or (user_assignment is not None)
        
    if not has_access:
        logger.warning(f"Stream Access denied: user_id={current_user['id']}, role={current_user.get('role')}, persona_id={persona_id}")
        raise HTTPException(
            status_code=403, 
            detail=f"No access to this persona (ID: {persona_id}). Admin must assign it to you."
        )
    
    connector = get_llm_connector()
    
    if not connector._central_model:
        raise HTTPException(status_code=400, detail="No central model loaded")
        
    # Enforce session ownership if provided
    if session_id:
        if not connector.validate_session_ownership(session_id, current_user["id"]):
            raise HTTPException(
                status_code=403, 
                detail="Unauthorized: This session does not belong to you or has been deleted."
            )
    
    async def generate():
        nonlocal session_id
        if not session_id:
            session_id = await connector.start_conversation(current_user["id"], template_id=str(persona_id))
        
        try:
            from App.api.dependencies.graph_engine import graph_engine
            full_response = ""
            async for content in graph_engine.astream(session_id, current_user["id"], message, persona_id):
                full_response += content
                yield f"data: {json.dumps({'content': content, 'session_id': session_id})}\n\n"
            
            # Post-stream persistence (since we bypassed the graph 'save' node)
            await connector.add_message(session_id, "user", message)
            await connector.add_message(session_id, "assistant", full_response)
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
