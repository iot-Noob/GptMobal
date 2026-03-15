from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import asyncio

from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import get_current_user
from App.repository.systemPromptREP import SystemPromptRepo
from App.schemas.systemPromptSchemas import SystemPromptCreate, SystemPromptUpdate
from App.api.dependencies.lcConnector import get_llm_connector

router = APIRouter(prefix="/prompt" )


@router.post("/create")
async def create_prompt(
    role: str = Query(..., description="Role name for this prompt (e.g., 'Analyst', 'Assistant')"),
    prompt: str = Query(..., description="The system prompt text"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new system prompt. Admin only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create prompts")
    
    repo = SystemPromptRepo(db)
    result = repo.create_prompt(
        user_id=current_user["id"],
        role=role,
        prompt=prompt
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.get("/my-prompts")
async def get_my_prompts(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get all prompts for current user.
    """
    repo = SystemPromptRepo(db)
    is_admin = current_user.get("role") == "admin"
    prompts = repo.get_prompts(current_user["id"], is_admin)
    return {"prompts": prompts}


@router.get("/details/{prompt_id}")
async def get_prompt_details(
    prompt_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get details of a specific prompt.
    """
    repo = SystemPromptRepo(db)
    is_admin = current_user.get("role") == "admin"
    
    prompt = repo.get_prompt_by_id(prompt_id, current_user["id"], is_admin)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found or unauthorized")
    
    return prompt


@router.patch("/update/{prompt_id}")
async def update_prompt(
    prompt_id: int,
    role: str = Query(None, description="New role name"),
    prompt: str = Query(None, description="New prompt text"),
    is_active: bool = Query(None, description="Set active status"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Update a prompt. Admin only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can update prompts")
    
    repo = SystemPromptRepo(db)
    is_admin = True  # Always admin here
    
    # Build update dict (exclude None values)
    update_dict = {}
    if role is not None:
        update_dict["role"] = role
    if prompt is not None:
        update_dict["prompt"] = prompt
    if is_active is not None:
        update_dict["is_active"] = is_active
    
    if not update_dict:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    result = repo.update_prompt(prompt_id, current_user["id"], is_admin, update_dict)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.delete("/delete/{prompt_id}")
async def delete_prompt(
    prompt_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Soft delete a prompt. Admin only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete prompts")
    
    repo = SystemPromptRepo(db)
    is_admin = True  # Always admin here
    
    result = repo.soft_delete_prompt(prompt_id, current_user["id"], is_admin)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


# ==================== ADMIN ONLY ENDPOINTS ====================

@router.post("/admin/assign-to-user")
async def assign_prompt_to_user(
    prompt_id: int = Query(..., description="ID of the prompt"),
    target_user_id: int = Query(..., description="ID of the user to assign to"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Assign prompt to a different user"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can assign prompts")
    
    repo = SystemPromptRepo(db)
    result = repo.assign_prompt_to_user(prompt_id, target_user_id, current_user["id"])
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.post("/admin/restore/{prompt_id}")
async def restore_prompt(
    prompt_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Restore a deleted prompt"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can restore prompts")
    
    repo = SystemPromptRepo(db)
    result = repo.restore_prompt(prompt_id, is_admin=True)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.get("/admin/deleted-prompts")
async def get_deleted_prompts(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Get all deleted prompts"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this")
    
    repo = SystemPromptRepo(db)
    prompts = repo.get_deleted_prompts(is_admin=True)
    return {"deleted_prompts": prompts}


@router.get("/admin/all-prompts")
async def get_all_prompts_admin(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Get all prompts in system"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this")
    
    repo = SystemPromptRepo(db)
    prompts = repo.get_prompts(current_user["id"], is_admin=True)
    return {"prompts": prompts}


@router.get("/admin/users-with-prompts")
async def get_users_with_prompts(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Get all users with their assigned prompts"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this")
    
    repo = SystemPromptRepo(db)
    result = repo.get_all_users_with_prompts(is_admin=True)
    return {"users_with_prompts": result}
 