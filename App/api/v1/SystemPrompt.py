from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import get_current_user
from App.repository.systemPromptREP import SystemPromptRepo
from App.schemas.systemPromptSchemas import SystemPromptCreate, SystemPromptUpdate

router = APIRouter(prefix="/prompt", tags=["SystemPrompt"])


@router.post("/create")
async def create_prompt(
    prompt_data: SystemPromptCreate,
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
        model_id=prompt_data.model_id,
        persona_name=prompt_data.persona_name,
        prompt_text=prompt_data.prompt_text
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
    update_data: SystemPromptUpdate,
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
    if update_data.persona_name is not None:
        update_dict["persona_name"] = update_data.persona_name
    if update_data.prompt_text is not None:
        update_dict["prompt_text"] = update_data.prompt_text
    if update_data.is_active is not None:
        update_dict["is_active"] = update_data.is_active
    
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


@router.post("/admin/assign-to-model")
async def assign_prompt_to_model(
    prompt_id: int = Query(..., description="ID of the prompt"),
    model_id: int = Query(..., description="ID of the model to assign to"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Assign prompt to a different model"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can reassign prompts")
    
    repo = SystemPromptRepo(db)
    result = repo.assign_prompt_to_model(prompt_id, model_id, current_user["id"])
    
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
