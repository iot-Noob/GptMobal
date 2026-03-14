import os
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json

from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import get_current_user
from App.repository.llmRegistryRepo import LLM_RegistryRepo
from App.repository.userRepository import UserRepository
from App.core.settings import settings
from App.schemas.llmSchemas import LLMRegister, LLMUpdate
from App.api.dependencies.lcConnector import get_llm_connector, LcConnector

llm_router = APIRouter()


@llm_router.post("/register")
async def register_new_llm(
    model_name: str = Query(..., description="Custom name for the LLM"),
    model_filename: str = Query(..., description="Filename of the model in the models directory"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Register a new LLM with custom name. Admin only.
    - User provides custom name and model filename
    - System validates the file exists in MODELS_PATH
    - System auto-calculates file size
    - System checks for duplicate names
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can register LLMs")
    try:
        repo = LLM_RegistryRepo(db)
        
        # Check for duplicate model name
        duplicate_check = repo.check_duplicate_model_name(model_name)
        if duplicate_check["is_duplicate"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Model with name '{model_name}' already exists. Please use a different name."
            )
        
        # Get the base models path from settings
        base_models_path = settings.MODELS_PATH
        
        # Construct the full path
        model_path = base_models_path / model_filename
        
        # Validate the model path is within the allowed directory
        path_validation = repo.validate_model_path_within_base(model_path, base_models_path)
        if not path_validation["is_valid"]:
            raise HTTPException(status_code=400, detail=path_validation["message"])
        
        # Get file size from validation result
        file_size = path_validation.get("file_size_bytes")
        
        # Register the LLM in DB
        from App.api.databases.Tables import RegisterLLM
        new_llm = RegisterLLM(
            model_name=model_name,
            model_path=str(model_path),
            is_active=False  # Not active by default - must be activated manually
        )
        repo.db.add(new_llm)
        repo.db.commit()
        repo.db.refresh(new_llm)

        return {
            "message": "Model registered successfully",
            "llm_id": new_llm.id,
            "model_name": model_name,
            "model_filename": model_filename,
            "model_path": str(model_path),
            "file_size_bytes": file_size
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@llm_router.get("/my-models")
async def list_user_models(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get LLM models for the current user.
    Normal users see only their assigned LLMs.
    Admins see all LLMs organized by user.
    """
    repo = LLM_RegistryRepo(db)
    is_admin = current_user.get("role") == "admin"
    return repo.get_llms(user_id=current_user["id"], is_admin=is_admin)


@llm_router.get("/details/{llm_id}")
async def get_llm_details(
    llm_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get details of a specific LLM.
    Normal users can only view their own LLMs.
    Admins can view any LLM.
    """
    repo = LLM_RegistryRepo(db)
    is_admin = current_user.get("role") == "admin"
    
    llm = repo.get_llm_by_id(llm_id, current_user["id"], is_admin)
    if not llm:
        raise HTTPException(status_code=404, detail="LLM not found or unauthorized")
    
    # Get file size
    model_path = Path(llm["model_path"])
    file_size = repo.get_file_size(model_path)
    llm["file_size_bytes"] = file_size
    
    return llm


@llm_router.patch("/update/{llm_id}")
async def update_model_info(
    llm_id: int,
    update_data: LLMUpdate,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Update LLM information. Admin only.
    Can also set/unset global status.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can update LLM information")
    
    repo = LLM_RegistryRepo(db)
    
    # Build update dict from pydantic model (exclude None values)
    update_dict = {}
    if update_data.model_name is not None:
        # Check for duplicate name
        duplicate_check = repo.check_duplicate_model_name(update_data.model_name)
        if duplicate_check["is_duplicate"]:
            # Allow if it's the same LLM being updated
            existing_llm = repo.get_llm_by_id(llm_id, current_user["id"], True)
            if existing_llm and existing_llm["model_name"] != update_data.model_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model with name '{update_data.model_name}' already exists"
                )
        update_dict["model_name"] = update_data.model_name
    
    if update_data.model_filename is not None:
        # Validate new file path
        base_models_path = settings.MODELS_PATH
        new_path = base_models_path / update_data.model_filename
        path_validation = repo.validate_model_path_within_base(new_path, base_models_path)
        if not path_validation["is_valid"]:
            raise HTTPException(status_code=400, detail=path_validation["message"])
        update_dict["model_path"] = str(new_path)
    
    if update_data.vram_estimate_gb is not None:
        update_dict["vram_estimate_gb"] = update_data.vram_estimate_gb
    
    if update_data.is_active is not None:
        update_dict["is_active"] = update_data.is_active
    
    if not update_dict:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    success = repo.update_llm(
        llm_id=llm_id, 
        user_id=current_user["id"], 
        is_admin=True,  # Always admin here
        update_data=update_dict
    )
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": "Update successful"}


@llm_router.post("/admin/activate/{llm_id}")
async def activate_llm_for_all(
    llm_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Activate an LLM so all users can use it.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can activate LLMs")
    
    repo = LLM_RegistryRepo(db)
    result = repo.set_llm_active(llm_id, current_user["id"], True, True)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@llm_router.post("/admin/deactivate/{llm_id}")
async def deactivate_llm(
    llm_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Deactivate an LLM so no users can use it.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can deactivate LLMs")
    
    repo = LLM_RegistryRepo(db)
    result = repo.set_llm_active(llm_id, current_user["id"], True, False)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@llm_router.delete("/delete/{llm_id}")
async def delete_model(
    llm_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Soft delete an LLM (move to trash). Admin only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete LLMs")
    
    repo = LLM_RegistryRepo(db)
    
    result = repo.soft_delete_llm(llm_id, current_user["id"], True)  # Always admin here
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# @llm_router.post("/activate/{llm_id}")
# async def activate_llm(
#     llm_id: int,
#     db: Session = Depends(get_db),
#     current_user: Dict = Depends(get_current_user)
# ):
#     """
#     Activate an LLM (mark as live/loaded).
#     Users can only activate their own LLMs.
#     """
#     repo = LLM_RegistryRepo(db)
#     is_admin = current_user.get("role") == "admin"
    
#     result = repo.set_llm_active(llm_id, current_user["id"], is_admin, True)
#     if not result["success"]:
#         raise HTTPException(status_code=400, detail=result["message"])
#     return result


# @llm_router.post("/deactivate/{llm_id}")
# async def deactivate_llm(
#     llm_id: int,
#     db: Session = Depends(get_db),
#     current_user: Dict = Depends(get_current_user)
# ):
#     """
#     Deactivate an LLM (mark as not live/unloaded).
#     Users can only deactivate their own LLMs.
#     """
#     repo = LLM_RegistryRepo(db)
#     is_admin = current_user.get("role") == "admin"
    
#     result = repo.set_llm_active(llm_id, current_user["id"], is_admin, False)
#     if not result["success"]:
#         raise HTTPException(status_code=400, detail=result["message"])
#     return result


# ==================== ADMIN ONLY ENDPOINTS ====================

@llm_router.post("/admin/assign-to-user")
async def assign_llm_to_user(
    llm_id: int = Query(..., description="ID of the LLM to assign"),
    target_user_id: int = Query(..., description="ID of the user to assign the LLM to"),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Assign an LLM to another user.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can assign LLMs to other users")
    
    repo = LLM_RegistryRepo(db)
    result = repo.assign_llm_to_user(llm_id, target_user_id, current_user["id"])
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@llm_router.get("/admin/all-llms")
async def get_all_llms_admin(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Get all LLMs in the system (including deleted).
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this resource")
    
    repo = LLM_RegistryRepo(db)
    llms = repo.get_all_llms_for_admin(current_user["id"])
    return {"llms": llms}


@llm_router.get("/admin/deleted-llms")
async def get_deleted_llms(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Get all deleted LLMs (trash bin).
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this resource")
    
    repo = LLM_RegistryRepo(db)
    llms = repo.get_deleted_llms(is_admin=True)
    
    if llms is None:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "deleted_llms": [
            {
                "id": llm.id,
                "user_id": llm.user_id,
                "model_name": llm.model_name,
                "model_path": llm.model_path,
                "is_deleted": llm.is_deleted,
                "created_at": llm.created_at.isoformat() if llm.created_at else None
            }
            for llm in llms
        ]
    }


@llm_router.post("/admin/restore/{llm_id}")
async def restore_llm(
    llm_id: int,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Restore a deleted LLM from trash.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can restore LLMs")
    
    repo = LLM_RegistryRepo(db)
    result = repo.restore_llm(llm_id, is_admin=True)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@llm_router.get("/admin/all-users")
async def get_all_users_for_assignment(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Admin only: Get all users (for LLM assignment).
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this resource")
    
    user_repo = UserRepository(db)
    users = user_repo.get_all_users()
    
    return {
        "users": [
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_admin": user.is_admin,
                "is_active": user.is_active
            }
            for user in users
        ]
    }


@llm_router.get("/admin/models-directory")
async def get_models_directory(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the configured models directory path.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this resource")
    
    return {
        "models_directory": str(settings.MODELS_PATH),
        "exists": settings.MODELS_PATH.exists(),
        "is_directory": settings.MODELS_PATH.is_dir() if settings.MODELS_PATH.exists() else False
    }


@llm_router.get("/available-models")
async def get_available_models(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get all model files available in the models directory.
    Returns filename, full path, and size for each model.
    """
    repo = LLM_RegistryRepo(db=None)
    models = repo.get_models_in_directory(settings.MODELS_PATH)
    
    return {
        "models_directory": str(settings.MODELS_PATH),
        "total_models": len(models),
        "models": models
    }


# ==================== CENTRAL MODEL MANAGEMENT ====================
# Note: Central model management (load/unload/info) is handled in langChainsRoutes.py


@llm_router.get("/enabled")
async def get_enabled_llms(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get all enabled LLMs (not deleted)."""
    repo = LLM_RegistryRepo(db)
    llms = repo.get_all_enabled_llms()
    return {"enabled_llms": llms}

@llm_router.get("/admin/llms-with-owners")
async def get_llms_with_owners(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Admin: Get all LLMs with who activated them"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can access this")
    
    repo = LLM_RegistryRepo(db)
    result = repo.get_llms_with_owners(is_admin=True)
    return {"llms_with_owners": result}

