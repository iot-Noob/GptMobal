from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from pathlib import Path
from App.api.databases.Tables import RegisterLLM, User
from App.core.LoggingInit import get_core_logger
import os

logger = get_core_logger(__name__)

class LLM_RegistryRepo:
    def __init__(self, db: Session):
        self.db = db

    # --- READ OPERATIONS ---

    def get_llms(self, user_id: int, is_admin: bool = False) -> Dict[int, Any]:
        """
        Fetches LLMs based on role. 
        Admin: All LLMs (including inactive/deleted).
        User: Only active, non-deleted models belonging to them.
        """
        try:
            query = self.db.query(RegisterLLM)
            
            if is_admin:
                # Admin gets everything not hard-deleted
                results = query.all()
                admin_data = {}
                for llm in results:
                    if llm.user_id not in admin_data:
                        admin_data[llm.user_id] = {}
                    
                    admin_data[llm.user_id][llm.id] = {
                        "model_name": llm.model_name,
                        "model_path": llm.model_path,
                        "is_active": llm.is_active,
                        "is_deleted": llm.is_deleted,
                        "updated_at": llm.updated_at,
                        "created_at": llm.created_at
                    }
                return admin_data
            else:
                # Normal user: Only active, non-deleted models belonging to them
                results = query.filter(
                    and_(
                        RegisterLLM.user_id == user_id,
                        RegisterLLM.is_deleted == False,
                        RegisterLLM.is_active == True  # Only active LLMs
                    )
                ).all()
                
                return {
                    llm.id: {
                        "user_id": llm.user_id,
                        "details": {
                            "model_name": llm.model_name,
                            "model_path": llm.model_path,
                            "is_active": llm.is_active
                        },
                        "created_at": llm.created_at,
                        "updated_at": llm.updated_at,
                        "deleted_at": llm.deleted_at if hasattr(llm, 'deleted_at') else None
                    } for llm in results
                }

        except Exception as e:
            logger.error(f"❌ Error fetching LLM Registry: {e}")
            return {}

    # --- UPDATE OPERATIONS ---

    def update_llm(self, llm_id: int, user_id: int, is_admin: bool, update_data: Dict[str, Any]):
        """
        Admin can update any model/path. 
        User can only update their own assigned models.
        """
        try:
            query = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id)
            
            if not is_admin:
                query = query.filter(RegisterLLM.user_id == user_id)
            
            llm = query.first()
            if not llm:
                return False

            # Restriction: Only admin can toggle is_active
            if "is_active" in update_data and not is_admin:
                update_data.pop("is_active")
                logger.warning(f"⚠️ User {user_id} tried to toggle status without admin rights.")

            for key, value in update_data.items():
                setattr(llm, key, value)
            
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Update failed: {e}")
            return False

    # --- DELETE & RESTORE (Soft Delete) ---

    def soft_delete_llm(self, llm_id: int, user_id: int, is_admin: bool):
        try:
            query = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id)
            if not is_admin:
                query = query.filter(RegisterLLM.user_id == user_id)
            
            llm = query.first()
            if not llm:
                return {"success": False, "message": "LLM not found"}
            
            # Check if already deleted
            if llm.is_deleted:
                return {"success": False, "message": "LLM is already deleted"}
            
            llm.is_deleted = True
            self.db.commit()
            return {"success": True, "message": "LLM moved to trash"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Delete failed: {e}")
            return {"success": False, "message": f"Delete failed: {str(e)}"}

    def restore_llm(self, llm_id: int, is_admin: bool):
        """Only Admin can restore a deleted LLM"""
        if not is_admin:
            return {"success": False, "message": "Only admins can restore LLMs"}
        try:
            llm = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id).first()
            if not llm:
                return {"success": False, "message": "LLM not found"}
            
            # Check if already not deleted
            if not llm.is_deleted:
                return {"success": False, "message": "LLM is not deleted, cannot restore"}
            
            llm.is_deleted = False
            self.db.commit()
            return {"success": True, "message": "LLM restored successfully"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Restore failed: {e}")
            return {"success": False, "message": f"Restore failed: {str(e)}"}

    def set_llm_active(self, llm_id: int, user_id: int, is_admin: bool, is_active: bool):
        """
        Set LLM as active/inactive (loaded/unloaded).
        Users can only activate their own LLMs.
        Admins can activate any LLM.
        """
        try:
            query = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id)
            if not is_admin:
                query = query.filter(RegisterLLM.user_id == user_id)
            
            llm = query.first()
            if not llm:
                return {"success": False, "message": "LLM not found or unauthorized"}
            
            # Check if already in desired state
            if llm.is_active == is_active:
                status = "active" if is_active else "inactive"
                return {"success": False, "message": f"LLM is already {status}"}
            
            # Cannot activate a deleted LLM
            if is_active and llm.is_deleted:
                return {"success": False, "message": "Cannot activate a deleted LLM"}
            
            llm.is_active = is_active
            self.db.commit()
            
            status = "activated" if is_active else "deactivated"
            return {"success": True, "message": f"LLM {status} successfully"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Set active failed: {e}")
            return {"success": False, "message": f"Failed: {str(e)}"}

    def get_deleted_llms(self, is_admin: bool):
        """Admin only view for trash bin"""
        if not is_admin:
            return None
        return self.db.query(RegisterLLM).filter(RegisterLLM.is_deleted == True).all()

    # --- DUPLICATE CHECK ---

    def check_duplicate_model_name(self, model_name: str, user_id: int) -> Dict[str, Any]:
        """
        Check if a model with the same custom name already exists for this user.
        Returns dict with is_duplicate and existing LLM info.
        """
        try:
            existing = self.db.query(RegisterLLM).filter(
                RegisterLLM.model_name == model_name,
                RegisterLLM.user_id == user_id,
                RegisterLLM.is_deleted == False
            ).first()
            
            if existing:
                return {
                    "is_duplicate": True,
                    "existing_llm_id": existing.id,
                    "message": f"Model with name '{model_name}' already exists"
                }
            
            return {"is_duplicate": False, "message": "No duplicate found"}
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return {"is_duplicate": False, "message": "Error checking duplicates"}

    def get_file_size(self, model_path: Path) -> Optional[int]:
        """Get file size in bytes"""
        try:
            if model_path.exists() and model_path.is_file():
                return os.path.getsize(model_path)
            return None
        except Exception as e:
            logger.error(f"Get file size failed: {e}")
            return None

    def get_models_in_directory(self, base_path: Path) -> List[Dict[str, Any]]:
        """
        Get all model files in the models directory.
        Returns list of dicts with filename, full_path, and size.
        """
        try:
            if not base_path.exists() or not base_path.is_dir():
                return []
            
            models = []
            for file_path in base_path.iterdir():
                if file_path.is_file():
                    # Get file size
                    file_size = os.path.getsize(file_path)
                    models.append({
                        "filename": file_path.name,
                        "full_path": str(file_path),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "size_gb": round(file_size / (1024 * 1024 * 1024), 2)
                    })
            
            return models
        except Exception as e:
            logger.error(f"Get models in directory failed: {e}")
            return []

    # --- PATH VALIDATION ---

    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """
        Validates if the model path exists and is accessible.
        Returns dict with is_valid and message.
        """
        try:
            if not model_path:
                return {"is_valid": False, "message": "Model path cannot be empty"}
            
            # Convert to absolute path if relative
            path = os.path.abspath(model_path)
            
            if not os.path.exists(path):
                return {"is_valid": False, "message": f"Path does not exist: {path}"}
            
            if not os.path.isfile(path):
                return {"is_valid": False, "message": f"Path is not a file: {path}"}
            
            if not os.access(path, os.R_OK):
                return {"is_valid": False, "message": f"Path is not readable: {path}"}
            
            # Get file size
            file_size = os.path.getsize(path)
            
            return {
                "is_valid": True, 
                "message": "Path is valid",
                "absolute_path": path,
                "file_size_bytes": file_size
            }
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return {"is_valid": False, "message": f"Error validating path: {str(e)}"}

    def validate_model_path_within_base(self, model_path: Path, base_path: Path) -> Dict[str, Any]:
        """
        Validates if the model path exists, is a file, and is within the base directory.
        Returns dict with is_valid and message.
        """
        try:
            # Resolve base path to absolute
            base_path = base_path.resolve()
            
            # If model_path is just a filename, join with base_path
            if not str(model_path).startswith(str(base_path)):
                model_path = base_path / model_path
            
            # Resolve to absolute path
            model_path = model_path.resolve()
            
            if not model_path.exists():
                return {
                    "is_valid": False, 
                    "message": f"Model file not found: {model_path.name}. Please ensure the file exists in the models directory."
                }
            
            if not model_path.is_file():
                return {
                    "is_valid": False, 
                    "message": f"Path is not a file: {model_path}"
                }
            
            # Check if path is within base directory (prevent directory traversal)
            try:
                model_path.relative_to(base_path)
            except ValueError:
                return {
                    "is_valid": False, 
                    "message": f"Invalid path: {model_path}. File must be within the models directory: {base_path}"
                }
            
            if not os.access(model_path, os.R_OK):
                return {
                    "is_valid": False, 
                    "message": f"File is not readable: {model_path}"
                }
            
            file_size = os.path.getsize(model_path)
            
            return {
                "is_valid": True, 
                "message": "Path is valid",
                "absolute_path": str(model_path),
                "relative_path": str(model_path.relative_to(base_path)),
                "file_size_bytes": file_size
            }
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return {"is_valid": False, "message": f"Error validating path: {str(e)}"}

    # --- ASSIGN LLM TO USER ---

    def assign_llm_to_user(self, llm_id: int, target_user_id: int, admin_id: int) -> Dict[str, Any]:
        """
        Admin assigns an LLM to another user.
        Returns dict with success status and message.
        """
        try:
            # Check if LLM exists
            llm = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id).first()
            if not llm:
                return {"success": False, "message": "LLM not found"}
            
            # Check if target user exists
            target_user = self.db.query(User).filter(User.id == target_user_id).first()
            if not target_user:
                return {"success": False, "message": "Target user not found"}
            
            # Check if admin has permission (admin_id should be admin)
            admin = self.db.query(User).filter(User.id == admin_id).first()
            if not admin or not admin.is_admin:
                return {"success": False, "message": "Only admins can assign LLMs to other users"}
            
            # Update the LLM's user_id
            old_user_id = llm.user_id
            llm.user_id = target_user_id
            self.db.commit()
            
            logger.info(f"LLM {llm_id} assigned from user {old_user_id} to user {target_user_id} by admin {admin_id}")
            return {
                "success": True, 
                "message": f"LLM assigned to user {target_user_id} successfully",
                "llm_id": llm_id,
                "new_owner_id": target_user_id
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Assign LLM failed: {e}")
            return {"success": False, "message": f"Error assigning LLM: {str(e)}"}

    # --- GET SINGLE LLM ---

    def get_llm_by_id(self, llm_id: int, user_id: int, is_admin: bool) -> Optional[Dict[str, Any]]:
        """
        Get single LLM details.
        Admin can view any LLM.
        Normal user can only view their own active LLMs.
        """
        try:
            query = self.db.query(RegisterLLM).filter(RegisterLLM.id == llm_id)
            
            if not is_admin:
                query = query.filter(
                    RegisterLLM.user_id == user_id,
                    RegisterLLM.is_active == True,  # Only active
                    RegisterLLM.is_deleted == False
                )
            
            llm = query.first()
            if not llm:
                return None
            
            return {
                "id": llm.id,
                "user_id": llm.user_id,
                "model_name": llm.model_name,
                "model_path": llm.model_path,
                "vram_estimate_gb": llm.vram_estimate_gb,
                "is_active": llm.is_active,
                "is_deleted": llm.is_deleted,
                "created_at": llm.created_at.isoformat() if llm.created_at else None,
                "updated_at": llm.updated_at.isoformat() if llm.updated_at else None
            }
        except Exception as e:
            logger.error(f"Get LLM by ID failed: {e}")
            return None

    # --- GET ALL AVAILABLE LLMS (For Assignment) ---

    def get_all_llms_for_admin(self, admin_id: int) -> List[Dict[str, Any]]:
        """
        Admin gets all LLMs (including deleted) for management purposes.
        """
        try:
            # Verify admin
            admin = self.db.query(User).filter(User.id == admin_id).first()
            if not admin or not admin.is_admin:
                return []
            
            llms = self.db.query(RegisterLLM).all()
            return [
                {
                    "id": llm.id,
                    "user_id": llm.user_id,
                    "model_name": llm.model_name,
                    "model_path": llm.model_path,
                    "vram_estimate_gb": llm.vram_estimate_gb,
                    "is_active": llm.is_active,
                    "is_deleted": llm.is_deleted,
                    "created_at": llm.created_at.isoformat() if llm.created_at else None
                }
                for llm in llms
            ]
        except Exception as e:
            logger.error(f"Get all LLMs for admin failed: {e}")
            return []