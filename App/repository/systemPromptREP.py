from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from App.api.databases.Tables import RegisterLLM, User, SystemPrompt
from App.core.LoggingInit import get_core_logger

logger = get_core_logger(__name__)


class SystemPromptRepo:
    def __init__(self, db: Session):
        self.db = db

    def create_prompt(self, user_id: int, model_id: int, persona_name: str, prompt_text: str) -> Dict[str, Any]:
        """Create a new system prompt"""
        try:
            # Check if model exists and belongs to user (or user is admin)
            model = self.db.query(RegisterLLM).filter(RegisterLLM.id == model_id).first()
            if not model:
                return {"success": False, "message": "Model not found"}
            
            new_prompt = SystemPrompt(
                user_id=user_id,
                model_id=model_id,
                persona_name=persona_name,
                prompt_text=prompt_text,
                is_active=True
            )
            self.db.add(new_prompt)
            self.db.commit()
            self.db.refresh(new_prompt)
            
            return {
                "success": True,
                "message": "System prompt created",
                "prompt_id": new_prompt.id
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Create prompt failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def get_prompts(self, user_id: int, is_admin: bool = False) -> List[Dict[str, Any]]:
        """
        Get prompts for user.
        Admin sees all non-deleted prompts.
        User sees only: assigned to them + not deleted + is_active.
        Deleted prompts are NEVER shown, even to admins.
        """
        try:
            query = self.db.query(SystemPrompt)
            
            # Always filter out deleted prompts - even admins cannot see deleted prompts
            query = query.filter(SystemPrompt.is_deleted == False)
            
            if not is_admin:
                # Normal user: only own prompts that are active
                query = query.filter(
                    SystemPrompt.user_id == user_id,
                    SystemPrompt.is_active == True
                )
            
            prompts = query.all()
            return [
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "model_id": p.model_id,
                    "persona_name": p.persona_name,
                    "prompt_text": p.prompt_text,
                    "version_number": p.version_number,
                    "is_active": p.is_active,
                    "is_deleted": p.is_deleted,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in prompts
            ]
        except Exception as e:
            logger.error(f"Get prompts failed: {e}")
            return []

    def get_prompt_by_id(self, prompt_id: int, user_id: int, is_admin: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get single prompt by ID.
        Admin can view any non-deleted prompt.
        User can only view own prompts that are active and not deleted.
        Deleted prompts are NEVER shown, even to admins.
        """
        try:
            query = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id)
            
            # Always filter out deleted prompts - even admins cannot see deleted prompts
            query = query.filter(SystemPrompt.is_deleted == False)
            
            if not is_admin:
                query = query.filter(
                    SystemPrompt.user_id == user_id,
                    SystemPrompt.is_active == True
                )
            
            prompt = query.first()
            if not prompt:
                return None
            
            return {
                "id": prompt.id,
                "user_id": prompt.user_id,
                "model_id": prompt.model_id,
                "persona_name": prompt.persona_name,
                "prompt_text": prompt.prompt_text,
                "version_number": prompt.version_number,
                "is_active": prompt.is_active,
                "is_deleted": prompt.is_deleted,
                "created_at": prompt.created_at.isoformat() if prompt.created_at else None
            }
        except Exception as e:
            logger.error(f"Get prompt by ID failed: {e}")
            return None

    def update_prompt(self, prompt_id: int, user_id: int, is_admin: bool, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a prompt (user can update own, admin can update any)"""
        try:
            query = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id)
            
            if not is_admin:
                query = query.filter(SystemPrompt.user_id == user_id)
            
            prompt = query.first()
            if not prompt:
                return {"success": False, "message": "Prompt not found or unauthorized"}
            
            for key, value in update_data.items():
                if hasattr(prompt, key):
                    setattr(prompt, key, value)
            
            self.db.commit()
            return {"success": True, "message": "Prompt updated"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Update prompt failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def soft_delete_prompt(self, prompt_id: int, user_id: int, is_admin: bool) -> Dict[str, Any]:
        """Soft delete a prompt (check if already deleted)"""
        try:
            query = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id)
            
            if not is_admin:
                query = query.filter(SystemPrompt.user_id == user_id)
            
            prompt = query.first()
            if not prompt:
                return {"success": False, "message": "Prompt not found or unauthorized"}
            
            if prompt.is_deleted:
                return {"success": False, "message": "Prompt is already deleted"}
            
            prompt.is_deleted = True
            self.db.commit()
            return {"success": True, "message": "Prompt moved to trash"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Delete prompt failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def restore_prompt(self, prompt_id: int, is_admin: bool) -> Dict[str, Any]:
        """Restore a deleted prompt (admin only, check if not deleted)"""
        if not is_admin:
            return {"success": False, "message": "Only admins can restore prompts"}
        
        try:
            prompt = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
            if not prompt:
                return {"success": False, "message": "Prompt not found"}
            
            if not prompt.is_deleted:
                return {"success": False, "message": "Prompt is not deleted, cannot restore"}
            
            prompt.is_deleted = False
            self.db.commit()
            return {"success": True, "message": "Prompt restored"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Restore prompt failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def assign_prompt_to_user(self, prompt_id: int, target_user_id: int, admin_id: int) -> Dict[str, Any]:
        """Admin assigns a prompt to another user"""
        try:
            prompt = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
            if not prompt:
                return {"success": False, "message": "Prompt not found"}
            
            # Prevent assigning deleted prompts
            if prompt.is_deleted:
                return {"success": False, "message": "Cannot assign a deleted prompt"}
            
            target_user = self.db.query(User).filter(User.id == target_user_id).first()
            if not target_user:
                return {"success": False, "message": "Target user not found"}
            
            admin = self.db.query(User).filter(User.id == admin_id).first()
            if not admin or not admin.is_admin:
                return {"success": False, "message": "Only admins can assign prompts"}
            
            old_user_id = prompt.user_id
            prompt.user_id = target_user_id
            self.db.commit()
            
            return {
                "success": True,
                "message": f"Prompt assigned to user {target_user_id}",
                "prompt_id": prompt_id,
                "new_owner_id": target_user_id
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Assign prompt failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def assign_prompt_to_model(self, prompt_id: int, model_id: int, admin_id: int) -> Dict[str, Any]:
        """Admin assigns a prompt to a different model"""
        try:
            prompt = self.db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
            if not prompt:
                return {"success": False, "message": "Prompt not found"}
            
            # Prevent assigning deleted prompts
            if prompt.is_deleted:
                return {"success": False, "message": "Cannot assign a deleted prompt"}
            
            model = self.db.query(RegisterLLM).filter(RegisterLLM.id == model_id).first()
            if not model:
                return {"success": False, "message": "Model not found"}
            
            admin = self.db.query(User).filter(User.id == admin_id).first()
            if not admin or not admin.is_admin:
                return {"success": False, "message": "Only admins can reassign prompts"}
            
            old_model_id = prompt.model_id
            prompt.model_id = model_id
            self.db.commit()
            
            return {
                "success": True,
                "message": f"Prompt assigned to model {model_id}",
                "prompt_id": prompt_id,
                "new_model_id": model_id
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Assign prompt to model failed: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def get_deleted_prompts(self, is_admin: bool) -> List[Dict[str, Any]]:
        """Get deleted prompts (admin only)"""
        if not is_admin:
            return []
        
        try:
            prompts = self.db.query(SystemPrompt).filter(SystemPrompt.is_deleted == True).all()
            return [
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "model_id": p.model_id,
                    "persona_name": p.persona_name,
                    "is_deleted": p.is_deleted,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in prompts
            ]
        except Exception as e:
            logger.error(f"Get deleted prompts failed: {e}")
            return []
