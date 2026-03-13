from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SystemPromptCreate(BaseModel):
    """Schema for creating a new system prompt"""
    model_id: int = Field(..., description="ID of the LLM model")
    persona_name: str = Field(..., description="Name for this persona (e.g., 'Analyst')")
    prompt_text: str = Field(..., description="The system prompt text")


class SystemPromptUpdate(BaseModel):
    """Schema for updating a system prompt"""
    persona_name: Optional[str] = None
    prompt_text: Optional[str] = None
    is_active: Optional[bool] = None


class SystemPromptResponse(BaseModel):
    """Schema for system prompt response"""
    id: int
    user_id: int
    model_id: int
    persona_name: str
    prompt_text: str
    version_number: int
    is_active: bool
    is_deleted: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
