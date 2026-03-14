from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SystemPromptCreate(BaseModel):
    """Schema for creating a new system prompt"""
    role: str = Field(..., description="Role name for this prompt (e.g., 'Analyst', 'Assistant')")
    prompt: str = Field(..., description="The system prompt text")


class SystemPromptUpdate(BaseModel):
    """Schema for updating a system prompt"""
    role: Optional[str] = None
    prompt: Optional[str] = None
    is_active: Optional[bool] = None


class SystemPromptResponse(BaseModel):
    """Schema for system prompt response"""
    id: int
    user_id: int
    role: str
    prompt: str
    is_active: bool
    is_deleted: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
