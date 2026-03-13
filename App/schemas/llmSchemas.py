from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LLMRegister(BaseModel):
    """Schema for registering a new LLM"""
    model_name: str = Field(..., description="Custom name for the LLM (can be different from filename)")
    model_filename: str = Field(..., description="Filename of the model in the models directory")


class LLMUpdate(BaseModel):
    """Schema for updating an LLM (admin only)"""
    model_name: Optional[str] = Field(None, description="Custom name for the LLM")
    model_filename: Optional[str] = Field(None, description="Filename in the models directory")
    vram_estimate_gb: Optional[float] = Field(None, description="Estimated VRAM usage in GB")
    is_active: Optional[bool] = Field(None, description="Whether the LLM is active")


class LLMResponse(BaseModel):
    """Schema for LLM response"""
    id: int
    user_id: int
    model_name: str
    model_path: str
    vram_estimate_gb: Optional[float] = None
    file_size_bytes: Optional[int] = None
    is_active: bool
    is_deleted: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
