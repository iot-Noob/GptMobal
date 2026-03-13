from sqlalchemy import Column, Integer, String, Boolean, DateTime,func,ForeignKey,Float,Text
from datetime import datetime
from typing import List, Optional
from App.api.dependencies.sqlite_connector import Base, engine
# User Model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    password_hash = Column(String)
    user_role = Column(String, default="user")  # user, admin
    is_active = Column(Boolean, default=True)
    disabled = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class RegisterLLM(Base): # <--- Must inherit from Base
    __tablename__ = "registered_llms"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # <--- Reference the table.column string
    model_name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    vram_estimate_gb = Column(Float, nullable=True) # Key for your RX 5600 XT
    is_active = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# 3. System Prompt Model
class SystemPrompt(Base):
    __tablename__ = "system_prompts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("registered_llms.id"))
    persona_name = Column(String, index=True) # e.g., 'Analyst'
    prompt_text = Column(Text, nullable=False) # Use Text for long prompts
    version_number = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

Base.metadata.create_all(bind=engine)

