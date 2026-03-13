from fastapi import APIRouter
from .langChainsRoutes import chain_route
from .Users import router as users_router
from .LlmReg import router as llm_router
v1_router=APIRouter()
v1_router.include_router(chain_route,tags=["LLM"])
v1_router.include_router(users_router,tags=["User"])
v1_router.include_router(llm_router,tags=["RegisterLLM"])