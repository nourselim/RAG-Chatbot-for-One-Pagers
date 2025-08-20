from fastapi import APIRouter
from .api import profiles as profiles_routes
from .api import chat as chat_routes

router = APIRouter()
router.include_router(profiles_routes.router, prefix="/profiles", tags=["profiles"])
router.include_router(chat_routes.router,     prefix="/sessions", tags=["chat"])
