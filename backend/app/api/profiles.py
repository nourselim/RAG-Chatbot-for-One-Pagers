from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List

router = APIRouter()

@router.get("/{email}")
def get_profile(email: str) -> Dict[str, Any]:
    for p in DB_PROFILES:
        if (p.get("contact") or {}).get("email","").lower() == email.lower():
            return p
    raise HTTPException(status_code=404, detail="Profile not found")
@router.get("/")
def list_profiles():
    return [
        {"name": p.get("name"),
         "email": (p.get("contact") or {}).get("email")}
        for p in DB_PROFILES
    ]
