# app/api/chat.py
from uuid import UUID
from typing import List, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..deps import get_db
from ..models import ChatMessage

router = APIRouter()

class MessageIn(BaseModel):
    role: Literal["user", "bot"]
    message: str

class MessageOut(BaseModel):
    id: int
    session_id: UUID
    role: str
    message: str
    timestamp: str

    @classmethod
    def from_row(cls, m: ChatMessage):
        return cls(
            id=m.id, session_id=m.session_id, role=m.role,
            message=m.message, timestamp=m.timestamp.isoformat()
        )

@router.get("/{session_id}/messages", response_model=List[MessageOut])
def list_messages(session_id: UUID,
                  order: Literal["asc","desc"]=Query("asc"),
                  limit: int = Query(100, ge=1, le=1000),
                  db: Session = Depends(get_db)):
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
    stmt = stmt.order_by(ChatMessage.timestamp.asc() if order=="asc" else ChatMessage.timestamp.desc()).limit(limit)
    return [MessageOut.from_row(m) for m in db.execute(stmt).scalars().all()]

@router.post("/{session_id}/messages", response_model=MessageOut, status_code=201)
def add_message(session_id: UUID, body: MessageIn, db: Session = Depends(get_db)):
    row = ChatMessage(session_id=session_id, role=body.role, message=body.message)
    db.add(row); db.commit(); db.refresh(row)
    return MessageOut.from_row(row)
