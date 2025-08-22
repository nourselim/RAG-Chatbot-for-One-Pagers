# app/api/chat.py
from uuid import UUID
from typing import List, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..deps import get_db
from ..models import ChatMessage
from sqlalchemy import func
from datetime import datetime, timezone
from typing import Optional

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
    row = ChatMessage(session_id=session_id, role=body.role, message=body.message, timestamp=datetime.now(timezone.utc),)
    db.add(row); db.commit(); db.refresh(row)
    return MessageOut.from_row(row)





class SessionSummary(BaseModel):
    session_id: UUID
    started_at: str
    updated_at: str
    last_role: Optional[str] = None
    last_message: Optional[str] = None


@router.get("/", response_model=List[SessionSummary])
def list_sessions(limit: int = 50, db: Session = Depends(get_db)):
    # one row per session_id with min/max timestamps
    sub = (
        select(
            ChatMessage.session_id.label("session_id"),
            func.min(ChatMessage.timestamp).label("started_at"),
            func.max(ChatMessage.timestamp).label("updated_at"),
        )
        .group_by(ChatMessage.session_id)
        .subquery()
    )

    # join to fetch last message for each session
    last = (
        select(
            ChatMessage.session_id, ChatMessage.role, ChatMessage.message, ChatMessage.timestamp
        )
        .join(sub, ChatMessage.session_id == sub.c.session_id)
        .where(ChatMessage.timestamp == sub.c.updated_at)
        .subquery()
    )

    rows = db.execute(
        select(
            sub.c.session_id, sub.c.started_at, sub.c.updated_at,
            last.c.role.label("last_role"), last.c.message.label("last_message")
        )
        .order_by(sub.c.updated_at.desc())
        .limit(limit)
    ).all()

    return [
        SessionSummary(
            session_id=r.session_id,
            started_at=r.started_at.isoformat(),
            updated_at=r.updated_at.isoformat(),
            last_role=r.last_role,
            last_message=r.last_message,
        )
        for r in rows
    ]