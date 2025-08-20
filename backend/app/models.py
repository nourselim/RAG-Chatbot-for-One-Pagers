import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from .db import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(8), nullable=False)      # 'user' | 'bot'
    message: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column("timestamp", DateTime, nullable=False)

    __table_args__ = (
        CheckConstraint("role in ('user','bot')", name="ck_chat_messages_role"),
    )
