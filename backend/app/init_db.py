from .db import Base, engine
from .models import ChatMessage

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
