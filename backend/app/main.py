
from fastapi import FastAPI
from .routes import router as api_router

app = FastAPI(title="Deloitte Skills Finder")
app.include_router(api_router)
@app.get("/")
def root():
    return {"message": "Hello Deloitte Skills Finder 👋"}
