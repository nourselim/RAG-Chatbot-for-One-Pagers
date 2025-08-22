from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as api_router

app = FastAPI(title="DeBotte")

# CORS: allow Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API routers
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Hello DeBotte"}
