"""
FastAPI Main Application Entrypoint for Emovision Backend.
Integrates REST Endpoints, WebSocket Real-Time Stream, CORS Middleware, Database Initialization,
and OpenAPI/Swagger Documentation at /docs.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.database import init_db
from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.api.analytics import router as analytics_router
from app.api.ws import router as ws_router

# Initialize SQLite database tables
init_db()

app = FastAPI(
    title="Emovision Backend API",
    version=settings.VERSION,
    description="Real-Time Multi-Person Facial Expression Recognition API Layer",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning clean JSON error responses."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Include Routers
app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(analytics_router)
app.include_router(ws_router)

@app.get("/", include_in_schema=False)
def root_redirect():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health"
    }
