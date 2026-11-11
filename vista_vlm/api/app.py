import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from vista_vlm.api.routes import router
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.api_server")

def create_app() -> FastAPI:
    app = FastAPI(
        title="VISTA-VLM Inference & Intelligence API",
        description="Vision Intelligence through Spatial and Textual Alignment for Industrial Anomaly Detection and Localization",
        version="7.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configurable CORS with safe local development defaults
    allowed_origins = os.environ.get(
        "VISTA_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    ).split(",")
    origins_list = [o.strip() for o in allowed_origins if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Inspection request could not be processed.", "path": request.url.path}
        )

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vista_vlm.api.app:app", host="0.0.0.0", port=8000, reload=False)
