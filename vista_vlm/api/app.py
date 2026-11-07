import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from vista_vlm.api.routes import router, get_pipeline
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.api_server")

def create_app() -> FastAPI:
    app = FastAPI(
        title="VISTA-VLM Inference & Intelligence API",
        description="Vision Intelligence through Spatial and Textual Alignment for Industrial Anomaly Detection and Localization",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable CORS for local Studio frontend and external clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "path": request.url.path}
        )

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vista_vlm.api.app:app", host="0.0.0.0", port=8000, reload=False)
