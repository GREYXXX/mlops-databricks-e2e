from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import comparison, config, experiments, models, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Repo root .env (local dev); production injects env vars instead
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    yield
    # Shutdown


app = FastAPI(
    title="MLOps E2E Dashboard",
    description="Dashboard for monitoring the MLOps E2E pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(experiments.router)
app.include_router(models.router)
app.include_router(comparison.router)
app.include_router(config.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = (FRONTEND_DIR / full_path).resolve()
        if file_path.is_relative_to(FRONTEND_DIR) and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
