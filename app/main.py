import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Base, engine
from app.routers import auth, knowledge, messages, patients, sessions, upload

# ---- logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# Ensure upload directories exist before anything references them
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(os.path.join(settings.upload_dir, "knowledge"), exist_ok=True)


def _resolve_frontend_dir() -> Path | None:
    frontend_dir = Path(settings.frontend_dir)
    if not frontend_dir.is_absolute():
        frontend_dir = (Path(__file__).resolve().parent.parent / frontend_dir).resolve()
    if frontend_dir.exists() and frontend_dir.is_dir():
        return frontend_dir
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all DB tables
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Consilium API",
    description="Clinical decision-support backend for Consilium.",
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

# Serve uploaded files at /files/<filename>
app.mount("/files", StaticFiles(directory=settings.upload_dir), name="files")

PREFIX = "/api/v1"
app.include_router(auth.router, prefix=PREFIX)
app.include_router(sessions.router, prefix=PREFIX)
app.include_router(messages.router, prefix=PREFIX)
app.include_router(upload.router, prefix=PREFIX)
app.include_router(knowledge.router, prefix=PREFIX)
app.include_router(patients.router, prefix=PREFIX)

frontend_dir = _resolve_frontend_dir()
if frontend_dir is not None:
    frontend_assets_dir = frontend_dir / "assets"
    if frontend_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_assets_dir)), name="frontend-assets")

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


if frontend_dir is not None:
    @app.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(frontend_dir / "index.html")


    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend_spa(full_path: str):
        if full_path.startswith("api/") or full_path == "health" or full_path.startswith("files/"):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = frontend_dir / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dir / "index.html")


def main():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()


