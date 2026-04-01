import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()


