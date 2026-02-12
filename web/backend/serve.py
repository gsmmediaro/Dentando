"""Entrypoint to run the FastAPI app on Railway."""

import os

import uvicorn


def main() -> None:
    """Start Uvicorn using PORT from environment."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web.backend.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
