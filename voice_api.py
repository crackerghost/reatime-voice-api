"""Compatibility entrypoint for the modular Voice Cloning server."""

from server.app import app

__all__ = ["app"]


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("VOICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("VOICE_PORT", "8000")),
    )
