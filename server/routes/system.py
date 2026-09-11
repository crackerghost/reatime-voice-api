"""System and static UI route registration."""

from fastapi.staticfiles import StaticFiles


def register(app, *, health, ready, api_config, web_dir):
    app.get("/health")(health)
    app.get("/ready")(ready)
    app.get("/api/config")(api_config)
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
