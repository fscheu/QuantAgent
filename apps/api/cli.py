"""CLI commands for serving the HTTP API."""

import click
import uvicorn


@click.group(help="Run the QuantAgent HTTP API.")
def api_group():
    pass


@api_group.command(help="Serve the read-only QuantAgent API.", epilog="Example: quantagent api serve --port 8000")
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface address to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="TCP port to bind.")
def serve(host, port):
    uvicorn.run("apps.api.app:app", host=host, port=port)
