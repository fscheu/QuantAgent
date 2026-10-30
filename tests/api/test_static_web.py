import importlib
import asyncio

from httpx import ASGITransport, AsyncClient


def test_web_dist_serves_assets_and_spa_without_capturing_api(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<main>web app</main>")
    (tmp_path / "asset.txt").write_text("asset")
    monkeypatch.setenv("QUANTAGENT_WEB_DIST", str(tmp_path))
    import apps.api.app as api_app
    api_app = importlib.reload(api_app)
    async def requests():
        async with AsyncClient(transport=ASGITransport(app=api_app.app), base_url="http://test") as client:
            assert (await client.get("/")).text == "<main>web app</main>"
            assert (await client.get("/dashboard/settings")).text == "<main>web app</main>"
            assert (await client.get("/asset.txt")).text == "asset"
            assert (await client.get("/api/missing")).status_code == 404

    asyncio.run(requests())
