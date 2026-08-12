import asyncio

import httpx

from katalog.models import Catalogue
from katalog.store import CatalogueStore
from katalog.web import app


class FakeEmbedder:
    model_name = "web-fake-v1"

    def embed(self, texts):
        return [[1.0, 2.0 if "cold" in text.casefold() else 0.1] for text in texts]


def test_http_add_search_delete_round_trip(tmp_path):
    store = CatalogueStore(tmp_path / "web.sqlite3", FakeEmbedder())
    store.seed(Catalogue([], [], []))
    app.state.store = store

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            created = await client.post(
                "/api/products",
                json={
                    "sku": "AC-10001",
                    "name": "Cold reagent",
                    "manufacturer": "ACME",
                    "category": "LAB_CHEMICAL",
                    "price": "10 PLN",
                    "description": "cold storage reagent",
                },
            )
            assert created.status_code == 201
            assert created.json()["product"]["sku"] == "AC-10001"

            exact = await client.get("/api/search", params={"q": "AC-10001"})
            assert exact.json()["results"][0]["match_type"] == "exact_sku"
            semantic = await client.get("/api/search", params={"q": "cold"})
            assert semantic.json()["results"][0]["sku"] == "AC-10001"

            conflict = await client.post(
                "/api/products",
                json={"sku": "AC-10001", "name": "Duplicate", "manufacturer": "ACME"},
            )
            assert conflict.status_code == 409

            deleted = await client.delete("/api/products/ac-10001")
            assert deleted.status_code == 200
            assert deleted.json()["deleted_sku"] == "AC-10001"
            missing = await client.get("/api/search", params={"q": "AC-10001"})
            assert missing.json()["results"] == []
            assert (await client.delete("/api/products/AC-10001")).status_code == 404

    asyncio.run(scenario())


def test_http_rejects_unknown_category(tmp_path):
    store = CatalogueStore(tmp_path / "invalid.sqlite3", FakeEmbedder())
    store.seed(Catalogue([], [], []))
    app.state.store = store

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/api/products",
                json={
                    "sku": "AC-1",
                    "name": "Product",
                    "manufacturer": "ACME",
                    "category": "OTHER",
                },
            )

    response = asyncio.run(scenario())
    assert response.status_code == 422
