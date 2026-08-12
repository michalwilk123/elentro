"""HTTP interface for persistent catalogue search and runtime mutations."""

from __future__ import annotations

import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import DATA, build_catalogue
from .search import SearchResult
from .store import (
    CatalogueStore,
    InvalidProduct,
    ProductConflict,
    ProductInput,
    ProductNotFound,
)

STATIC_DIR = Path(__file__).parent / "static"
DATABASE_PATH = DATA / "katalog.sqlite3"

app = FastAPI(docs_url=None, redoc_url=None)
_store_lock = threading.Lock()


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    category: str | None = None
    package: str = ""
    price: str = ""
    attributes: str = ""
    description: str = ""
    applications: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/search")
def search(
    q: str = "", manufacturer: str = "", category: str = "", limit: int = 10
) -> dict:
    if not q.strip():
        return {"results": []}
    results = _get_store().search(
        q,
        manufacturer=manufacturer or None,
        category=category or None,
        limit=max(1, limit),
    )
    return {"results": [_as_json(result) for result in results]}


@app.post("/api/products", status_code=status.HTTP_201_CREATED)
def add_product(data: ProductCreate) -> dict:
    try:
        product = _get_store().add(ProductInput(**data.model_dump()))
    except InvalidProduct as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ProductConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "product": _product_json(product),
        "issues": [str(issue) for issue in product.issues],
    }


@app.delete("/api/products/{sku}")
def delete_product(sku: str) -> dict:
    try:
        deleted = _get_store().delete(sku)
    except ProductNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"deleted_sku": deleted.canonical_sku, "deleted_aliases": deleted.aliases}


def _get_store() -> CatalogueStore:
    configured = getattr(app.state, "store", None)
    if configured is not None:
        return configured

    with _store_lock:
        configured = getattr(app.state, "store", None)
        if configured is None:
            configured = CatalogueStore(DATABASE_PATH)
            if not configured.seeded:
                configured.seed(build_catalogue())
            app.state.store = configured
        return configured


def _as_json(result: SearchResult) -> dict:
    return {
        **_product_json(result.product),
        "score": result.score,
        "match_type": result.match_type,
        "explanation": result.explanation,
    }


def _product_json(product) -> dict:
    return {
        "sku": product.canonical_sku,
        "aliases": product.alias_skus,
        "name": product.name,
        "manufacturer": product.manufacturer,
        "category": product.category,
        "package": str(product.package),
        "price": str(product.price),
    }


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
