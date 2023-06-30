from fastapi import APIRouter, Depends, HTTPException

from container.api.dependencies import get_token_header

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

fake_items_db = {"one": {"name": "two"}, "three": {"name": "three"}}


@router.get("/")
async def read_items():
    return fake_items_db


@router.get("/{prediction_id}")
async def read_item(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}
