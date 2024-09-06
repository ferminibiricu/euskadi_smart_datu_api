from fastapi import APIRouter

router = APIRouter()

@router.get("/routes")
def get_transport_routes():
    return {"message": "Transport routes data will be here"}
