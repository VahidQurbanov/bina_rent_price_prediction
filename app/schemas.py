from pydantic import BaseModel

# Schemas for apartment features (input)
class Apartment(BaseModel):
    rooms: int
    area_m2: float
    floor_current: int
    floor_total: int
    district: str
    building_type: str
    city: str

# Schema for API response (output)
class PredictionResponse(BaseModel):
    predicted_price: float
