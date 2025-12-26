from pydantic import BaseModel

# Mənzil xüsusiyyətləri üçün şemalar (input)
class Apartment(BaseModel):
    rooms: int
    area_m2: float
    floor_current: int
    floor_total: int
    district: str
    building_type: str
    city: str

# API response üçün şema (çıxış)
class PredictionResponse(BaseModel):
    predicted_price: float
