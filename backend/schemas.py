from pydantic import BaseModel, Field


class HouseInput(BaseModel):
    district: str
    area_sqm: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=1)
    living_rooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=1)
    floor: int = Field(..., ge=1)
    total_floors: int = Field(..., ge=1)
    has_elevator: int = Field(..., ge=0, le=1)
    building_age_years: int = Field(..., ge=0)
    orientation: str
    decoration: str
    property_type: str
    developer: str
    subway_line: str
    subway_distance_m: float | None = Field(default=None, ge=0)
    school_district_tier: str
    parking_ratio: float = Field(..., ge=0)
    management_fee_yuan_per_sqm: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_price_wan: float
    model_version: str
    latency_ms: float