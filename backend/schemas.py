from pydantic import BaseModel, Field


class HouseInput(BaseModel):
    # --- Required (core 9) ---
    district: str
    area_sqm: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=1)
    bathrooms: int = Field(..., ge=1)
    floor: int = Field(..., ge=1)
    total_floors: int = Field(..., ge=1)
    has_elevator: int = Field(..., ge=0, le=1)
    building_age_years: int = Field(..., ge=0)
    decoration: str

    # --- Optional (sensible defaults applied if omitted) ---
    living_rooms: int | None = Field(default=None, ge=0)
    orientation: str | None = None
    property_type: str | None = None
    developer: str | None = None
    subway_line: str | None = None
    subway_distance_m: float | None = Field(default=None, ge=0)
    school_district_tier: str | None = None
    parking_ratio: float | None = Field(default=None, ge=0)
    management_fee_yuan_per_sqm: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    predicted_price_wan: float
    model_version: str
    latency_ms: float