# schemas.py
from pydantic import BaseModel, Field

class HouseFeatures(BaseModel):
    square_meters: float = Field(..., description="Total area of the house", example=120.5)
    bedrooms: int = Field(..., description="Number of bedrooms", example=3)
    bathrooms: int = Field(..., description="Number of bathrooms", example=2)
    year_built: int = Field(..., description="Year constructed", example=2015)