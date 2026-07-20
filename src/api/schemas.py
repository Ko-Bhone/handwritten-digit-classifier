from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    pixels: list[float] = Field(
        ...,
        min_length=400,
        max_length=400,
        description="Flattened 20x20 image containing exactly 400 pixel values")


class PredictionResponse(BaseModel):
    prediction:int
    confidence:float
    