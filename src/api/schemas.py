from pydantic import BaseModel, Field, field_validator

class PredictionRequest(BaseModel):
    pixels: list[float] = Field(
        ...,
        min_length=400,
        max_length=400,
        description="Flattened 20x20 image containing exactly 400 pixel values")

    @field_validator('pixels')
    @classmethod
    def validate_pixels(cls,value):
        if len(value) != 400:
            raise ValueError("Input must contain exactly 400 pixels values.")
        return value

    
class PredictionResponse(BaseModel):
    prediction:int
    confidence:float
    