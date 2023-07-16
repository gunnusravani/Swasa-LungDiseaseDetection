from typing import Optional
from pydantic import BaseModel, HttpUrl

class ImageCreate(BaseModel):
    img_file: str
    patient_name: str
    patient_dob: str
      # Add image_data field
