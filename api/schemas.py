from pydantic import BaseModel
from typing import Dict, Any

class PredictPayload(BaseModel):
    features: Dict[str, Any]
