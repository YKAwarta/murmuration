from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictPayload(BaseModel):
    features: Dict[str, Any]
    # Optional override of the default decision threshold (top calibrated probability)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
