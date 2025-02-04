# models.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List

class TokenInfo(BaseModel):
    token: str
    token_id: int
    logit: float
    probability: float

class UncertaintyMetrics(BaseModel):
    raw_entropy: float
    semantic_entropy: float
    token_predictions: List[TokenInfo]
    insight: Optional[str] = None  # Add this line

class UncertaintyAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    logits: List[float]
    prompt: str
    model_id: str
    token_info: List[TokenInfo]
    metadata: Optional[Dict] = None

class UncertaintyAnalysisResult(BaseModel):
    token_metrics: List[UncertaintyMetrics]
    overall_insight: Optional[str] = None
    model_config = ConfigDict(protected_namespaces=())