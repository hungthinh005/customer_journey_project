"""Pydantic schemas for the Churn Prevention System API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for churn prediction + recommendation."""

    customer_id: int = Field(..., description="Customer ID to predict for")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    use_llm_reranker: bool = Field(default=False, description="Whether to use LLM reranking")


class ProductRecommendation(BaseModel):
    """A single product recommendation."""

    stock_code: str
    description: Optional[str] = None
    score: float
    rank: int


class PredictionResponse(BaseModel):
    """Response schema with churn prediction and recommendations."""

    customer_id: int
    churn_probability: float
    churn_risk_level: str  # LOW, MEDIUM, HIGH
    p_alive: Optional[float] = None
    predicted_clv: Optional[float] = None
    recommendations: List[ProductRecommendation]
    retention_action: str
    model_info: dict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: dict
    version: str
