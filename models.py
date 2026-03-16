from typing import Optional, List
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    review_text: str = Field(..., description="Text to analyze for sentiment and vectors")
    review_id: Optional[str] = Field(default=None, description="Optional review ID for caching")


class InferenceResult(BaseModel):
    internalReviewId: Optional[str] = None
    sentimentLabel: str = Field(..., description="Sentiment: positive, negative, or neutral")
    score: float = Field(..., description="Confidence score for the sentiment")
    vectors: str = Field(..., description="Comma-separated list of detected vectors")


class BulkInferenceRequest(BaseModel):
    review_ids: Optional[List[str]] = Field(default=None, description="List of review IDs to analyze")
    review_texts: Optional[List[str]] = Field(default=None, description="List of review texts to analyze")


class ReviewInfo(BaseModel):
    internalReviewId: str
    reviewText: str


class HealthResponse(BaseModel):
    status: str
    cache_size: int


class StartJobResponse(BaseModel):
    job_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    cached: int
    errors: int
