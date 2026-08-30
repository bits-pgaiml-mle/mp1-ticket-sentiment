from typing import Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw ticket or review text")
    channel: Literal["email", "chat", "app"] = Field(default="app")


class TextResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
