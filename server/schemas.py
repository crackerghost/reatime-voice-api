"""Validated HTTP request schemas for the Voice API."""

from pydantic import BaseModel, Field, model_validator


def make_schemas(*, default_speed: float, step_min: int, step_max: int, default_step: int):
    class TTSRequest(BaseModel):
        text: str = Field(..., min_length=1, description="Text to speak (Devanagari, Hinglish or English)")
        speed: float = Field(default_speed, ge=0.3, le=2.0)
        nfe_step: int | None = Field(None, ge=step_min, le=step_max)
        nstep: int | None = Field(None, ge=step_min, le=step_max, description="Alias for nfe_step")

        @model_validator(mode="after")
        def resolve_nfe_step(self):
            if self.nstep is not None:
                if self.nfe_step is not None and self.nfe_step != self.nstep:
                    raise ValueError("nfe_step and nstep disagree; send only one")
                self.nfe_step = self.nstep
            if self.nfe_step is None:
                self.nfe_step = default_step
            return self

    class ChatMsg(BaseModel):
        role: str = Field(..., pattern="^(user|assistant)$")
        content: str = Field(..., min_length=1)

    class ChatRequest(BaseModel):
        messages: list[ChatMsg] = Field(..., min_length=1)
        temperature: float = Field(0.7, ge=0.0, le=2.0)

    class VisionRequest(BaseModel):
        image: str = Field(..., min_length=32, description="JPEG screenshot, base64 (data: prefix optional)")
        hash: str = Field("", description="Client-side change-detection id (cache key)")
        force: bool = Field(False, description="True = skip cache/coalescing and describe THIS frame now")

    return TTSRequest, ChatMsg, ChatRequest, VisionRequest
