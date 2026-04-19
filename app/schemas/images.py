from __future__ import annotations

from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1, max_length=10000)
    n: int = Field(1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: int | None = None
    # Per-request sampling overrides. When None the YAML default (or the
    # pipeline default if the YAML is silent) is used. Ranges are liberal
    # because different models have very different sweet spots — the YAML
    # is the authoritative source of "sensible defaults per model".
    negative_prompt: str | None = Field(None, max_length=10000)
    num_inference_steps: int | None = Field(None, ge=1, le=150)
    guidance_scale: float | None = Field(None, ge=0.0, le=30.0)
    # Short-name scheduler override — euler / euler_a / dpm++_2m /
    # dpm++_2m_karras / ddim / lms / heun / pndm / unipc / … See the
    # _SCHEDULERS map in DiffusersImageProvider for the full list.
    # FlowMatch pipelines (FLUX, SD3.x) have their own scheduler family
    # and will reject these names — match scheduler to model architecture.
    scheduler: str | None = Field(None, max_length=50)


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
