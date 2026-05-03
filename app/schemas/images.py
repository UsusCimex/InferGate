from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HighresFixSpec(BaseModel):
    """Two-pass high-resolution generation spec (base → upscale → img2img refine)."""
    model_config = ConfigDict(extra="forbid")

    scale: float = Field(2.0, gt=1.0, le=4.0)
    denoising_strength: float = Field(0.5, ge=0.0, le=1.0)
    steps: int | None = Field(None, ge=1, le=150)
    upscaler: str = Field("lanczos", pattern=r"^(nearest|bilinear|bicubic|lanczos)$")


class TextualInversionSpec(BaseModel):
    """Textual-inversion embedding to register in the pipeline's tokenizer."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
    token: str | list[str] | None = None
    weight_file: str | None = Field(None, max_length=200)


class LoraSpec(BaseModel):
    """LoRA adapter descriptor — repo id plus activation weight."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
    weight: float = Field(1.0, ge=-3.0, le=3.0)
    weight_file: str | None = Field(None, max_length=200)
    adapter_name: str | None = Field(None, max_length=64)


class ImageGenerationRequest(BaseModel):
    """Request body for /v1/images/generations."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    prompt: str = Field(..., min_length=1, max_length=10000)
    n: int = Field(1, ge=1, le=10)
    size: str | None = None
    response_format: str = "b64_json"
    seed: int | None = None
    negative_prompt: str | None = Field(None, max_length=10000)
    num_inference_steps: int | None = Field(None, ge=1, le=150)
    guidance_scale: float | None = Field(None, ge=0.0, le=30.0)
    scheduler: str | None = Field(None, max_length=50)
    loras: list[LoraSpec] | None = None
    textual_inversions: list[TextualInversionSpec] | None = None
    highres_fix: HighresFixSpec | None = None

    image: str | None = Field(None, max_length=20_000_000)
    mask: str | None = Field(None, max_length=20_000_000)
    denoising_strength: float | None = Field(None, ge=0.0, le=1.0)
    refiner_switch_at: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("loras")
    @classmethod
    def _cap_loras(cls, v: list[LoraSpec] | None) -> list[LoraSpec] | None:
        if v is not None and len(v) > 5:
            raise ValueError("loras: at most 5 adapters per request")
        return v

    @field_validator("textual_inversions")
    @classmethod
    def _cap_tis(cls, v: list[TextualInversionSpec] | None) -> list[TextualInversionSpec] | None:
        if v is not None and len(v) > 10:
            raise ValueError("textual_inversions: at most 10 embeddings per request")
        return v

    @model_validator(mode="after")
    def _mask_requires_image(self) -> ImageGenerationRequest:
        if self.mask is not None and self.image is None:
            raise ValueError("mask requires image: inpainting needs a base image to modify")
        return self


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
