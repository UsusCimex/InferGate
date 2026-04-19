from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LoraSpec(BaseModel):
    """Single LoRA adapter to apply during generation.

    - `id`: HuggingFace repo identifier (`user/repo-name`). Must match the
      model architecture of the target pipeline (an SDXL LoRA will not load
      into an SD 1.5 pipeline).
    - `weight`: scalar adapter strength. Negative values are legal and
      produce an anti-LoRA effect; 0.0 is a no-op.
    - `weight_file`: optional specific `.safetensors` filename inside the
      repo when it ships multiple (e.g. Redmond-style packs).
    - `adapter_name`: optional stable label for the adapter slot in the
      pipeline. Auto-assigned if omitted; supply a value if you want to
      pin a specific slot across requests.
    """
    id: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
    weight: float = Field(1.0, ge=-3.0, le=3.0)
    weight_file: str | None = Field(None, max_length=200)
    adapter_name: str | None = Field(None, max_length=64)


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
    # Per-request LoRA adapters, applied on top of the loaded pipeline.
    # Empty/None means no LoRAs active (any previously cached adapters are
    # deactivated for this request). Provider enforces a per-model cap and
    # reuses cached adapters across requests to avoid re-downloading.
    loras: list[LoraSpec] | None = None

    @field_validator("loras")
    @classmethod
    def _cap_loras(cls, v: list[LoraSpec] | None) -> list[LoraSpec] | None:
        if v is not None and len(v) > 5:
            raise ValueError("loras: at most 5 adapters per request")
        return v


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
