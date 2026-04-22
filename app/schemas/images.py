from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class HighresFixSpec(BaseModel):
    """Two-pass generation: compose at the request's `size`, upscale, then
    img2img-refine at higher resolution. Produces sharper detail than a
    single-pass generate at the target resolution (SDXL base tends to
    duplicate features above ~1024px natively).

    Pipeline runs:
      1. Generate at request.width × request.height (typically 512-1024)
      2. PIL-resize to (w*scale, h*scale) using `upscaler` resampling
      3. Img2img pass with `denoising_strength` (0.0=no change, 1.0=redraw)
    """
    scale: float = Field(2.0, gt=1.0, le=4.0)
    denoising_strength: float = Field(0.5, ge=0.0, le=1.0)
    steps: int | None = Field(None, ge=1, le=150)  # override for pass 2
    upscaler: str = Field("lanczos", pattern=r"^(nearest|bilinear|bicubic|lanczos)$")


class TextualInversionSpec(BaseModel):
    """Single Textual Inversion (aka embedding) to register with the tokenizer.

    - `id`: HuggingFace repo identifier (`user/repo-name`).
    - `token`: explicit trigger string to register the embedding under.
      Accepts either a single string or a list of strings — multi-token
      TIs (e.g. SDXL pivotal embeddings like `<s0><s1>`) need a list.
      If omitted, diffusers derives it from the filename.
    - `weight_file`: optional `.pt` / `.safetensors` file inside the repo
      when it ships multiple embeddings.

    TIs have no `weight` parameter — emphasis is applied via the prompt
    itself (`<token>`, or compel syntax `(<token>:1.5)`).
    """
    id: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
    token: str | list[str] | None = None
    weight_file: str | None = Field(None, max_length=200)


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
    # Per-request Textual Inversion embeddings. Cheap (~KB each) so the
    # per-request cap is loose. Once loaded they persist for the lifetime
    # of the provider — subsequent requests skip reload.
    textual_inversions: list[TextualInversionSpec] | None = None
    # Two-pass high-resolution generation (compose → upscale → refine).
    # When set, the `size` field becomes the base-pass resolution and the
    # final image is `size * scale`.
    highres_fix: HighresFixSpec | None = None

    # img2img / inpainting: base64 PNG/JPEG (data: URI prefix also accepted).
    # image only → img2img; image + mask → inpaint; mask alone rejected.
    image: str | None = Field(None, max_length=20_000_000)  # ~15MB base64
    mask: str | None = Field(None, max_length=20_000_000)
    # Strength for img2img/inpaint (unset → pipeline default, usually 0.8).
    denoising_strength: float | None = Field(None, ge=0.0, le=1.0)

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
