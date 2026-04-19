# Batch-build InferGate workers in parallel:
#   docker buildx bake -f deploy/docker-bake.hcl
#
# Build a single model:
#   docker buildx bake -f deploy/docker-bake.hcl worker-qwen-image
#
# Push to a registry:
#   REGISTRY=myreg.io/infergate TAG=v1 \
#     docker buildx bake -f deploy/docker-bake.hcl --push
#
# Adding a new model:
#   1. Create deploy/workers/<id>/requirements.txt
#   2. Add one tuple to the `item` matrix below
#   3. Add a service stanza in docker-compose.yml (~10 lines)
#
# Build-arg semantics are documented in deploy/Dockerfile.worker.

variable "REGISTRY" { default = "infergate" }
variable "TAG"      { default = "latest" }

variable "GPU_BASE_IMAGE" {
  default = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
}
variable "VLLM_IMAGE" {
  default = "vllm/vllm-openai:v0.19.0"
}
variable "CPU_BASE_IMAGE" {
  default = "python:3.12-slim"
}

group "default" {
  targets = ["worker"]
}

target "worker" {
  name       = "worker-${replace(item.id, ".", "-")}"
  context    = ".."
  dockerfile = "deploy/Dockerfile.worker"
  tags       = ["${REGISTRY}/worker-${replace(item.id, ".", "-")}:${TAG}"]
  args = {
    BASE_IMAGE          = item.base
    APT_PACKAGES        = item.apt
    WORKER_REQUIREMENTS = "deploy/workers/${item.id}/requirements.txt"
    POST_INSTALL        = item.post
  }
  matrix = {
    item = [
      # ─── Text (vLLM) ─────────────────────────────────────────────────
      { id = "qwen3.5-4b",        base = VLLM_IMAGE,     apt = "",                post = "" },
      { id = "qwen3.5-9b",        base = VLLM_IMAGE,     apt = "",                post = "" },
      { id = "qwen3-8b",          base = VLLM_IMAGE,     apt = "",                post = "" },
      { id = "llama3.1-8b",       base = VLLM_IMAGE,     apt = "",                post = "" },
      # ─── Image (diffusers) ───────────────────────────────────────────
      { id = "sd35-medium",       base = GPU_BASE_IMAGE, apt = "",                post = "" },
      { id = "sdxl-base",         base = GPU_BASE_IMAGE, apt = "",                post = "" },
      { id = "hunyuan-dit",       base = GPU_BASE_IMAGE, apt = "",                post = "" },
      { id = "flux1-dev",         base = GPU_BASE_IMAGE, apt = "",                post = "" },
      { id = "flux1-schnell",     base = GPU_BASE_IMAGE, apt = "build-essential", post = "" },
      { id = "flux2-klein-4b",    base = GPU_BASE_IMAGE, apt = "",                post = "" },
      { id = "qwen-image",        base = GPU_BASE_IMAGE, apt = "build-essential", post = "" },
      { id = "z-image-turbo",     base = GPU_BASE_IMAGE, apt = "git build-essential", post = "" },
      # ─── Autoregressive (non-diffusion) ──────────────────────────────
      { id = "janus-pro-1b",      base = GPU_BASE_IMAGE, apt = "git",             post = "" },
      { id = "janus-pro-7b",      base = GPU_BASE_IMAGE, apt = "git build-essential", post = "" },
      # ─── TTS ─────────────────────────────────────────────────────────
      { id = "kokoro-82m",        base = CPU_BASE_IMAGE, apt = "gcc",             post = "python -m spacy download en_core_web_sm" },
      { id = "openaudio-s1-mini", base = GPU_BASE_IMAGE, apt = "git",             post = "" },
    ]
  }
}
