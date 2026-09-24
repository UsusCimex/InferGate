from app.providers.image._lora import LoraCache


class _PipeWithoutPeft:
    def __init__(self):
        self.disable_calls = 0

    def disable_lora(self):
        self.disable_calls += 1
        raise ValueError("PEFT backend is required for this method.")


def test_no_loras_on_fresh_pipeline_does_not_touch_peft():
    pipe = _PipeWithoutPeft()
    LoraCache("z-image-turbo").apply(pipe, None, {}, "/models")
    LoraCache("z-image-turbo").apply(pipe, [], {}, "/models")
    assert pipe.disable_calls == 0


def test_no_loras_after_adapters_were_loaded_disables_them():
    class _Pipe:
        disabled = False

        def disable_lora(self):
            self.disabled = True

    pipe = _Pipe()
    cache = LoraCache("sdxl-base")
    cache._cache[("some/lora", None)] = "adapter_0"
    cache.apply(pipe, None, {}, "/models")
    assert pipe.disabled
