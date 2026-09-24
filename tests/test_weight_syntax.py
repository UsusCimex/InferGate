from app.providers.image._compel import has_weight_syntax, strip_weight_syntax


def test_positive_weights_keep_their_phrase():
    prompt = "(A single bat:1.4), a baseball item, on a (pure white background:1.3)"
    assert strip_weight_syntax(prompt) == "A single bat, a baseball item, on a pure white background"


def test_non_positive_weights_drop_their_phrase():
    assert strip_weight_syntax("a cat, (dog:-1.2), (bird:0)") == "a cat, , "


def test_colons_inside_the_phrase_survive():
    assert strip_weight_syntax("(ratio 16:9 frame:1.2)") == "ratio 16:9 frame"


def test_plain_prompts_are_untouched():
    for prompt in ["a cat on a mat", "a cat (sitting) on a mat", "time 12:30"]:
        assert strip_weight_syntax(prompt) == prompt


def test_stripped_prompt_has_no_weight_syntax():
    assert not has_weight_syntax(strip_weight_syntax("(tack sharp focus:1.2), (bat:1.4)"))
