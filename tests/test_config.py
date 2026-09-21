from documind.config import Settings


def test_blank_optional_cost_settings_are_treated_as_unset(monkeypatch):
    monkeypatch.setenv("DOCUMIND_OPENAI_INPUT_COST_PER_MILLION", "")
    monkeypatch.setenv("DOCUMIND_OPENAI_OUTPUT_COST_PER_MILLION", "")

    settings = Settings(_env_file=None)

    assert settings.openai_input_cost_per_million is None
    assert settings.openai_output_cost_per_million is None
