from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_initial_view_renders_without_errors():
    app_path = Path(__file__).parents[1] / "frontend" / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "DocuMind"
    assert app.file_uploader[0].label == "PDF, PNG, or JPEG"
    assert app.info[0].value.startswith("Upload an invoice")
