"""
Smoke test: every dashboard page renders without raising an exception.
Uses Streamlit's built-in AppTest (no browser needed).
"""

import pytest

from src import config

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(config.PROJECT_ROOT / "app.py")
PAGES = ["Project Overview", "Exploratory Data Analysis", "Model Evaluation",
         "Churn Prediction", "Retention Recommendations"]


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained")
def test_prediction_form_submits():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Churn Prediction").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(e.value for e in at.error) + \
        " ".join(s.value for s in at.success)
    assert "Prediction:" in text
