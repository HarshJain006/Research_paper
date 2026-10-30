import html
import os
from pathlib import Path

import streamlit as st


def render_app() -> None:
    st.set_page_config(
        page_title="Marginal — AI Research Paper Explainer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    try:
        backend_url = st.secrets.get("BACKEND_URL", backend_url)
    except FileNotFoundError:
        pass

    index_path = Path(__file__).with_name("index.html")
    page_html = index_path.read_text(encoding="utf-8")
    page_html = page_html.replace(
        'value="http://localhost:8000"',
        f'value="{html.escape(backend_url, quote=True)}"',
        1,
    )

    if backend_url == "http://localhost:8000":
        st.warning(
            "Set BACKEND_URL in Streamlit Cloud App settings > Secrets to your "
            "deployed FastAPI backend URL. The default localhost address only "
            "works when running the backend on your own computer."
        )

    st.iframe(page_html, height=1200)


if __name__ == "__main__":
    render_app()
