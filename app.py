"""TalkTable web app and remote MCP server (the Hugging Face Space entry point).

    .venv/bin/python app.py

Chat UI:     http://127.0.0.1:7860
MCP server:  http://127.0.0.1:7860/gradio_api/mcp   (streamable HTTP)

Configuration comes from environment variables or `.env` (see .env.example). On a
Space, set them as secrets: POSTGRES_URL, MONGO_URL, LLM_PROVIDER, LLM_MODEL, and
HF_TOKEN or GROQ_API_KEY.
"""

from __future__ import annotations

import logging
import sys

from talktable.config import get_settings
from talktable.pipeline import get_talktable
from talktable.webui import CSS, FORCE_DARK_JS, build_app, build_theme

logger = logging.getLogger("talktable.app")


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub", "mcp.server.lowlevel.server"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Load the embedding model and connect to the databases once, before serving.
    engine = get_talktable()
    logger.info("online databases: %s", ", ".join(engine.online) or "none")

    build_app().launch(
        mcp_server=True,
        theme=build_theme(),
        css=CSS,
        js=FORCE_DARK_JS,
        ssr_mode=False,  # no Node.js server process: less memory on a free CPU Space
        footer_links=["api", "gradio"],
    )


if __name__ == "__main__":
    main()
