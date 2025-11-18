#!/usr/bin/env python
"""Entry point for Gradio web UI."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_news.gradio_ui import create_gradio_app

if __name__ == "__main__":
    create_gradio_app()

