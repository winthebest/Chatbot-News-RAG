#!/usr/bin/env python
"""Entry point for RAG chatbot."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_news.chatbot import main

if __name__ == "__main__":
    main()

