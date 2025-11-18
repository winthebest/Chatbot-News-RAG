# RAG News - Vietnamese News RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for Vietnamese news articles from VNExpress. This project enables intelligent question-answering over news content by combining web scraping, bilingual text processing, vector embeddings, and local LLM integration.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Purpose](#project-purpose)
- [File Structure & Functions](#file-structure--functions)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Workflow](#project-workflow)
- [Configuration](#configuration)

---

## Project Overview

**RAG News** is an end-to-end RAG pipeline that:

1. **Crawls** Vietnamese news articles from VNExpress (AI/Technology section)
2. **Processes** articles in both Vietnamese and English (automatic translation)
3. **Chunks** articles into semantic segments with overlap for better context
4. **Embeds** text chunks using BAAI/bge-m3 multilingual embedding model
5. **Stores** embeddings in Qdrant vector database for fast similarity search
6. **Retrieves** relevant context for user queries
7. **Generates** answers using local LLM (Ollama) with retrieved context

### Key Features

- **Bilingual Support**: Processes articles in both Vietnamese and English
- **Semantic Search**: Uses state-of-the-art BGE-M3 embeddings (1024 dimensions)
- **Re-ranking**: Optional Cross-Encoder re-ranking for improved relevance
- **Vector Database**: Qdrant for efficient similarity search
- **Local LLM**: Integrates with Ollama for privacy-preserving inference
- **Web UI**: Gradio-based web interface for easy interaction
- **Automated Pipeline**: Complete workflow from crawling to chatbot
- **Progress Tracking**: Real-time progress bars and status updates

---

## Project Purpose

This project is designed to:

1. **Demonstrate RAG Architecture**: Showcase a complete RAG system implementation
2. **Vietnamese NLP**: Address the need for Vietnamese language RAG systems
3. **Local AI**: Enable private, local AI inference without cloud dependencies
4. **News Intelligence**: Provide intelligent Q&A over news content
5. **Educational**: Serve as a learning resource for RAG, embeddings, and vector databases
6. **Production-Ready**: Structured codebase suitable for extension and deployment


---


## File Structure & Functions

### Project Structure

```
RAG_News/
├── data/
│   ├── raw/                    # Raw crawled data
│   │   ├── vnexpress_links.csv
│   │   └── vnexpress_articles.json
│   └── processed/              # Processed chunks
│       ├── vnexpress_chunks_vi.jsonl
│       └── vnexpress_chunks_en.jsonl
│
├── news_crawler/               # Web scraping module
│   ├── crawl_pipeline.py       # Main crawl script (all-in-one)
│   ├── url_crawler.py          # URL collection (legacy)
│   ├── content_crawler.py      # Content extraction (legacy)
│   ├── prepare_data_rag.py     # Chunking logic (legacy)
│   └── helper_functions.py     # Selenium utilities
│
├── src/rag_news/               # RAG core module
│   ├── embeddings.py           # BGE-M3 embedding functions
│   ├── qdrant.py              # Qdrant connection & utilities
│   ├── reranker.py            # Cross-encoder re-ranking model
│   ├── ingest.py              # Embed & ingest to Qdrant
│   ├── query.py               # Query/search interface
│   ├── retriever.py           # RAG retrieval logic
│   ├── chatbot.py             # Interactive CLI chatbot
│   └── gradio_ui.py           # Gradio web interface
│
├── scripts/                     # Entry point scripts
│   ├── crawl_pipeline.py       # Run crawl pipeline
│   ├── ingest_chunks.py        # Run ingestion
│   ├── query_qdrant.py         # Run query
│   ├── rag_chatbot.py          # Run CLI chatbot
│   └── gradio_ui.py            # Run Gradio web UI
│
└── requirements.txt            # Python dependencies
```


---

## Installation

### Prerequisites

- Python 3.8+
- Chrome/Chromium browser (for Selenium)
- Qdrant server (local or remote)
- Ollama (for local LLM)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd RAG_News
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Qdrant

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Follow Qdrant installation guide
# https://qdrant.tech/documentation/guides/installation/
```

### Step 4: Install Ollama

```bash
# Download from https://ollama.ai
# Or use package manager
```

### Step 5: Pull Ollama Model

```bash
ollama pull qwen2.5:7b
```

---


## Project Workflow

### Complete End-to-End Workflow

```bash
# Step 1: Crawl and prepare data
python scripts/crawl_pipeline.py --step all --start-page 1 --end-page 10

# Step 2: Ingest chunks to Qdrant
python scripts/ingest_chunks.py --collection vnexpress_news --include-text

# Step 3: Test query
python scripts/query_qdrant.py --query "Test question" --top-k 5

# Step 4: Run chatbot (CLI)
python scripts/rag_chatbot.py

# Or run web UI (Gradio)
python scripts/gradio_ui.py
```


## Usage Guide

### Method 1: Web UI (Gradio) - Recommended for End Users

The easiest way to interact with the RAG system is through the Gradio web interface.

#### Quick Start:
```bash
python scripts/gradio_ui.py
```

Then open your browser and navigate to: http://localhost:7860

#### Features:
- **Interactive Chat**: Type questions and get answers with conversation history
- **Settings Panel**: Adjust Top-K, re-ranking, and initial candidates in real-time
- **Source Display**: View referenced articles with scores and clickable URLs
- **Examples**: Try pre-loaded example questions
- **Mobile Support**: Responsive design works on mobile/tablet

#### Advanced Options:
```bash
# Run on custom port
python scripts/gradio_ui.py --server-port 8080

# Use CPU instead of GPU
python scripts/gradio_ui.py --device cpu

# Create public share link
python scripts/gradio_ui.py --share

# Disable re-ranking for faster responses
python scripts/gradio_ui.py --no-reranker

# Use different Ollama model
python scripts/gradio_ui.py --ollama-model llama2:7b
```

### Method 2: CLI Chatbot

For terminal-based interaction:

```bash
python scripts/rag_chatbot.py
```

Type your questions and press Enter. Type `exit` or `quit` to stop.

### Method 3: Query Script

For quick one-off queries:

```bash
python scripts/query_qdrant.py --query "Your question here" --top-k 5
```

With re-ranking:
```bash
python scripts/query_qdrant.py \
  --query "Your question" \
  --top-k 5 \
  --use-reranker \
  --initial-candidates 20
```

---

## Configuration

### Environment Variables (Optional)

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_API_KEY=your_key_here
export OLLAMA_MODEL=qwen2.5:7b
```

### Default Settings

- **Embedding Model**: `BAAI/bge-m3`
- **Reranker Model**: `BAAI/bge-reranker-base` (optional)
- **Vector Size**: 1024
- **Batch Size**: 128
- **Chunk Size**: 900 characters
- **Overlap**: 200 characters
- **Top-K**: 5 results
- **Initial Candidates** (for re-ranking): top_k * 3 (default: 15)
- **Qdrant Host**: localhost
- **Qdrant Port**: 6333
- **Ollama Model**: qwen2.5:7b

---

## Contact

If you have any questions or concerns, please contact me via p.votrongtien@gmail.com or https://www.facebook.com/neitrong.20/

---


