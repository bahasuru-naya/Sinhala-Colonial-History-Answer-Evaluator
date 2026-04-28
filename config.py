"""
Configuration for Sinhala Answer Evaluator System
"""
import os
from pathlib import Path

# Disable ChromaDB telemetry to prevent posthog errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge"
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
UI_DIR = BASE_DIR / "ui"

# OLLAMA Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b  "  # Using gemma4 e2b for Sinhala support
OLLAMA_TIMEOUT = 300

# ChromaDB Configuration
CHROMADB_PATH = str(DATA_DIR / "chromadb")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50

# RAG Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.8

# Questions and Marking Guides
QUESTIONS_FILE = DATA_DIR / "questions.json"
MARKING_GUIDES_FILE = DATA_DIR / "marking_guides.json"

# Ontology Configuration
ONTOLOGY_FILE = DATA_DIR / "ontology.ttl"
ONTOLOGY_NS = "http://example.org/colonial-sri-lanka/"

# Scoring Configuration
MAX_MARKS = 20
MIN_MARKS = 0

# Agent Configuration
AGENT_TIMEOUT = 120
MAX_RETRIES = 3

# Logging
LOG_LEVEL = "INFO"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# UI Configuration
STREAMLIT_THEME = "light"
PAGE_ICON = str(UI_DIR / "img" / "logo.png")
PAGE_TITLE = "Sinhala Colonial History Answer Evaluator"
