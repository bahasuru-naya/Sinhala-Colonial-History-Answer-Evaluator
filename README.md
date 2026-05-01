# 🇱🇰 Sinhala Colonial History Answer Evaluator

An advanced, AI-powered system designed to automatically evaluate and score student answers on Sri Lankan Colonial History (Portuguese, Dutch, and British periods) in the Sinhala language.

## 🌟 Overview

The **Sinhala Colonial History Answer Evaluator** is a sophisticated educational tool that leverages state-of-the-art AI techniques to provide objective, detailed, and context-aware grading. By combining Retrieval-Augmented Generation (RAG) with formal Ontologies, the system ensures that evaluations are grounded in historical facts and adhere to specific marking rubrics.

## ✨ Key Features

- **🎯 Intelligent Scoring:** Automated grading out of 20 marks based on predefined criteria.
- **📚 Hybrid Architecture:** Combines RAG (for factual retrieval) with Ontology (for semantic understanding of historical concepts).
- **🗣️ Native Sinhala Support:** Specifically optimized for the Sinhala language using the Gemma model via Ollama.
- **📉 Detailed Feedback:** Provides strengths, gaps, and specific justifications for awarded marks.
- **💎 Premium UI:** A modern, responsive Streamlit interface with rich aesthetics and real-time processing visualization.
- **🏛️ Historical Scope:** Covers the Portuguese, Dutch, and British colonial eras in Sri Lanka.

## 🏗️ System Architecture

The system follows a multi-layered agentic approach:

1.  **UI Layer:** Built with Streamlit, providing an interactive dashboard for students and educators.
2.  **Orchestration Layer:** Managed by `AnswerEvaluationOrchestrator` using LangGraph to coordinate between retrieval and evaluation.
3.  **Knowledge Layer:**
    -   **RAG Pipeline:** Uses ChromaDB for vector-based retrieval of historical text chunks.
    -   **Ontology Handler:** Uses RDFLib to query a structured Knowledge Graph of colonial entities and relationships.
4.  **Inference Layer:** Powered by **Ollama (Gemma model)** for natural language understanding and generation in Sinhala.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running.
- Gemma model pulled in Ollama:
  ```bash
  ollama pull gemma
  ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bahasuru-naya/Sinhala-Colonial-History-Answer-Evaluator.git
    cd Sinhala-Colonial-History-Answer-Evaluator
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

You can start the application using the provided scripts:

**Windows:**
```bash
./run_app.bat
```

**Linux/Mac:**
```bash
./run_app.sh
```

Alternatively, run directly via Streamlit:
```bash
streamlit run ui/app.py
```

## 📂 Project Structure

- `ui/`: Main Streamlit application and assets.
- `utils/`: Core logic for RAG, Ontology, and Ollama integration.
- `data/`: Knowledge base, vector store, and marking guides.
- `models/`: Agentic workflow and evaluation logic.
- `config.py`: System-wide configuration and paths.

## 🛠️ Technologies Used

- **Framework:** Streamlit
- **LLM Engine:** Ollama (Gemma)
- **RAG:** LangChain, ChromaDB, Sentence-Transformers
- **Ontology:** RDFLib (Turtle format)
- **Logic:** LangGraph
- **Language:** Python

---

© 2026 Sinhala Answer Evaluator Project. Developed for Educational Excellence.
