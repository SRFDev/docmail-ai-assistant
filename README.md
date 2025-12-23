# DocMail AI Assistant

**A fully functional, proof-of-concept AI physician's assistant that drafts patient email replies based on a doctor's specific communication style and medical guidelines.**

DocMail demonstrates a professional-grade RAG (Retrieval-Augmented Generation) pipeline built on AWS, designed to handle sensitive communications with "Safety First" architectural principles.

## Project Overview

*   **Mission:** Assist physicians by drafting empathetic, medically safe, and stylistically accurate email replies.
*   **Core Capability:** Uses synthetic data generation to model specific physician personas (tone, brevity, disclaimer usage) and retrieves them to ground the LLM's response.
*   **Architecture:** AWS-Native.

## Technical Stack

*   **Language:** Python 3.12
*   **Cloud Provider:** AWS
*   **LLM:** Anthropic Claude 3.5 Sonnet (via Amazon Bedrock)
*   **Embeddings:** Amazon Titan Text v2 (via Amazon Bedrock)
*   **Vector Store:** ChromaDB (Local/Containerized)
*   **Backend:** FastAPI
*   **Frontend:** Streamlit

## Key Features

1.  **Synthetic Data Engine:** A custom pipeline (`scripts/generate_dataset.py`) that uses LLMs to generate high-quality, privacy-compliant training data (Patient/Doctor email pairs).
2.  **Style-Aware RAG:** Retrieves physician past replies based on semantic similarity to the current patient scenario.
3.  **Safety Guardrails:** Prompt engineering designed to prevent diagnosis over email and flag urgent situations (e.g., Statin side effects).

## Setup & Installation

1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Configure AWS credentials (requires Bedrock access).
4.  Generate data: `python -m scripts.generate_dataset`
5.  Ingest vector index: `python -m scripts.ingest`
6.  Run API: `uvicorn backend.main:app --reload`

