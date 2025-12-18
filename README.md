# Howie AI Assistant

**Live Demo:** [Like to running app](https://howie-ai-assistant.streamlit.app/) 

**Author:** Steven Feinstein | [Dev Site](https://srf.dev/) | [LinkedIn profile](https://www.linkedin.com/in/srfeinstein/)

---

## Overview

Howie is a professional-grade, multimodal AI assistant designed to serve as a comprehensive knowledge base for a specific product. This project demonstrates a complete, end-to-end RAG (Retrieval-Augmented Generation) pipeline built on Google Cloud Platform and the LlamaIndex framework.

The application ingests and synthesizes information from diverse sources, including technical documents (PDFs) and video tutorials, to provide accurate, context-aware, and sourced answers to user questions.

This repository serves as a flagship portfolio project and a reusable template for building robust, production-ready AI applications.

## Key Features & Design Principles

This project was built with a focus on professional software engineering and AI best practices:

*   **Multimodal RAG Pipeline:** Ingests both PDF documents and MP4 video files, extracting structured data from video content using Gemini 2.5 Pro.
*   **IP-Compliant by Design:** The entire knowledge base is built on self-created content to demonstrate a professional, legally-sound development process.
*   **Robust Infrastructure as Code:** The ingestion script uses idempotent "get-or-create" patterns to provision and manage all necessary GCP resources (GCS Buckets, Vertex AI Vector Search Indexes & Endpoints).
*   **Professional Project Structure:** The codebase is organized into a modular, scalable package structure with clear separation of concerns (configuration, core services, data processing, API).
*   **Optimized & Secure Containerization:** The FastAPI backend is packaged into a lean (<1GB) and secure multi-stage Docker image, optimized for deployment on Google Cloud Run.
*   **Comprehensive Debugging & State Management:** The development process involved deep debugging of the AI stack, resulting in a robust architecture that includes a persistent `Docstore` to prevent data inconsistency and a "manual transmission" ingestion method to bypass library limitations.

## Tech Stack

*   **AI/LLM:** LlamaIndex, Google GenAI SDK (Gemini 2.5 Pro, gemini-embedding-001)
*   **Cloud Platform:** Google Cloud Platform (Vertex AI Vector Search, Cloud Run, GCS, Artifact Registry)
*   **Backend:** FastAPI
*   **Frontend:** Streamlit
*   **Containerization:** Docker
*   **Dependency Management:** venv, pip-tools

## Architecture Diagram

<img width="781" height="241" alt="Untitled Diagram drawio (8)" src="https://github.com/user-attachments/assets/fa3a2d43-a9d8-4333-92ba-b9a0bca1b04b" />



## Running the Project Locally

To set up and run this project in your own GCP environment, please follow these steps. All commands should be run from the project's root directory (`howie/`).

1.  **Prerequisites:**
    *   Python 3.12+
    *   Google Cloud SDK (`gcloud` CLI) installed and authenticated (`gcloud auth login`).
    *   A GCP project with billing enabled.
    *   Docker installed.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/SRFDev/howie-ai-assistant.git
    cd howie-ai-assistant
    ```

3.  **Setup Environment & Dependencies:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install dependencies for the ingestion script and local development
    pip install -r requirements-ingest.txt

    # Install the project in editable mode to make local packages importable
    pip install -e .
    ```

4.  **Configure the Application:**
    ```bash
    # Copy the template to create your local config file
    cp config.toml.template config.toml
    ```
    *   Open `config.toml` and replace all placeholder values (e.g., `YOUR_GCP_PROJECT_ID`) with your own.

5.  **Run the Ingestion Pipeline:**
    *   This will provision cloud resources and ingest the data. This may take 20-60 minutes on the first run.
    ```bash
    python -m scripts.ingest
    ```
    *   To start fresh, you can reset all cloud and local data stores:
    ```bash
    python -m scripts.ingest --reset
    ```

6.  **Run the Services:**
    *   **Backend (in one terminal):**
        ```bash
        uvicorn backend.main:app --reload
        ```
    *   **Frontend (in another terminal):**
        ```bash
        streamlit run frontend/app.py
        ```
---
