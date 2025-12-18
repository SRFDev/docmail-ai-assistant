# Copyright © 2025 SRF Development, Inc. All rights reserved.
#
# This file is part of the "DocMail" project.
#
# This project is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by the Open Source
# Initiative.
#
# This project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with this project.
# If not, see <https://opensource.org/licenses/MIT>.
#
# SPDX-License-Identifier: MIT
import logging
import json
import os
import argparse
from pathlib import Path

# Local imports
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.aws_service import AwsService

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def load_style_data(file_path: str):
    """
    Reads the physician style JSONL file and converts it to LlamaIndex Documents.
    Each line is expected to be a JSON object with 'physician_reply' and metadata.
    """
    documents = []
    path = Path(file_path)
    
    if not path.exists():
        logging.warning(f"Data file not found at {path}. Skipping load.")
        return []

    logging.info(f"Loading data from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                # We want to index the Physician's Reply text
                text = record.get("physician_reply", "")
                
                # We can store other fields as metadata for filtering later
                metadata = {
                    "specialty": record.get("physician_persona", {}).get("specialty", "General"),
                    "tone": str(record.get("physician_persona", {}).get("tone", [])),
                    "scenario": record.get("medical_scenario", {}).get("topic", "")
                }
                
                if text:
                    doc = Document(text=text, metadata=metadata, id_=f"style_{i}")
                    documents.append(doc)
            except json.JSONDecodeError:
                logging.error(f"Skipping invalid JSON on line {i}")
                
    logging.info(f"Loaded {len(documents)} documents.")
    return documents


def ingest(config: AppConfig, reset: bool = False):
    # 1. Initialize AWS Service (to bind LLM/Embeddings to Settings)
    aws = AwsService.get_instance(config)

    # Force initialization of models to update global Settings
    logging.info("Waking up Bedrock models...")
    _ = aws.llm         # Triggers lazy load & Settings.llm assignment
    _ = aws.embed_model # Triggers lazy load & Settings.embed_model assignment

    # 2. Setup ChromaDB (Local)
    logging.info(f"Initializing ChromaDB at {config.chroma_persist_dir}...")
    db = chromadb.PersistentClient(path=config.chroma_persist_dir)
    
    if reset:
        logging.info(f"Resetting collection: {config.collection_name}")
        try:
            db.delete_collection(config.collection_name)
        except Exception:
            pass # Collection might not exist
            
    chroma_collection = db.get_or_create_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load Data
    documents = load_style_data(config.physician_style_path)
    
    if not documents:
        logging.warning("No documents to ingest. Make sure you run the synthetic data notebook first!")
        return

    # 4. Create/Update Index
    logging.info("Creating embeddings and building index...")
    # This automatically calls Bedrock Embedding (via Settings) and stores in Chroma
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    
    logging.info("✅ Ingestion Complete.")

if __name__ == "__main__":
    setup_logging(logger_name="docmail", log_level="INFO")
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Ingest DocMail data.")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store")
    args = parser.parse_args()

    config = AppConfig()
    ingest(config, reset=args.reset)

