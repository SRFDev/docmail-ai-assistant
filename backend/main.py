# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Custom modules
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.aws_service import AwsService

from prompts.manager import initialize_prompt_manager, get_prompt_manager
from backend.models import DraftRequest, DraftResponse

# LlamaIndex / Database
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
import chromadb

# Configure Logger
logger = logging.getLogger(__name__)


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("INFO:     Starting DocMail application...")
    
    # 1. Setup logging & Config
    setup_logging(logger_name="docmail", log_level="INFO")
    app.state.config = AppConfig()
    
    # 2. Initialize AWS Service (Singleton)
    logger.info(f"INFO:     Initializing AWS Service (Region: {app.state.config.aws_region})...")
    app.state.aws = AwsService.get_instance(app.state.config)
    
    # 3. Wake up Models (Lazy Load Fix)
    _ = app.state.aws.llm
    _ = app.state.aws.embed_model
    
    try:
        # 4. Connect to ChromaDB (The Vector Store)
        logger.info(f"INFO:     Connecting to ChromaDB at {app.state.config.chroma_persist_dir}...")
        db = chromadb.PersistentClient(path=app.state.config.chroma_persist_dir)
        chroma_collection = db.get_collection(app.state.config.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 5. Load the Index
        # (This was the missing variable in the snippet)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # 6. Load Prompts (New Batch 2 Logic)
        logger.info("INFO:     Loading Prompt Manager...")
        initialize_prompt_manager(app.state.config.prompts_path)
        prompt_manager = get_prompt_manager()
        
        # Retrieve the template string from TOML
        qa_template_str = prompt_manager.get_prompt("docmail", "rag_system_prompt")
        qa_template = PromptTemplate(qa_template_str)
        
        # 7. Create the Query Engine (The RAG "Brain")
        logger.info("INFO:     Building RAG Query Engine...")
        app.state.query_engine = index.as_query_engine(
            similarity_top_k=app.state.config.top_k_retrieval,
            response_mode="compact",
            text_qa_template=qa_template
        )
        logger.info("INFO:     Engine Ready.")
        
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize Vector Store or Prompts: {e}", exc_info=True)
        # We don't exit(1) here to allow the app to start and return 500s (easier to debug)
        app.state.query_engine = None

    yield
    
    # --- SHUTDOWN ---
    logger.info("INFO:     Shutting down DocMail application...")


# Initialize App
app = FastAPI(title="DocMail API", lifespan=lifespan)

# CORS (Allow local frontend)
origins = [
    "http://localhost:3000",
    "http://localhost:8501", # Streamlit default port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    status = "healthy" if app.state.query_engine else "degraded"
    return {"status": status, "service": "docmail-aws"}

@app.post("/generate", response_model=DraftResponse)
async def generate_draft(request: DraftRequest):
    """
    Receives a patient email and returns a RAG-generated draft reply.
    """
    if not app.state.query_engine:
        raise HTTPException(status_code=503, detail="Draft engine is not initialized.")
        
    logger.info(f"Generating draft for input length: {len(request.patient_email)}")
    
    try:
        # Execute RAG Query
        # Note: In LlamaIndex, 'query' is synchronous by default, but we run it in an async path.
        # For heavy loads we'd wrap this, but for MVP it's fine.
        response = app.state.query_engine.query(request.patient_email)
        
        # Extract Sources (for the UI "Transparency" feature)
        sources = []
        for node in response.source_nodes:
            meta = node.metadata
            # Format: [Cardiology - Statin Side Effects]
            info = f"[{meta.get('specialty', 'Gen')} - {meta.get('scenario', 'Unknown')}]"
            sources.append(info)
            
        return DraftResponse(
            draft_reply=str(response),
            source_nodes=sources
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

