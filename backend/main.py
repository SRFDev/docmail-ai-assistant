# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Custom modules
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.aws_service import AwsService
from core.runpod_service import RunPodService 

from prompts.manager import initialize_prompt_manager, get_prompt_manager
from backend.models import DraftRequest, DraftResponse

# LlamaIndex / Database
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
import chromadb

# Configure Logger
logger = logging.getLogger(__name__)

# --- Lifespan Manager (The Brain Factory) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    setup_logging(logger_name="docmail", log_level="INFO")
    logger.info("INFO:     Starting DocMail application...")
    
    # 1. Load Config
    try:
        app.state.config = AppConfig()
    except Exception as e:
        logger.critical(f"Config Load Failed: {e}")
        raise e
    
    # 2. Initialize AWS Service (ALWAYS REQUIRED)
    # Why? We need it for 'embed_model' (Titan v2) to talk to ChromaDB, 
    # even if we are using RunPod for text generation.
    try:
        logger.info(f"INFO:     Initializing AWS Infrastructure (Region: {app.state.config.aws_region})...")
        app.state.aws = AwsService.get_instance(app.state.config)
    except Exception as e:
        logger.error(f"AWS Init Failed (Embeddings will be unavailable): {e}")
        app.state.aws = None

    # 3. Initialize RunPod Service (Always load if keys exist)
    try:
        if app.state.config.runpod_api_key and app.state.config.runpod_endpoint_id:
            logger.info("INFO:     Initializing RunPod Service...")
            app.state.runpod = RunPodService(
                api_key=app.state.config.runpod_api_key,
                endpoint_id=app.state.config.runpod_endpoint_id
            )
        else:
            logger.warning("WARNING: RunPod credentials missing. Service unavailable.")
            app.state.runpod = None
    except Exception as e:
        logger.error(f"RunPod Init Failed: {e}")
        app.state.runpod = None

    # 4. Connect to Vector Store (RAG)
    try:
        logger.info(f"INFO:     Connecting to ChromaDB at {app.state.config.chroma_persist_dir}...")
        db = chromadb.PersistentClient(path=app.state.config.chroma_persist_dir)
        chroma_collection = db.get_collection(app.state.config.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load Index using AWS Embeddings
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
            embed_model=app.state.aws.embed_model
        )
        app.state.index = index
        logger.info("INFO:     Vector Store Ready.")
        
    except Exception as e:
        logger.warning(f"WARNING: Vector Store/RAG functionality degraded: {e}")
        app.state.index = None

    # 5. Load Prompts
    initialize_prompt_manager(app.state.config.prompts_path)

    yield
    
    # --- SHUTDOWN ---
    logger.info("INFO:     Shutting down DocMail application...")


# Initialize App
app = FastAPI(title="DocMail API", lifespan=lifespan)

# CORS
origins = ["http://localhost:3000", "http://localhost:8501"]
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
    rag_status = "active" if app.state.index else "inactive"
    return {
        "status": "healthy", 
        "llm_mode": app.state.config.llm_source,
        "rag_status": rag_status
    }

@app.post("/generate", response_model=DraftResponse)
async def generate_draft(request: DraftRequest):
    """
    Receives a patient email and returns a draft reply.
    Uses the configured LLM Provider (RunPod or Bedrock).
    """
    logger.info(f"Generating draft for input length: {len(request.patient_email)}")

    # 1. Determine Strategy
    # Priority: API Request > Config Default > Default to 'bedrock'
    requested_source = request.model_source or app.state.config.llm_source
    requested_source = requested_source.lower()
    
    # Fetch Prompts (Strategy: Use System Prompt from TOML)
    prompt_manager = get_prompt_manager()
    client = None
    system_prompt = ""
    source_label = ""

    # 2. Select Client
    if "runpod" in requested_source:
        if not app.state.runpod:
            raise HTTPException(status_code=503, detail="RunPod service is not configured or failed to start.")
        
        client = app.state.runpod
        # Use the Fine-Tuned "Physician" Prompt
        system_prompt = prompt_manager.get_prompt("docmail", "physician_system_prompt")
        source_label = "Fine-Tuned Llama 3 (RunPod)"
        
    else:
        # Default to Bedrock
        if not app.state.aws:
             raise HTTPException(status_code=503, detail="AWS Bedrock service is not configured.")
             
        client = app.state.aws
        # Use the Complex "RAG" Prompt
        system_prompt = prompt_manager.get_prompt("docmail", "rag_system_prompt")
        source_label = "Claude 4.5 Sonnet (AWS Bedrock)"

    # 3. Execute
    try:
        logger.info(f"🧠 Routing to: {source_label}")
        
        draft_text = client.generate_draft(
            system_prompt=system_prompt,
            user_prompt=request.patient_email,
            # Tuning: RunPod likes 0.6 temp, Claude prefers 0.3 for medical
            temperature=0.6 if "runpod" in requested_source else 0.3,
            max_tokens=1024
        )
        
        return DraftResponse(
            draft_reply=draft_text,
            source_nodes=[source_label] # We will add real RAG nodes here later
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

