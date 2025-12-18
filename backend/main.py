# Copyright © 2025 SRF Development, Inc. All rights reserved.
#
# This file is part of the "Howie AI Assistant" project.
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
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn
from contextlib import asynccontextmanager
import vertexai
import logging
from fastapi import HTTPException

# custom modules
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.gcs_service import GCSService
from core.vertex_ai_service import VertexAIService
from prompts.manager import initialize_prompt_manager, get_prompt_manager
from backend.models import QueryRequest, QueryResponse, SourceNode, initialize_global_models


# Configure Logger
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on STARTUP ---
    logger.info("INFO:     Starting application...")
    
    # 1. Setup logging
    setup_logging(logger_name="howie", log_level="INFO")

    # 2. Load configuration
    app.state.config = AppConfig()
    
    #3 Initialize global models (LLM, Embeddings)
    embed_model, llm = initialize_global_models(app.state.config)

    # 2. Initialize the Vertex AI SDK. This is the key step.
    logger.info(f"INFO:     Initializing Vertex AI for project '{app.state.config.gcp_project_id}' in region '{app.state.config.gcp_region}'...")
    vertexai.init(project=app.state.config.gcp_project_id, location=app.state.config.gcp_region)
    logger.info("INFO:     Vertex AI initialized successfully.")
    
    # 3. Instantiate your centralized services
    app.state.gcs = GCSService(app.state.config)
    app.state.vertex_service = VertexAIService(app.state.config, embed_model=embed_model, storage_service=app.state.gcs)

    app.state.vertex_service.provision_vertex_resources()
    app.state.vertex_service.connect_and_load()

    index = app.state.vertex_service.get_index()
    if not index:
        logger.error("Failed to retrieve Vertex AI Index. Exiting.")
        exit(1)
    logger.info("Vertex AI Service initialized.")
    
    # Initialize prompt manager
    initialize_prompt_manager(app.state.config.prompts_path)

    prompt_manager = get_prompt_manager()
    if not prompt_manager:
        logger.error("Failed to initialize prompt manager. Exiting.")
        exit(1)
    logger.info("Prompt Manager initialized.")
    
    # 4. Create the query engine and store it in our app_state dictionary
    #    This makes it accessible to our API endpoints.
    logger.info("INFO:     Loading RAG query engine...")
    app.state.query_engine = app.state.vertex_service.get_query_engine(llm)
    
    # Optional: Update the query engine's prompt template
    # qa_template = prompt_manager.get_prompt("rag", "qa_system_prompt")
    # query_engine.update_prompts(...)
    
    logger.info("INFO:     Query engine loaded. Application is ready.")
    
    yield
    
    # --- Code to run on SHUTDOWN ---
    logger.info("INFO:     Shutting down application...")
    # You could add cleanup code here if needed, like closing database connections.


# Initialize the FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",  # Example: frontend running locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)



# --- API Endpoints ---

class QueryRequest(BaseModel): # You'll need to import BaseModel from pydantic
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Receives a user query and returns an answer from the RAG engine.
    """
    if not app.state.query_engine:
        raise HTTPException(status_code=503, detail="Query engine is not available.")
        
    query_engine = app.state.query_engine
    
    # Use the pre-loaded query engine to answer the question
    response = await query_engine.aquery(request.query)
    
    # You can return the full response or just the text
    return {"answer": response.response, "sources": [node.metadata for node in response.source_nodes]}