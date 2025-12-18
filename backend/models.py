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
from pydantic import BaseModel
from typing import List, Dict, Any
from core.vertex_ai_service import VertexTextEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from config.loader import AppConfig
import google.auth
from llama_index.core import Settings


class QueryRequest(BaseModel):
    """The input model for a user's query."""
    query: str
    # You could add more here later, like user_id, session_id, etc.

class SourceNode(BaseModel):
    """The output model for a single source node."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """The output model for the AI's answer."""
    answer: str
    sources: List[SourceNode]
    

def initialize_global_models(config: AppConfig):
    """
    Initializes the LLM and embedding model for LlamaIndex.
    """
    credentials, _ = google.auth.default()

    embed_model = VertexTextEmbedding(
        model_name=config.llm_embedding_model_name,
        project=config.gcp_project_id,
        location=config.gcp_region,
        credentials=credentials,
        embed_batch_size=1
    )
    
    # gemini_embedding_model = VertexTextEmbedding("text-embedding-005")
    llm = GoogleGenAI(
        model=config.llm_model_name,
        vertexai_config={"project": config.gcp_project_id, "location": config.gcp_region},
        credentials=credentials,
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    return embed_model, llm
