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
from config.loader import AppConfig
from config.logger_config import setup_logging
from prompts.manager import get_prompt_manager, initialize_prompt_manager
from core.gcs_service import GCSService
from core.vertex_ai_service import VertexAIService
from scripts.ingest import init_models
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
import logging
import google.auth
import vertexai


def init(config: AppConfig):
    """Initializes the service clients and returns them."""
    
    embed_model, llm = init_models(config)

    # Initialize the Vertex AI client with the project and region
    vertexai.init(project=config.gcp_project_id, location=config.gcp_region)


    logger.info("Initializing service clients...")
    gcs = GCSService(config)
    logger.info("GCS Service initialized.")
    
    # Initialize Vertex AI service
    logger.info("Initializing Vertex AI Service...")
    vertex = VertexAIService(config=config, embed_model=embed_model, storage_service=gcs)
   
    vertex.provision_vertex_resources()
    vertex.connect_and_load()

    index = vertex.get_index()
    if not index:
        logger.error("Failed to retrieve Vertex AI Index. Exiting.")
        exit(1)
    logger.info("Vertex AI Service initialized.")
    
    # Initialize prompt manager
    initialize_prompt_manager(config.prompts_path)

    prompt_manager = get_prompt_manager()
    if not prompt_manager:
        logger.error("Failed to initialize prompt manager. Exiting.")
        exit(1)
    logger.info("Prompt Manager initialized.")
    
    # Initialize the query engine
    logger.info("Initializing query engine...")
    query_engine = index.as_query_engine(similarity_top_k=3)

    return gcs, vertex, index, query_engine, prompt_manager


if __name__ == "__main__":
    # Setup logging
    setup_logging(logger_name="howie", log_level="WARNING")
    logger = logging.getLogger(__name__)
    logger.info("Starting Howie Assistant Query Script...")

    # Load configuration
    config = AppConfig()

    gcs, vertex, index, query_engine, prompt_manager = init(config)

    qa_template_str = prompt_manager.get_prompt("rag", "qa_system_prompt")

    # You can integrate this with LlamaIndex's prompt templating system
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=qa_template_str,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="{question_str}" # LlamaIndex will fill this last
            ),
        ]
    )
    query_engine.update_prompts({"chat_template": chat_template})
    logger.info("Query engine initialized with custom chat template.")
    logger.info("Ready to accept queries.")

    # Start query loop
    while True:
        user_input = input("Enter a query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            logger.info("Exiting program.")
            break

        query_str = user_input.strip()

        print("Running the query against the index...")
        response = query_engine.query(query_str)
        print("\nResponse:")
        print(response.response)
        print("-" * 80)

