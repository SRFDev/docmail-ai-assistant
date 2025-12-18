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
import json
import os
import argparse
import hashlib
import logging

# from shapely import node

from config.logger_config import setup_logging
from config.loader import AppConfig
from prompts.manager import initialize_prompt_manager, get_prompt_manager
from core.gcs_service import GCSService
from core.vertex_ai_service import VertexAIService
from core import constants
from core.manifest import load_manifest, save_manifest
from core.hash import calculate_hashes_of_sources


# Google Cloud/AI imports
import google.auth
import vertexai
from google import genai
from google.genai import types

# Type imports
from pydantic import BaseModel, Field
from typing import List
import instructor

# LlamaIndex imports
from llama_index.readers.file import UnstructuredReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, Settings


# Use a Pydantic model to define the structure of the output you want from the AI model.
class VideoAction(BaseModel):
    action: str = Field(description="Description of the action.")
    timestamp: float = Field(
        description="Timestamp in seconds where the action occurs."
    )

class VideoData(BaseModel):
    summary: str = Field(description="A brief summary of the video content.")
    actions: List[VideoAction] = Field(
        description="List of key actions observed in the video."
    )


def init_models(config: AppConfig):
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


def get_video_data(config: AppConfig) -> VideoData:
    """
    Generates a structured summary of the video using Google GenAI.

    Args:   video_blob: The GCS URI or local path to the video file.
    Returns:  A VideoSummary object containing the summary and actions.
    """
    # Check cache for existing video summary
    video_filename = config.video_src_path
    cache_dir = constants.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True) # Create the directory if it doesn't exist

    # Create a unique cache file path
    cache_filepath = os.path.join(cache_dir, f"{os.path.basename(video_filename)}.summary.json")

    video_data: VideoData = None

    # Check if the cache exists
    if os.path.exists(cache_filepath):
        logger.info(f"INFO:     Found cached summary at '{cache_filepath}'. Loading from cache.")
        with open(cache_filepath, 'r') as f:
            video_data_json = json.load(f)
            # video_data_json = json.loads(video_data_json_str)  # Ensure the response is valid JSON
        video_data = VideoData.model_validate(video_data_json)
    else:
        logger.info("INFO:     No cache found. Calling Gemini API to generate summary...")
        credentials, _ = google.auth.default()
        prompt_manager = get_prompt_manager()
        json_generation_prompt = prompt_manager.get_prompt("video_analysis", "structured_summary")

        # Initialize the Vertex AI client with the project and region
        # It automatically uses the credentials from the environment.
        client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.gcp_region,
            credentials=credentials,
        )

        # Patch the client with instructor
        client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

        # THE INGESTION STEP: acting as the courier for the video file to the AI model.
        prompt_parts = [
            types.Part.from_uri(file_uri=config.video_gcs_uri, 
                                mime_type=config.video_mime_type),
            json_generation_prompt,
        ]
    
        # THE MODEL CALL: The model works with the file that is now local to it
        logger.info("Generating video summary...")
        response = client.models.generate_content(
            model=config.llm_model_name, 
            contents=prompt_parts,  # [video_part, structured_summary_prompt],
            config={
                "responseMimeType": "application/json",
                "responseSchema": VideoData,
            },
        )

        if not response.text:
            logger.info("No response text received from the AI model.")
            return
        if not isinstance(response.text, str):
            logger.info("Response text is not a string:", response.text)
            return

        # Validate and parse the response text into a VideoSummary model
        try:
            # This will raise an error if the response does not match the VideoSummary model
            # or if the JSON is malformed.
            responseText = response.text.strip()
            if not responseText.startswith("{") or not responseText.endswith("}"):
                logger.info("Response text is not a valid JSON object:", responseText)
                return
        except ValueError as e:
            logger.info("Error parsing response text as JSON:", e)
            return  
        responseJson = json.loads(responseText)  # Ensure the response is valid JSON
        video_data = VideoData.model_validate(responseJson)

        # Save the result to the cache
        with open(cache_filepath, 'w') as f:
            json.dump(responseJson, f, indent=2)
        logger.info(f"INFO:     Saved new summary to cache at '{cache_filepath}'.")

    return video_data


def parse_data(config: AppConfig):

    # --- Video Processing ---
    video_data: VideoData = get_video_data(config)
    video_name = config.video_src_path
    video_name_hash = hashlib.md5(video_name.encode()).hexdigest()[:6]  # Short hash for uniqueness
    video_id = f"video:{video_name_hash}:0"

    video_nodes = []
    video_nodes.append(Document(
        id_=f"{video_id}_summary",
        text=video_data.summary, 
        metadata={"source": "video_summary"}
        ))

    # Add actions as individual documents
    for i, action in enumerate(video_data.actions):
        node_id=f"video:{video_name_hash}_action_{i}"
        video_nodes.append(Document(
            id_=node_id,
            text=action.action,
            metadata={
                    "source": "video_action",
                    "timestamp": action.timestamp,
                },
            )
        )

    # --- PDF Processing ---
    pdf_name = config.pdf_src_path
    pdf_name_hash = hashlib.md5(pdf_name.encode()).hexdigest()[:6]  # Short hash for uniqueness
    pdf_id = f"pdf:{pdf_name_hash}:0"

    # Use UnstructuredReader to read the PDF in visual order
    loader = UnstructuredReader()
    unstructured_pdf_docs = loader.load_data(file=config.pdf_src_path)

    # Get the text chunks from the PDF documents
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    text_chunks = parser.split_text(unstructured_pdf_docs[0].get_content())
    # text_chunks = parser.get_nodes_from_documents(unstructured_pdf_docs)

    # Add metadata to each node
    pdf_nodes = []
    for i, chunk in enumerate(text_chunks):
        node_id = f"{pdf_id}_chunk_{i}"
        node = Document(
            id_=node_id,
            text=chunk,
            metadata={
                "source_id": pdf_id,
                "file_name": config.pdf_src_path,
            }
        )
        pdf_nodes.append(node)


    return video_nodes + pdf_nodes


def parse_and_ingest_if_necessary(config: AppConfig, vertex: VertexAIService):
    """
    Checks if the data has already been parsed and ingested.
    If not, it parses the data and ingests it into the Vertex AI index.
    """

    # Calculate hashes of the sources
    hashes = calculate_hashes_of_sources(config)
    
    # Load the manifest to check if we need to re-ingest
    manifest = load_manifest(config)
    
    if manifest is None:
        logger.info("Manifest not found. Parsing and ingesting data...")
        nodes = parse_data(config)
        vertex.ingest_nodes(nodes)
        save_manifest(config, hashes)
        logger.info("Data parsed and ingested successfully.")
    else:
        logger.info("Manifest found. Checking for changes...")
        if hashes != manifest:
            logger.info("Changes detected in source files. Re-parsing and ingesting data...")
            nodes = parse_data(config)
            vertex.ingest_nodes(nodes)
            save_manifest(config, hashes)
            logger.info("Data re-parsed and re-ingested successfully.")
        else:
            logger.info("No changes detected. Skipping parsing and ingestion.")


if __name__ == "__main__":

    # Set up logging for the script
    setup_logging(logger_name="howie", log_level="INFO") 
    logger = logging.getLogger(__name__)

    logger.info("Starting the ingestion script...")

    #  Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest data into the Howie knowledge base.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="If set, this flag will delete the existing Vertex AI index and endpoint."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="If set, this flag will upload Howie's data to GCS."
    )
    args = parser.parse_args()

    # Use AppConfig class to load config.toml.  
    config = AppConfig()

    embed_model, llm = init_models(config)

    # Initialize the Vertex AI client with the project and region
    vertexai.init(project=config.gcp_project_id, location=config.gcp_region)

    gcs = GCSService(config)
    vertex = VertexAIService(config=config, embed_model=embed_model, storage_service=gcs)

    if args.reset:
        vertex.reset_resources()

        logger.info("Resetting all local and remote resources...")
        try:
            docstore_path = constants.CACHE_DOCSTORE_PATH
            if os.path.exists(docstore_path):
                os.remove(docstore_path)
                logger.info(f"Successfully deleted local docstore at '{docstore_path}'.")
            
            # Also delete the video summary cache
            video_summary_path = constants.CACHE_VIDEO_SUMMARY_PATH
            if os.path.exists(video_summary_path):
                os.remove(video_summary_path)
                logger.info(f"Successfully deleted video summary cache at '{video_summary_path}'.")

        except OSError as e:
            logger.error(f"Error removing cache files: {e}", exc_info=True)
            
        # Remove the local manifest file if it exists
        try:
            manifest_path = constants.CACHE_INGESTION_MANIFEST_PATH # Add this to your constants
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
                logger.info(f"Successfully deleted local manifest at '{manifest_path}'.")
        except OSError as e:
            logger.error(f"Error removing manifest file: {e}", exc_info=True)


        logger.info("--- All local and remote resources reset. ---")

    elif args.upload:
        gcs.ensure_bucket_exists()
        gcs.upload_file(config.video_src_path, config.video_dest_path)
        gcs.upload_file(config.pdf_src_path, config.pdf_dest_path)
        logger.info("Uploaded video and PDF manual to GCS.")  
    else:
        gcs.ensure_bucket_exists()

        # gcs.list_files(prefix="data/")
        vertex.provision_vertex_resources()
        vertex.connect_and_load()

        # Initialize the prompt manager
        initialize_prompt_manager(config.prompts_path)

        # Parse the data from the video and PDF sources if necessary
        parse_and_ingest_if_necessary(config, vertex)

        logger.info("Ingestion completed successfully.")

