# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import sys

# Local imports
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.aws_service import AwsService
from prompts.manager import initialize_prompt_manager, get_prompt_manager

# LlamaIndex
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def init_query_engine(config: AppConfig):
    """Initializes the RAG engine for CLI usage."""
    
    # 1. AWS & Models
    aws = AwsService.get_instance(config)
    _ = aws.llm
    _ = aws.embed_model
    
    # 2. Vector Store
    print(f"Connecting to ChromaDB at {config.chroma_persist_dir}...")
    db = chromadb.PersistentClient(path=config.chroma_persist_dir)
    chroma_collection = db.get_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # 3. Prompts
    initialize_prompt_manager(config.prompts_path)
    pm = get_prompt_manager()
    qa_template_str = pm.get_prompt("docmail", "rag_system_prompt")
    qa_template = PromptTemplate(qa_template_str)
    
    # 4. Engine
    query_engine = index.as_query_engine(
        similarity_top_k=config.top_k_retrieval,
        text_qa_template=qa_template
    )
    
    return query_engine

def main():
    setup_logging(logger_name="docmail_cli", log_level="WARNING")
    config = AppConfig()
    
    print("--- DocMail Interactive CLI ---")
    print("Initializing...")
    
    try:
        query_engine = init_query_engine(config)
        print("✅ Engine Ready.")
        print("Paste a patient email below (or type 'quit' to exit).")
        print("-" * 50)
        
        while True:
            user_input = input("\nPatient Email > ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            print("\nDrafting Reply...")
            response = query_engine.query(user_input)
            
            print("\n" + "="*20 + " DRAFT REPLY " + "="*20)
            print(str(response))
            print("="*53)
            
            print("\n[Style References Used:]")
            for node in response.source_nodes:
                meta = node.metadata
                print(f"- {meta.get('specialty')} / {meta.get('scenario')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    