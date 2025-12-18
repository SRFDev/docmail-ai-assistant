import logging
from config.loader import AppConfig
from config.logger_config import setup_logging
from core.aws_service import AwsService
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb

def test_retrieval():
    # 1. Setup
    setup_logging(logger_name="test_rag", log_level="INFO")
    logger = logging.getLogger(__name__)
    config = AppConfig()
    
    # Init AWS (Set global LLM/Embed settings)
    aws = AwsService.get_instance(config)

    # Force initialization of models to update global Settings
    logging.info("Waking up Bedrock models...")
    _ = aws.llm         # Triggers lazy load & Settings.llm assignment
    _ = aws.embed_model # Triggers lazy load & Settings.embed_model assignment

    logger.info("Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=config.chroma_persist_dir)
    chroma_collection = db.get_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    
    # 3. Create Query Engine
    # We ask for 3 similar examples
    retriever = index.as_retriever(similarity_top_k=3)
    
    # 4. Test Query
    # We simulate a new patient email scenario
    query = "Patient is worried about a new rash after taking medication."
    logger.info(f"Querying: '{query}'")
    
    nodes = retriever.retrieve(query)
    
    logger.info(f"Found {len(nodes)} relevant examples:")
    for i, node in enumerate(nodes):
        # We print the metadata to verify we got the right 'tone' matches
        meta = node.metadata
        logger.info(f"--- Match {i+1} (Score: {node.score:.4f}) ---")
        logger.info(f"Specialty: {meta.get('specialty')}")
        logger.info(f"Scenario: {meta.get('scenario')}")
        logger.info(f"Excerpt: {node.get_content()[:100]}...")

if __name__ == "__main__":
    test_retrieval()

