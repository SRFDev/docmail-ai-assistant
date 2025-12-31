# core/aws_service.py
import boto3
import json
import logging
from typing import Optional
from botocore.exceptions import ClientError

# LlamaIndex Imports (for Embeddings)
from llama_index.embeddings.bedrock import BedrockEmbedding

# Interface
from core.llm_interface import LLMProvider

logger = logging.getLogger(__name__)

class AwsService(LLMProvider):
    _instance = None

    def __init__(self, config):
        """
        Private Constructor. Use get_instance() instead.
        """
        self.config = config
        self.region = config.aws_region
        
        try:
            # 1. Initialize Boto3 Clients
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime", 
                region_name=self.region
            )
            
            # 2. Initialize Embeddings (Titan v2) - Needed for ChromaDB
            # We wrap this in LlamaIndex's class for easy integration later
            self.embed_model = BedrockEmbedding(
                model_name=config.embed_model_id,
                client=self.bedrock_runtime
            )
            
            logger.info("✅ AWS Bedrock Clients Initialized")
            
        except Exception as e:
            logger.critical(f"❌ Failed to connect to AWS Bedrock: {e}")
            raise e

    @classmethod
    def get_instance(cls, config):
        """Singleton Accessor"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def generate_draft(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Implementation of the LLMProvider interface for AWS Bedrock (Claude 3/3.5).
        """
        logger.info(f"Generating draft via AWS Bedrock ({self.config.llm_model_id})...")
        
        # Claude 3 Messages API Payload
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.3),
            "system": system_prompt,
            "messages": [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]
        })
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config.llm_model_id,
                body=body
            )
            
            # Parse Response
            response_body = json.loads(response.get("body").read())
            # Extract text from Claude's response structure
            return response_body["content"][0]["text"]
            
        except ClientError as e:
            logger.error(f"AWS Bedrock API Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected Error in AWS Generation: {e}")
            raise e