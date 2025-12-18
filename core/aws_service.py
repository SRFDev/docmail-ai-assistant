import logging
import boto3
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings

# Assuming you have a similar config setup as Howie
# If not, we can simplify this import later.
from config.loader import AppConfig 

logger = logging.getLogger(__name__)

class AwsService:
    """
    Centralized service for AWS operations (S3, Bedrock).
    Replaces the old VertexAIService.
    """
    _instance = None

    def __init__(self, config: AppConfig):
        if AwsService._instance is not None:
            raise Exception("AwsService is a singleton. Use get_instance().")
        
        self.config = config
        self._s3_client = None
        self._llm = None
        self._embed_model = None
        
        # AWS Configuration
        # We use the profile we created: 'docmail'
        self.session = boto3.Session(profile_name='docmail')
        self.region = "us-east-1"
        
        # The ID we just validated
        self.llm_model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.embed_model_id = "amazon.titan-embed-text-v2:0" # Standard, cheap, effective

        AwsService._instance = self

    @staticmethod
    def get_instance(config: AppConfig = None):
        if AwsService._instance is None:
            if config is None:
                raise ValueError("Config required for first initialization")
            AwsService(config)
        return AwsService._instance

    @property
    def s3_client(self):
        """Lazy load S3 client"""
        if self._s3_client is None:
            self._s3_client = self.session.client('s3', region_name=self.region)
        return self._s3_client

    @property
    def llm(self):
        """Lazy load Bedrock LLM (Claude Sonnet)"""
        if self._llm is None:
            logger.info(f"Initializing Bedrock LLM: {self.llm_model_id}")
            self._llm = Bedrock(
                model=self.llm_model_id,
                profile_name='docmail',
                region_name=self.region,
                context_size=200000, 
                temperature=0.7,
                max_tokens=4096  # <--- FIX: ADD THIS LINE (Was likely defaulting to 512)
            )
            # Bind to global Settings for LlamaIndex convenience
            Settings.llm = self._llm
        return self._llm

    @property
    def embed_model(self):
        """Lazy load Bedrock Embeddings (Titan v2)"""
        if self._embed_model is None:
            logger.info(f"Initializing Bedrock Embeddings: {self.embed_model_id}")
            self._embed_model = BedrockEmbedding(
                model_name=self.embed_model_id,
                profile_name='docmail',
                region_name=self.region
            )
            # Bind to global Settings
            Settings.embed_model = self._embed_model
        return self._embed_model

    def upload_file(self, file_path: str, object_name: str = None):
        """Upload a file to the S3 bucket defined in config"""
        if object_name is None:
            object_name = file_path.split("/")[-1]
            
        bucket = self.config.s3_bucket_name # Ensure config.toml has this key!
        
        try:
            self.s3_client.upload_file(file_path, bucket, object_name)
            logger.info(f"Uploaded {file_path} to s3://{bucket}/{object_name}")
            return f"s3://{bucket}/{object_name}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise