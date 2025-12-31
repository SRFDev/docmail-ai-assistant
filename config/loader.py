# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import tomllib # Use 'tomli' for Python < 3.11
import os
import logging
from dotenv import load_dotenv
from core import constants

# Load Environment Variables from .env file
load_dotenv()

# Setup local logger just for config loading issues
logger = logging.getLogger(__name__)


class AppConfig:
    """
    Application configuration class for DocMail.
    Loads configuration from config.toml and overrides with Environment Variables.
    """

    def __init__(self, config_path=constants.CONFIG_FILE_PATH):
        try:
            with open(config_path, "rb") as f:
                self._config = tomllib.load(f)

            # --- 1. LLM Strategy Selection ---
            # Priority: ENV VAR > TOML > Default ('bedrock')
            toml_source = self._config.get("app", {}).get("llm_source", "bedrock")
            self.llm_source = os.getenv("LLM_SOURCE", toml_source).lower()

            # --- 2. AWS Configuration ---
            self.aws_region = self._config["aws"]["region"]
            self.s3_bucket_name = self._config["aws"]["s3_bucket_name"]
            
            # Bedrock Models
            self.llm_model_id = self._config["aws"]["llm_model_id"]
            self.embed_model_id = self._config["aws"]["embed_model_id"]

            # --- 3. Data Sources ---
            # Using .get() is safer to avoid crashes if 'data' section is optional
            self.physician_style_path = self._config.get("data", {}).get("physician_style_path")
            
            # --- 4. Vector Store (ChromaDB) ---
            self.chroma_persist_dir = self._config["vector_store"]["persist_dir"]
            self.collection_name = self._config["vector_store"]["collection_name"]
            self.top_k_retrieval = self._config["vector_store"]["top_k_retrieval"]

            # --- 5. Prompts ---
            self.prompts_path = self._config["prompts"]["prompts_path"]
            
            # --- 6. RunPod Specifics ---
            # Endpoint ID comes from TOML (Infrastructure)
            self.runpod_endpoint_id = self._config.get("runpod", {}).get("endpoint_id")            

            # API Key MUST come from Environment (.env), NOT TOML
            self.runpod_api_key = os.getenv("RUNPOD_API_KEY")
            
            # --- 7. Validation ---
            if self.llm_source == "runpod":
                if not self.runpod_endpoint_id:
                     raise ValueError("❌ Config Error: LLM_SOURCE is 'runpod' but 'endpoint_id' is missing in config.toml")
                if not self.runpod_api_key:
                    raise ValueError("❌ Config Error: LLM_SOURCE is 'runpod' but RUNPOD_API_KEY not found in env.")            

        except FileNotFoundError:
            logger.critical(f"CRITICAL: Config file not found at {config_path}")
            exit(1)
        except KeyError as e:
            logger.critical(f"CRITICAL: Missing required key in config.toml: {e}")
            exit(1)

# Example usage
if __name__ == "__main__":
    try:
        config = AppConfig()
        print(f"✅ Config Loaded Successfully")
        print(f"   Mode: {config.llm_source.upper()}")
        print(f"   AWS Region: {config.aws_region}")
        print(f"   RunPod Endpoint: {config.runpod_endpoint_id or 'N/A'}")
    except Exception as e:
        print(e)

