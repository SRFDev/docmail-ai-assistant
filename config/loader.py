# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import tomllib # Use 'tomli' for Python < 3.11
import os
from core import constants

class AppConfig:
    """
    Application configuration class for DocMail.
    Loads configuration from config.toml and provides access to settings.
    """

    def __init__(self, config_path=constants.CONFIG_FILE_PATH):
        try:
            with open(config_path, "rb") as f:
                self._config = tomllib.load(f)

            # AWS Configuration
            self.aws_region = self._config["aws"]["region"]
            self.s3_bucket_name = self._config["aws"]["s3_bucket_name"]
            
            # Bedrock Models
            self.llm_model_id = self._config["aws"]["llm_model_id"]
            self.embed_model_id = self._config["aws"]["embed_model_id"]

            # Data Sources
            self.physician_style_path = self._config["data"]["physician_style_path"]
            
            # Vector Store (ChromaDB)
            self.chroma_persist_dir = self._config["vector_store"]["persist_dir"]
            self.collection_name = self._config["vector_store"]["collection_name"]
            self.top_k_retrieval = self._config["vector_store"]["top_k_retrieval"]

            # Prompts
            self.prompts_path = self._config["prompts"]["prompts_path"]

        except FileNotFoundError:
            print(f"Error: {config_path} not found.")
            exit()
        except KeyError as e:
            print(f"Error: Missing key in config.toml: {e}")
            exit()

# Example usage
if __name__ == "__main__":
    config = AppConfig()
    print(f"AWS Region: {config.aws_region}")
    print(f"S3 Bucket: {config.s3_bucket_name}")
    print(f"Model: {config.llm_model_id}")
    print(f"Vector Store: {config.chroma_persist_dir}")

    