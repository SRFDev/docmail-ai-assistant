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
import tomllib # Use 'tomli' for Python < 3.11
from core import constants


class AppConfig:
    """
    Application configuration class.
    Loads configuration from config.toml and provides access to settings.
    """

    def __init__(self, config_path=constants.CONFIG_FILE_PATH):
        try:
            with open(config_path, "rb") as f:
                self._config = tomllib.load(f)

            self.gcp_project_id = self._config["gcp"]["gcp_project_id"]
            self.gcp_region = self._config["gcp"]["gcp_region"]
            self.gcs_bucket_name = self._config["gcp"]["gcs_bucket_name"]

            self.data_sources = self._config["data_sources"]

            self.pdf_src_path = self._config["data_sources"]["pdf_src_path"]
            self.pdf_dest_path = self._config["data_sources"]["pdf_dest_path"]
            self.video_gcs_uri = self._config["data_sources"]["video_gcs_uri"]
            self.video_src_path = self._config["data_sources"]["video_src_path"]
            self.video_dest_path = self._config["data_sources"]["video_dest_path"]
            self.video_mime_type = self._config["data_sources"].get("video_mime_type", "video/mp4")
            
            self.llm_model_name = self._config["llm"]["model_name"]
            self.llm_embedding_model_name = self._config["llm"]["embedding_model_name"]

            self.rag_tuning = self._config["rag_tuning"]

            self.chunk_size = self._config["rag_tuning"]["chunk_size"]
            self.chunk_overlap = self._config["rag_tuning"]["chunk_overlap"]
            self.top_k_retrieval = self._config["rag_tuning"]["top_k_retrieval"]

            self.vector_search = self._config["vector_search"]
            self.vs_index_name = self._config["vector_search"]["vs_index_name"]
            self.vs_index_endpoint_name = self._config["vector_search"]["vs_index_endpoint_name"]
            self.vs_index_deployment_name = self._config["vector_search"]["vs_index_deployment_name"]
            self.vs_dimensions = self._config["vector_search"]["vs_dimensions"]
            self.insert_batch_size = self._config["vector_search"]["insert_batch_size"]

            self.prompts_path = self._config["prompts"]["prompts_path"]


        except FileNotFoundError:
            print("Error: config.toml not found.")
            exit()
        except tomllib.TOMLDecodeError as e:
            print(f"Error decoding config.toml: {e}")
            exit()

    def get(self, key, default=None):
        return self.config.get(key, default)


# Example usage
if __name__ == "__main__":
    config = AppConfig()

    print("Project ID:", config.gcp_project_id)
    print("Region:", config.gcp_region)
    print("GCS Bucket Name:", config.gcs_bucket_name)
    print("RAG Tuning Parameters:", config.rag_tuning)

    print("Chunk Size:", config.rag_tuning.get("chunk_size", 256))
    print("Chunk Overlap:", config.rag_tuning.get("chunk_overlap", 20))
    print("Top K Retrieval:", config.rag_tuning.get("top_k_retrieval", 3))

    print("Data Sources:", config.data_sources)
    print("RAG Tuning:", config.rag_tuning)

    print("Prompts path:", config.prompts_path)


# # Ensure the config file exists
# if not os.path.exists("config.toml"):
#     print("Error: config.toml file is missing. Please create it based on the template.")
#     exit()

# # Ensure the config file is valid
# try:
#     with open("config.toml", "rb") as f:
#         tomllib.load(f)
# except tomllib.TOMLDecodeError as e:
#     print(f"Error decoding config.toml: {e}")
#     exit()  


