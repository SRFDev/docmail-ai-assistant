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
from google.cloud import storage
from config.loader import AppConfig
import logging


logger = logging.getLogger(__name__)


class GCSService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = storage.Client(project=self.config.gcp_project_id)
        # self.bucket = self.ensure_bucket_exists()


    def upload_file(self, local_file_path, gcs_destination_path):

        blob = self.bucket.blob(gcs_destination_path)

        blob.upload_from_filename(local_file_path)
        logger.info(f"File {local_file_path} uploaded to {gcs_destination_path}.")

    
    def download_file(self, gcs_source_path, local_destination_path):

        blob = self.bucket.blob(gcs_source_path)

        blob.download_to_filename(local_destination_path)
        logger.info(f"File {gcs_source_path} downloaded to {local_destination_path}.")


    def list_files(self, prefix=None):

        blobs = self.client.list_blobs(self.config.gcs_bucket_name)
        file_list = [blob.name for blob in blobs]

        logger.info(f"Files in bucket {self.config.gcs_bucket_name} with prefix '{prefix}':")
        for file in file_list:
            print(file)

        return file_list
    
    def upload_string(self, content: str, destination_blob_name: str):
        """
        Uploads a string to a blob in the GCS bucket.
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(content)
        logger.info(f"String uploaded to {destination_blob_name} in bucket {self.config.gcs_bucket_name}.")
        

    def ensure_bucket_exists(self):
        
        bucket = self.client.bucket(self.config.gcs_bucket_name)

        if not bucket.exists():
            bucket.create(location=self.config.gcp_region)
            logger.info(f"Bucket {self.config.gcs_bucket_name} created in region {self.config.gcp_region}.")
        else:
            logger.info(f"Bucket {self.config.gcs_bucket_name} already exists.")
        
        self.bucket = bucket
        
        return bucket
    

    def delete_file(self, gcs_file_path):
        """
        Deletes a file from the GCS bucket.
        """
        blob = self.bucket.blob(gcs_file_path)

        if blob.exists():
            blob.delete()
            logger.info(f"File {gcs_file_path} deleted from bucket {self.config.gcs_bucket_name}.")
        else:
            logger.info(f"File {gcs_file_path} does not exist in bucket {self.config.gcs_bucket_name}.")
    

    def get_gcs_uri(self, file_path):
        """
        Returns the GCS URI for a given file path.
        """
        return f"gs://{self.config.gcs_bucket_name}/{file_path}"
    

# Test the GCSService class
if __name__ == "__main__":

    # Example usage
    config = AppConfig()
    gcs_service = GCSService(config)

    # Ensure bucket exists
    gcs_service.ensure_bucket_exists()

    # List files currently in the bucket
    gcs_service.list_files(prefix="uploads/")

    # Upload a README as a test 
    gcs_service.upload_file("README.md", "uploads/README.md")

    # Get GCS URI
    print(gcs_service.get_gcs_uri("uploads/README.md"))

    # List files in the bucket
    gcs_service.list_files(prefix="uploads/")

    # Download a file
    gcs_service.download_file("uploads/README.md", "downloaded_file.md")
    
    # Delete a file
    gcs_service.delete_file("uploads/README.md")

    # List files currently in the bucket
    gcs_service.list_files(prefix="uploads/")

