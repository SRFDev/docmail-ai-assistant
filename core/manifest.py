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
from core import constants
import json
import os
import logging


logger = logging.getLogger(__name__)

def load_manifest(config: AppConfig):
    """
    Loads the manifest file from the specified path.
    Returns the parsed JSON content.
    """
    manifest_path = constants.CACHE_INGESTION_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        logger.info(f"Manifest file not found at {manifest_path}.")
        return None
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    return manifest_data

def save_manifest(config: AppConfig, data: dict):
    """
    Saves the provided data to the manifest file at the specified path.
    """
    manifest_path = constants.CACHE_INGESTION_MANIFEST_PATH
    with open(manifest_path, 'w') as f:
        json.dump(data, f, indent=2)
