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
"""
This module contains constant values used throughout the "Howie" application.
It serves as a single source of truth for conventional paths, filenames, and default values.
"""

# --- Directory Paths (relative to the project root) ---
CONFIG_DIR = "config"
PROMPTS_DIR = "prompts"
DATA_DIR = "data"
CACHE_DIR = ".cache"


# --- Filenames ---
CONFIG_FILE_NAME = "config.toml"
PROMPTS_FILE_NAME = "prompts.toml"
DOCSTORE_FILE_NAME = "docstore.json"
VIDEO_SUMMARY_CACHE_FILE_NAME = "steves-pour-over-method.mp4.summary.json"
INGESTION_MANIFEST_FILE_NAME = "ingestion_manifest.json"

# --- Full Paths (constructed for convenience) ---
# Note: These assume the application is run from the project root.
import os

# Construct full paths to config and prompt files
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)
PROMPTS_FILE_PATH = os.path.join(PROMPTS_DIR, PROMPTS_FILE_NAME)

# Construct full paths for cache files
CACHE_DOCSTORE_PATH = os.path.join(CACHE_DIR, DOCSTORE_FILE_NAME)
CACHE_VIDEO_SUMMARY_PATH = os.path.join(CACHE_DIR, VIDEO_SUMMARY_CACHE_FILE_NAME)
CACHE_INGESTION_MANIFEST_PATH = os.path.join(CACHE_DIR, INGESTION_MANIFEST_FILE_NAME)
