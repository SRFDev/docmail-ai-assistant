# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
This module contains constant values used throughout the "DocMail" application.
It serves as a single source of truth for conventional paths, filenames, and default values.
"""

import os

# --- Directory Paths (relative to the project root) ---
CONFIG_DIR = "config"
PROMPTS_DIR = "prompts"
DATA_DIR = "data"
CACHE_DIR = ".cache"
VECTOR_STORE_DIR = "chroma_db"

# --- Filenames ---
CONFIG_FILE_NAME = "config.toml"
PROMPTS_FILE_NAME = "prompts.toml"
DATASET_FILE_NAME = "physician_style_guide.jsonl"

# --- Full Paths (constructed for convenience) ---
# Note: These assume the application is run from the project root.

# Config & Prompts
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)
PROMPTS_FILE_PATH = os.path.join(PROMPTS_DIR, PROMPTS_FILE_NAME)

# Data
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILE_NAME)
VECTOR_STORE_PATH = os.path.join(os.getcwd(), VECTOR_STORE_DIR)

