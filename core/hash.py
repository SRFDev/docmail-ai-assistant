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
import hashlib


def calculate_file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    # Open the file in binary read mode ('rb')
    with open(file_path, "rb") as f:
        # Read the file in chunks (e.g., 4096 bytes)
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_hashes_of_sources(config: AppConfig):
    """
    Calculates the SHA-256 hashes of the video and PDF sources.
    Returns a dictionary with the file names as keys and their hashes as values.
    """
    hashes = {}
    
    # Calculate hash for video source
    video_hash = calculate_file_sha256(config.video_src_path)
    hashes[config.video_src_path] = video_hash
    
    # Calculate hash for PDF source
    pdf_hash = calculate_file_sha256(config.pdf_src_path)
    hashes[config.pdf_src_path] = pdf_hash
    
    return hashes