# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import streamlit as st
import requests
import tomllib  # Use 'tomli' for Python < 3.11 if needed
import os

# --- 0. Page Config (Professional Polish) ---
st.set_page_config(
    page_title="DocMail Assistant",
    page_icon="🩺",
    layout="centered"
)

# --- 1. Centralized Configuration Function ---
def get_api_url() -> str:
    """
    Determines the correct backend API endpoint URL based on the environment.
    Reads from Streamlit Secrets in production or a local config.toml for development.
    """
    # Check if running on Streamlit Community Cloud (where secrets are set)
    if "BACKEND_URL" in st.secrets:
        backend_url = st.secrets["BACKEND_URL"]
        # print(f"Using backend URL from Streamlit secrets: {backend_url}")
    else:
        # Fallback for local development
        # print("Loading backend URL from local config.toml")
        try:
            # Assumes your streamlit app is run from the 'docmail' project root
            config_path = "config/config.toml" 
            # Note: Depending on where you run 'streamlit run', this path might vary.
            # If running from root, "config/config.toml" is correct.
            # If running from frontend/, "../config/config.toml" is correct.
            # We'll try root first, then fallback.
            
            if not os.path.exists(config_path):
                 config_path = "config.toml" # Try root if config is moved

            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            backend_url = config.get("api", {}).get("backend_url", "http://127.0.0.1:8000")
        except FileNotFoundError:
            backend_url = "http://127.0.0.1:8000"
    
    # Append the DocMail specific endpoint
    return f"{backend_url}/generate"

# --- 2. Main Application ---
st.title("DocMail Assistant 🩺")
st.markdown("""
**Physician's Email Draft Tool**  
Paste a patient email below. DocMail will retrieve similar past replies and draft a response using your established style and safety guidelines.
""")

# Get the endpoint URL ONCE at the start
API_URL = get_api_url()
# st.sidebar.info(f"Connected to: `{API_URL}`") # Debug info

# --- 3. Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Display past messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources for assistant messages if available
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander(f"📚 References ({len(sources)})"):
                    for source in sources:
                        st.write(f"- {source}")

# --- 5. User Input ---
if prompt := st.chat_input("Paste patient email here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 6. Backend Interaction ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Analyzing style and drafting reply..."):
            try:
                # DocMail API call
                payload = {"patient_email": prompt}
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract fields from DraftResponse schema
                    draft = data.get("draft_reply", "Error: No draft returned.")
                    sources = data.get("source_nodes", [])
                    
                    # Display the Draft
                    message_placeholder.markdown(draft)
                    
                    # Display Sources (The backend formats them as strings now)
                    if sources:
                        with st.expander(f"📚 References ({len(sources)})"):
                            for source in sources:
                                st.write(f"- {source}")
                    
                    # Save to History
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": draft, 
                        "sources": sources
                    })

                else:
                    error_msg = f"**Error {response.status_code}:** {response.text}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except requests.exceptions.RequestException as e:
                error_msg = f"**Connection Error:** Could not reach backend at {API_URL}.\n\nDetails: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
# --- End of app.py ---
