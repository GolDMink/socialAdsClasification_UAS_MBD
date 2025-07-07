#!/bin/bash

# Create .streamlit directory in user's home if it doesn't exist
mkdir -p ~/.streamlit

# Run streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 