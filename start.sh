#!/bin/bash
# Simple bash script to start the Forensic RAG application

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
streamlit run src/app.py

