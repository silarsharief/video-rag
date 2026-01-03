"""
Forensic RAG - Main Streamlit Application
Edge-First Forensic Video Analysis System
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from src.core.ingest import VideoIngestor
from src.core.retrieval import ForensicSearch

# Import configurations
from config.settings import (
    VIDEO_STORAGE_DIR,
    CHROMADB_PATH,
    APP_TITLE,
    APP_ICON,
    APP_LAYOUT,
    DISPLAY_COLUMNS
)
from config.modes import list_available_modes

# Configure Streamlit page
st.set_page_config(layout=APP_LAYOUT, page_title=APP_TITLE)

st.title(f"{APP_ICON} Edge-First Forensic Video Analysis")

# Ensure video directory exists
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SIDEBAR: Configuration & Upload
# ============================================================================
with st.sidebar:
    st.header("1. Configuration")
    
    # Get available modes and add "All" option
    available_modes = ["All"] + list_available_modes()
    selected_mode = st.selectbox("Select Use Case Mode", available_modes)

    st.header("2. Upload Evidence")
    uploaded_files = st.file_uploader(
        "Upload CCTV Footage", 
        type=['mp4', 'mov'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} files selected.")
        
        if st.button("Process All Videos"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Save uploaded file
                file_path = VIDEO_STORAGE_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: '{uploaded_file.name}'...")
                
                # Pass 'general' if 'All' is selected during ingestion
                ingest_mode = "general" if selected_mode == "All" else selected_mode
                
                try:
                    ingestor = VideoIngestor(str(file_path), mode=ingest_mode)
                    ingestor.process_video()
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success("All videos processed successfully!")

# ============================================================================
# MAIN: Forensic Search Interface
# ============================================================================
st.header("2. Forensic Search")
query = st.text_input("Ask a question about the footage:", "Show me safety violations.")

# Only enable search if database exists
if query and CHROMADB_PATH.exists():
    search_engine = ForensicSearch()
    
    with st.spinner("Analyzing Graph & Vectors..."):
        # Pass None if "All" is selected to search across all modes
        mode_filter = None if selected_mode == "All" else selected_mode
        answer, evidence = search_engine.search(query, mode_filter=mode_filter)
    
    # Display AI-generated summary
    st.markdown(f"### 🤖 Analysis:\n{answer}")
    
    st.divider()
    
    # Display evidence clips
    if evidence:
        st.subheader(f"Visual Evidence ({len(evidence)} Matches)")
        
        cols = st.columns(DISPLAY_COLUMNS)
        for idx, ev in enumerate(evidence):
            with cols[idx % DISPLAY_COLUMNS]:
                video_name = ev.get('video', 'Unknown')
                st.markdown(f"**Result {idx+1}** from `{video_name}`")
                
                # Construct video path
                video_path = VIDEO_STORAGE_DIR / video_name
                
                if video_path.exists():
                    # Extract start time for video player
                    try:
                        start_sec = float(ev['time'].split('s')[0])
                    except:
                        start_sec = 0
                    st.video(str(video_path), start_time=int(start_sec))
                else:
                    st.error(f"Video file missing: {video_name}")
                
                # Show detailed metadata
                with st.expander("Read Details"):
                    st.caption(ev['description'])
                    st.markdown(f"**Mode:** {ev.get('mode', 'N/A')}")
                    
                    # Display YOLO detected objects
                    tags = ev.get('yolo_tags', [])
                    if tags:
                        st.markdown(f"**YOLO Detected:** {', '.join(tags)}")
                    
                    # Display detected persons
                    persons = ev.get('persons', [])
                    if persons:
                        st.markdown(f"**Persons:** {', '.join(persons)}")
    else:
        st.warning("No relevant footage found.")
elif query:
    st.warning("⚠️ No video database found. Please upload and process videos first.")

