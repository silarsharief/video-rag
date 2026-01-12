"""
Forensic RAG - Main Streamlit Application
Edge-First Forensic Video Analysis System
"""
import streamlit as st
import os
import sys
import re
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

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Card-like containers with subtle shadow */
    .stContainer {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    /* Expander styling - make headers more prominent */
    .streamlit-expanderHeader {
        font-size: 14px;
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }
    
    /* Improved dividers */
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #dee2e6;
    }
    
    /* Better alert boxes */
    .stAlert {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    /* Error alerts (violations) */
    div[data-baseweb="notification"] > div:has(svg[data-testid="stErrorIcon"]) {
        border-left-color: #dc3545;
    }
    
    /* Success alerts (compliance) */
    div[data-baseweb="notification"] > div:has(svg[data-testid="stSuccessIcon"]) {
        border-left-color: #28a745;
    }
    
    /* Info alerts (context) */
    div[data-baseweb="notification"] > div:has(svg[data-testid="stInfoIcon"]) {
        border-left-color: #17a2b8;
    }
    
    /* Video player styling */
    video {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        width: 100%;
    }
    
    /* Better spacing for captions */
    .stCaption {
        margin-top: 8px;
        color: #6c757d;
        font-size: 13px;
    }
    
    /* Improve column spacing */
    div[data-testid="column"] {
        padding: 0 12px;
    }
    
    /* Scene title styling */
    h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

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
            
            # Track processing statistics
            processed_count = 0
            skipped_count = 0
            failed_count = 0
            
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
                    result = ingestor.process_video()
                    
                    # Handle different processing results
                    if result and result.get('status') == 'skipped':
                        skipped_count += 1
                        st.info(f"ℹ️ **{uploaded_file.name}**: Already processed (skipped)")
                        st.caption(f"Originally processed on {result.get('previous_processing', {}).get('processed_at', 'unknown date')}")
                    elif result and result.get('status') == 'completed':
                        processed_count += 1
                        st.success(f"✅ **{uploaded_file.name}**: Processed successfully")
                        st.caption(f"{result.get('scene_count', 0)} scenes, {result.get('detection_count', 0)} detections")
                    else:
                        processed_count += 1  # Count as processed even if no result returned (backward compatibility)
                        
                except Exception as e:
                    failed_count += 1
                    st.error(f"❌ **{uploaded_file.name}**: Failed to process")
                    st.caption(f"Error: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show final summary
            status_text.empty()
            st.success(f"✅ **Processing Complete!**")
            st.write(f"- ✅ **Processed:** {processed_count}")
            st.write(f"- ⏭️ **Skipped (duplicates):** {skipped_count}")
            if failed_count > 0:
                st.write(f"- ❌ **Failed:** {failed_count}")

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
    
    # Parse analysis into scene-specific data for matching with evidence
    def parse_analysis(analysis_text):
        """Parse analysis text into per-scene dictionaries"""
        lines = analysis_text.split('\n')
        scenes_data = {}
        current_scene_num = None
        current_content = []
        general_analysis = []
        in_general = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect scene/video headers and extract scene number
            if any(marker in line.upper() for marker in ['SCENE', 'VIDEO']) and any(char.isdigit() for char in line[:30]):
                # Save previous scene
                if current_scene_num and current_content:
                    scenes_data[current_scene_num] = '\n'.join(current_content)
                
                # Extract scene number (1, 2, 3, etc.)
                import re
                match = re.search(r'(\d+)', line[:30])
                current_scene_num = int(match.group(1)) if match else None
                current_content = []
                in_general = False
                
            elif 'GENERAL ANALYSIS' in line.upper():
                # Save last scene before general analysis
                if current_scene_num and current_content:
                    scenes_data[current_scene_num] = '\n'.join(current_content)
                in_general = True
                current_scene_num = None
                
            elif in_general:
                general_analysis.append(line)
            elif current_scene_num:
                current_content.append(line)
        
        # Save last scene
        if current_scene_num and current_content:
            scenes_data[current_scene_num] = '\n'.join(current_content)
        
        return scenes_data, '\n'.join(general_analysis)
    
    # Parse the analysis
    scenes_analysis, general_text = parse_analysis(answer)
    
    # Display evidence clips with integrated analysis
    if evidence:
        st.markdown(f"### 📹 Found {len(evidence)} Scene(s)")
        st.markdown("")  # Spacing
        
        cols = st.columns(DISPLAY_COLUMNS)
        for idx, ev in enumerate(evidence):
            with cols[idx % DISPLAY_COLUMNS]:
                video_name = ev.get('video', 'Unknown')
                mode = ev.get('mode', 'N/A')
                time = ev.get('time', 'N/A')
                scene_num = idx + 1
                
                # Card-like container
                with st.container():
                    # Header with mode emoji and scene number
                    mode_emoji = {"traffic": "🚗", "factory": "🏭", "kitchen": "🍳", "general": "📽️"}
                    emoji = mode_emoji.get(mode, "📽️")
                    st.markdown(f"### {emoji} Scene {scene_num}")
                    st.caption(f"📂 `{video_name}` • ⏱️ {time} • 🏷️ {mode.title()}")
                    
                    # Construct video path
                    video_path = VIDEO_STORAGE_DIR / video_name
                    
                    if video_path.exists():
                        # Extract start time for video player
                        try:
                            start_sec = float(time.split('s')[0])
                        except:
                            start_sec = 0
                        st.video(str(video_path), start_time=int(start_sec))
                    else:
                        st.error(f"❌ Video file missing: {video_name}")
                    
                    # Display AI analysis for this scene (if available)
                    scene_analysis_text = scenes_analysis.get(scene_num, '')
                    if scene_analysis_text:
                        st.markdown("#### 🤖 AI Analysis")
                        
                        # Parse and format the analysis
                        for line in scene_analysis_text.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            
                            line_lower = line.lower()
                            if line_lower.startswith('violations:'):
                                violations_text = line.split(':', 1)[1].strip()
                                if violations_text and violations_text.lower() not in ['none', '']:
                                    st.error(f"🚨 {violations_text}")
                                else:
                                    st.success("✅ No violations detected")
                                    
                            elif line_lower.startswith('compliance:'):
                                compliance_text = line.split(':', 1)[1].strip()
                                if compliance_text and compliance_text.lower() not in ['none', '']:
                                    st.success(f"✅ {compliance_text}")
                                    
                            elif line_lower.startswith('context:'):
                                context_text = line.split(':', 1)[1].strip()
                                st.info(f"ℹ️ {context_text}")
                                
                            elif line_lower.startswith('key findings:'):
                                findings_text = line.split(':', 1)[1].strip()
                                st.write(findings_text)
                                
                            elif line.startswith('-') or line.startswith('•'):
                                st.markdown(f"  {line}")
                            else:
                                st.write(line)
                    
                    # Expandable technical details
                    with st.expander("🔍 Technical Details", expanded=False):
                        st.markdown("**Scene Description:**")
                        st.caption(ev['description'])
                        
                        st.markdown("")  # Spacing
                        
                        # Display YOLO detected objects as readable badges
                        tags = ev.get('yolo_tags', [])
                        if tags:
                            st.markdown("**🎯 Objects Detected:**")
                            tags_html = " ".join([
                                f'<span style="background-color:#2c5aa0;color:white;padding:5px 10px;border-radius:6px;margin:3px;display:inline-block;font-size:13px;font-weight:500;">{tag}</span>' 
                                for tag in tags
                            ])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        
                        st.markdown("")  # Spacing
                        
                        # Display detected persons with readable badges
                        persons = ev.get('persons', [])
                        if persons:
                            st.markdown("**👤 Persons Identified:**")
                            persons_html = " ".join([
                                f'<span style="background-color:#d97706;color:white;padding:5px 10px;border-radius:6px;margin:3px;display:inline-block;font-size:13px;font-weight:500;">{p}</span>' 
                                for p in persons
                            ])
                            st.markdown(persons_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
        
        # Display general analysis if present
        if general_text:
            st.markdown("---")
            with st.expander("📊 Overall Analysis", expanded=False):
                for line in general_text.split('\n'):
                    line = line.strip()
                    if line:
                        if 'overall summary:' in line.lower():
                            st.markdown(f"**{line}**")
                        else:
                            st.markdown(line)
    else:
        st.warning("⚠️ No relevant footage found.")
elif query:
    st.warning("⚠️ No video database found. Please upload and process videos first.")

