import streamlit as st
import os
from ingest import VideoIngestor
from retrieval import ForensicSearch

st.set_page_config(layout="wide", page_title="Forensic RAG")

st.title("🕵️ Edge-First Forensic Video Analysis")

# Ensure storage directory exists
VIDEO_DIR = "stored_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Sidebar: Ingestion
with st.sidebar:
    st.header("1. Configuration")
    # Update the selectbox options
    selected_mode = st.selectbox("Select Use Case Mode", ["traffic", "factory", "kitchen", "general"])

    st.header("2. Upload Evidence")
    # UPDATED: accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload CCTV Footage", type=['mp4', 'mov'], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} files selected.")
        
        if st.button("Process All Videos"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # 1. Save file
                file_path = os.path.join(VIDEO_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Update Status
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: '{uploaded_file.name}' in {selected_mode} mode...")
                
                # 3. Ingest
                try:
                    ingestor = VideoIngestor(file_path, mode=selected_mode)
                    ingestor.process_video()
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                
                # 4. Update Progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success("All videos processed successfully!")

# Main: Search
st.header("2. Forensic Search")
query = st.text_input("Ask a question about the footage:", "Show me all cars.")

if query and os.path.exists("chromadb"):
    search_engine = ForensicSearch()
    with st.spinner("Analyzing Graph & Vectors..."):
        # Search runs across ALL ingested videos automatically
        answer, evidence = search_engine.search(query, mode_filter=selected_mode)
    
    st.markdown(f"### 🤖 Analysis:\n{answer}")
    
    st.divider()
    if evidence:
        st.subheader(f"Visual Evidence ({len(evidence)} Matches)")
        
        cols = st.columns(2)
        for idx, ev in enumerate(evidence):
            with cols[idx % 2]:
                video_name = ev.get('video', 'Unknown')
                st.markdown(f"**Result {idx+1}** from `{video_name}`")
                
                # Play the specific file for this result
                video_path = os.path.join(VIDEO_DIR, video_name)
                
                if os.path.exists(video_path):
                    try:
                        start_sec = float(ev['time'].split('s')[0])
                    except:
                        start_sec = 0
                    st.video(video_path, start_time=int(start_sec))
                else:
                    st.error(f"Video file missing: {video_name}")
                
                with st.expander("Read Details"):
                    st.caption(ev['description'])
                    st.markdown(f"**Mode:** {ev.get('mode', 'N/A')}")
                    tags = ev.get('yolo_tags', [])
                    if tags:
                        st.markdown(f"**YOLO Detected:** {', '.join(tags)}")
    else:
        st.warning("No relevant footage found.")