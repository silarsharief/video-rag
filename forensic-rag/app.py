import streamlit as st
import os
from ingest import VideoIngestor
from retrieval import ForensicSearch

st.set_page_config(layout="wide", page_title="Forensic RAG")

st.title("🕵️ Edge-First Forensic Video Analysis")

# Sidebar: Ingestion
with st.sidebar:
    st.header("1. Upload Evidence")
    uploaded_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'mov'])
    
    if uploaded_file:
        # Save temp file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process Video (Ingest)"):
            with st.status("Processing on Neural Engine...", expanded=True) as status:
                st.write("Initializing YOLO & InsightFace...")
                ingestor = VideoIngestor("temp_video.mp4")
                st.write("Extracting Keyframes & Vectors...")
                ingestor.process_video()
                status.update(label="Ingestion Complete!", state="complete", expanded=False)

# Main: Search
st.header("2. Forensic Search")
query = st.text_input("Ask a question about the footage:", "Show me who stole the bag.")

if query and os.path.exists("chromadb"):
    search_engine = ForensicSearch()
    with st.spinner("Analyzing Graph & Vectors..."):
        answer, evidence = search_engine.search(query)
    
    # Display Results
    st.markdown(f"### 🤖 Analysis:\n{answer}")
    
    st.divider()
    st.subheader("Visual Evidence")
    
    cols = st.columns(3)
    for idx, ev in enumerate(evidence):
        with cols[idx]:
            st.info(f"Timestamp: {ev['time']}")
            # Parse start time for video seek
            start_sec = float(ev['time'].split('s')[0])
            st.video("temp_video.mp4", start_time=int(start_sec))
            st.caption(ev['description'])