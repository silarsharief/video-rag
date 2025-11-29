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
    
    # ... inside app.py ...
    
    st.divider()
    st.subheader(f"Visual Evidence ({len(evidence)} Matches Found)")
    
    cols = st.columns(2) # 2 columns looks better than 3 for video
    for idx, ev in enumerate(evidence):
        with cols[idx % 2]:
            # 1. Parse the Timestamps (e.g., "10.4s - 22.4s")
            try:
                time_range = ev['time'].replace('s', '').split(' - ')
                start_sec = float(time_range[0])
                end_sec = float(time_range[1])
            except:
                start_sec = 0
                end_sec = 0
            
            # 2. Display the Timestamp Label
            st.markdown(f"**Event {idx+1}:** `{ev['time']}`")
            
            # 3. The "Clip" Trick
            # We open the local file, read the bytes, but display it with start_time
            # Note: Streamlit doesn't support 'end_time' strictly, but this sets the start.
            st.video("temp_video.mp4", start_time=int(start_sec))
            
            # 4. Detailed Description
            with st.expander("Read Scene Details"):
                st.caption(ev['description'])
                st.markdown(f"**People Detected:** {len(ev.get('persons_present', []))}")