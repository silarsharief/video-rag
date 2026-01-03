"""
Video Ingestion Test Script
Tests ingestion of a single video file.
"""
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.ingest import VideoIngestor
from config.settings import VIDEO_STORAGE_DIR

# Test video path
video_path = VIDEO_STORAGE_DIR / "2_tr100.mp4"

if not video_path.exists():
    print(f"❌ Error: File '{video_path}' not found.")
    print(f"Please put a video file in {VIDEO_STORAGE_DIR}/")
else:
    print(f"🚀 Starting Ingestion for {video_path}...")
    ingestor = VideoIngestor(str(video_path), mode="general")
    ingestor.process_video()
    print("✅ Finished!")

