import cv2
import time
import uuid
import json
import logging
from datetime import timedelta
import numpy as np
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from insightface.app import FaceAnalysis
from database import ForensicDB
from dotenv import load_dotenv
import os

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Gemini VLM
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# OLD: model = genai.GenerativeModel('gemini-1.5-flash')
# NEW:
model = genai.GenerativeModel('gemini-2.0-flash')
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

class VideoIngestor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.db = ForensicDB()
        
        # Initialize Local Models (M1 Optimized)
        logger.info("Loading YOLO11 on MPS...")
        self.yolo = YOLO("yolo11n.pt") # Nano for speed
        
        logger.info("Loading InsightFace...")
        # Use CPU provider for Face Analysis on M1 (more stable than CoreML for this lib)
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        buffer_frames = []
        buffer_person_ids = set()
        last_capture_time = -10
        scene_start_time = 0
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            
            # 1. Event Trigger Logic (Run every 5th frame for speed)
            if frame_idx % 5 == 0:
                results = self.yolo(frame, verbose=False, device='mps')
                detections = results[0].boxes.cls.cpu().numpy() # Classes
                
                # Check for Persons (0), Cars (2), Backpacks (24), Knives (43)
                relevant_classes = {0, 2, 24, 43} 
                is_event = any(cls in relevant_classes for cls in detections)
                
                should_capture = False
                trigger_type = "None"
                
                # Rule A: Evidence (Event + 1s cooldown)
                if is_event and (current_time - last_capture_time > 1.0):
                    should_capture = True
                    trigger_type = "Event"
                
                # Rule B: Heartbeat (Empty scene + 10s cooldown)
                elif (current_time - last_capture_time > 10.0):
                    should_capture = True
                    trigger_type = "Heartbeat"
                
                if should_capture:
                    last_capture_time = current_time
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    buffer_frames.append(pil_img)
                    
                    # Run Face ID if person detected
                    if 0 in detections:
                        faces = self.face_app.get(frame)
                        for face in faces:
                            # Simple hash of embedding for ID (In prod, use vector search)
                            # Taking first 8 chars of hash for readability
                            fid = f"PID_{abs(hash(face.embedding.tobytes())) % 100000}"
                            buffer_person_ids.add(fid)
                    
                    logger.info(f"Captured {trigger_type} at {current_time:.2f}s")

            # 2. Batch Processing (Gemini) - Every 60 seconds of video or 15 frames
            if len(buffer_frames) >= 10 or (current_time - scene_start_time > 60 and buffer_frames):
                self._flush_buffer(buffer_frames, buffer_person_ids, scene_start_time, current_time)
                # Reset Buffer
                buffer_frames = []
                buffer_person_ids = set()
                scene_start_time = current_time
                
            frame_idx += 1
        # === FIX: FORCE FLUSH THE FINAL BUFFER ===
        if buffer_frames:
            logger.info("Flushing final batch...")
            self._flush_buffer(buffer_frames, buffer_person_ids, scene_start_time, frame_idx / fps)
        # =========================================    
        cap.release()
        self.db.close()
        logger.info("Ingestion Complete.")

    def _flush_buffer(self, frames, person_ids, start, end):
        logger.info(f"Analyzing Scene {start:.1f}s - {end:.1f}s with Gemini...")
        
        prompt = """
        # NEUTRAL PROMPT (No more "Forensic" bias)
        You are a video analyst. Analyze these CCTV keyframes.
        1. Describe the scene activity (traffic flow, pedestrian movement, weather).
        2. List all visible objects (cars, trucks, people, bicycles).
        3. Note any specific events (car parking, person entering building).
        
        Output pure JSON:
        {
            "summary": "Detailed, neutral description of the scene...",
            "activity_level": "Low/Medium/High",
            "objects_detected": ["white_van", "pedestrians", "trees"],
            "event_type": "Traffic/Routine/Loitering"
        }
        """
        try:
            response = model.generate_content(frames + [prompt], safety_settings=safety_settings)
            
            # 1. robust cleaning
            raw_text = response.text.strip()
            # Remove markdown code blocks if present
            if "```" in raw_text:
                raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
            
            # 2. Parse
            analysis = json.loads(raw_text)
            
            scene_id = str(uuid.uuid4())
            self.db.add_scene_node(
                self.video_name, 
                scene_id, 
                start, 
                end, 
                analysis.get('summary', 'No summary provided'), 
                list(person_ids)
            )
            logger.info(f"Saved Scene {scene_id}: {analysis.get('summary', '')[:50]}...")
            
        except Exception as e:
            # Print the RAW text so we can see why it failed
            logger.error(f"Gemini Error: {e}")
            if 'response' in locals() and response.text:
                logger.error(f"Raw Gemini Output was: {response.text[:100]}...")

if __name__ == "__main__":
    # Test run
    ingestor = VideoIngestor("test_video.mp4")
    ingestor.process_video()