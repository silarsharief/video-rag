import cv2
import time
import uuid
import json
import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from insightface.app import FaceAnalysis
from database import ForensicDB
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Using 2.0-flash as it appeared in your available list
model = genai.GenerativeModel('gemini-flash-latest')

MODES = {
    "traffic": { "model": "yolo11n.pt", "prompt": "Traffic Analyst. Identify vehicles/colors.", "classes": [2, 3, 5, 7] },
    "factory": { "model": "yolo11n.pt", "prompt": "Safety Inspector. PPE violations.", "classes": [0] },
    "kitchen": { "model": "yolo11n.pt", "prompt": "Health Inspector. Hygiene/Safety.", "classes": [0, 39, 41] },
    "general": { "model": "yolo11n.pt", "prompt": "Describe scene.", "classes": list(range(80)) }
}

class VideoIngestor:
    def __init__(self, video_path, mode="general"):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.mode = mode
        self.config = MODES.get(mode, MODES["general"])
        self.db = ForensicDB()
        self.yolo = YOLO(self.config['model']) 
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): return

        fps = cap.get(cv2.CAP_PROP_FPS)
        buffer_frames = []
        buffer_person_ids = set()
        buffer_object_tags = set()
        last_capture_time = -10
        scene_start_time = 0
        target_classes = set(self.config['classes'])
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            
            # Inference every 10th frame (Optimized for speed)
            if frame_idx % 10 == 0:
                results = self.yolo(frame, verbose=False)
                detections = results[0].boxes
                
                found_interesting = False
                for box in detections:
                    if int(box.cls[0]) in target_classes and float(box.conf[0]) > 0.5:
                        found_interesting = True
                        if int(box.cls[0]) != 0: 
                            buffer_object_tags.add(results[0].names[int(box.cls[0])])
                
                if found_interesting and (current_time - last_capture_time > 2.0):
                    last_capture_time = current_time
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buffer_frames.append(Image.fromarray(rgb))
                    
                    if 0 in detections.cls:
                        faces = self.face_app.get(frame)
                        for face in faces:
                            buffer_person_ids.add(f"PID_{abs(hash(face.embedding.tobytes())) % 100000}")

            # === BATCH SIZE 5 ===
            if len(buffer_frames) >= 5: 
                self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, current_time)
                buffer_frames = [] # Clear buffer
                buffer_person_ids = set()
                buffer_object_tags = set()
                scene_start_time = current_time
                
            frame_idx += 1
            
        if buffer_frames:
            self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, frame_idx / fps)
        cap.release()
        self.db.close()

    def _flush_buffer(self, frames, pids, tags, start, end):
        logger.info(f"Analyzing {start:.1f}s - {end:.1f}s...")
        prompt = self.config["prompt"] + " Output pure JSON: { 'summary': '...', 'objects': [] }"
        
        try:
            # === CRITICAL FIX: FORCED SLEEP ===
            # Wait 5 seconds before hitting Google to respect the 15 RPM limit
            time.sleep(5) 
            
            response = model.generate_content(frames + [prompt])
            text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            
            sid = str(uuid.uuid4())
            self.db.add_scene_node(self.video_name, self.mode, sid, start, end, 
                                   data.get('summary', 'No summary'), list(pids), list(tags))
        except Exception as e:
            logger.error(f"Gemini Error: {e}")