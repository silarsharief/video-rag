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
model = genai.GenerativeModel('gemini-2.0-flash')
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# === PROMPT CONFIGURATIONS ===
# === PROMPT CONFIGURATIONS ===

# Helper: Standard COCO Class IDs for reference
# 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light, 
# 10:fire hydrant, 11:stop sign, 24:backpack, 26:handbag, 28:suitcase, 
# 39:bottle, 40:wine glass, 41:cup, 42:fork, 43:knife, 44:spoon, 45:bowl, 
# 46-55: foods (banana, apple, sandwich, orange, broccoli...), 
# 66:mouse (pest), 67:remote, 68:microwave, 69:oven, 70:toaster, 71:sink, 72:refrigerator

MODES = {
    "traffic": {
        "model": "yolo11x.pt", 
        "prompt": """
            You are a Traffic Analyst.
            1. Identify vehicle types, COLORS, and pedestrians.
            2. Note any traffic violations (running lights, jaywalking).
            3. Describe the flow and density of the traffic.
            Output pure JSON with summary.
        """,
        # EXPANDED: Vehicles + Pedestrians + Signs + Lights + Bags (for finding specific people)
        "classes": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 24, 26, 28] 
    },
    
    "factory": {
        "model": "yolo_ppe.pt", # Custom PPE Model
        "prompt": """
            You are a Safety Inspector. 
            The system has already detected specific PPE items.
            1. Describe the worker's activity.
            2. Verify violations: If 'no_helmet' or 'no_mask' is detected, confirm it.
            3. Check for environmental hazards (spills, smoke, blocked paths).
            Output pure JSON with summary.
        """,
        # EXPANDED: Enable ALL custom classes from your PPE model (0-9)
        # 0:glove, 1:goggles, 2:helmet, 3:mask, 4:no_glove, 
        # 5:no_goggles, 6:no_helmet, 7:no_mask, 8:no_shoes, 9:shoes
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    },
    
    "kitchen": {
        "model": "yolo11x.pt", # Use 'x' for better small object detection (knives/pests)
        "prompt": """
            You are a Health Inspector. Analyze this kitchen.
            1. HYGIENE: Are staff detected? Are they wearing gloves/hairnets?
            2. SAFETY: Look for 'knife' objects left unsafe. Check for fire/smoke.
            3. PESTS: Look for mice/rats.
            4. CLEANLINESS: Check for spills or clutter on tables.
            Output pure JSON with summary.
        """,
        # EXPANDED: People + Bottles + Cups + Cutlery + Food + Pests + Appliances
        "classes": [0, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 66, 68, 69, 70, 71, 72]
    },
    
    # NEW: "General" Mode (Catch-all for any other video)
    "general": {
        "model": "yolo11x.pt",
        "prompt": """
            You are a General Video Observer. 
            1. Describe the scene, location, and mood.
            2. List the main objects and actions occurring.
            3. Note any anomalies or interesting events.
            Output pure JSON with summary.
        """,
        # ENABLE EVERYTHING: 0 to 79 (All COCO classes)
        "classes": list(range(80)) 
    }
}

class VideoIngestor:
    def __init__(self, video_path, mode="traffic"):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.mode = mode
        self.config = MODES[mode] # Load Config
        self.db = ForensicDB()
        
        # Load Model based on Mode
        # Note: If yolo11x.pt isn't downloaded, it will try to download it.
        # Ensure you have the file or internet access.
        model_name = self.config['model']
        logger.info(f"Loading {model_name} for {mode} mode on MPS...")
        
        # Fallback to yolo11n.pt if x is missing for demo stability, or ensure x is present
        if not os.path.exists(model_name) and model_name == "yolo11x.pt":
             logger.warning(f"{model_name} not found locally. Defaulting to yolo11n.pt for demo.")
             model_name = "yolo11n.pt"

        self.yolo = YOLO(model_name) 
        
        logger.info("Loading InsightFace...")
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        buffer_frames = []
        buffer_person_ids = set()
        buffer_object_tags = set()
        last_capture_time = -10
        scene_start_time = 0
        
        # Get relevant classes for this mode
        target_classes = set(self.config['classes'])
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            
            # 1. Event Trigger Logic (Run every 5th frame)
            if frame_idx % 5 == 0:
                results = self.yolo(frame, verbose=False, device='mps')
                
                detections = results[0].boxes
                names = results[0].names
                
                found_interesting_object = False
                
                for box in detections:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    
                    # Check if this object is relevant to our current MODE
                    if conf > 0.5 and cls_id in target_classes:
                        found_interesting_object = True
                        if cls_id != 0: 
                            buffer_object_tags.add(label)
                    # Also always capture High Conf Persons for FaceID
                    elif conf > 0.5 and cls_id == 0:
                         found_interesting_object = True

                should_capture = False
                trigger_type = "None"
                
                if found_interesting_object and (current_time - last_capture_time > 1.0):
                    should_capture = True
                    trigger_type = "Event"
                elif (current_time - last_capture_time > 10.0):
                    should_capture = True
                    trigger_type = "Heartbeat"
                
                if should_capture:
                    last_capture_time = current_time
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    buffer_frames.append(pil_img)
                    
                    if 0 in detections.cls:
                        faces = self.face_app.get(frame)
                        for face in faces:
                            fid = f"PID_{abs(hash(face.embedding.tobytes())) % 100000}"
                            buffer_person_ids.add(fid)
                    
                    logger.info(f"Captured {trigger_type} at {current_time:.2f}s | Mode: {self.mode}")

            # 2. Batch Processing
            if len(buffer_frames) >= 5 or (current_time - scene_start_time > 60 and buffer_frames):
                self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, current_time)
                buffer_frames = []
                buffer_person_ids = set()
                buffer_object_tags = set()
                scene_start_time = current_time
                
            frame_idx += 1
            
        if buffer_frames:
            logger.info("Flushing final batch...")
            self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, frame_idx / fps)
            
        cap.release()
        self.db.close()
        logger.info("Ingestion Complete.")

    def _flush_buffer(self, frames, person_ids, object_tags, start, end):
        logger.info(f"Analyzing Scene {start:.1f}s - {end:.1f}s with Gemini ({self.mode} mode)...")
        
        # Use the specific prompt from the config
        base_prompt = self.config["prompt"]
        
        # Append JSON requirement to ensure structured output
        full_prompt = base_prompt + """
        
        Output pure JSON format:
        {
            "summary": "Concise summary of events observed.",
            "objects_detected": ["list", "of", "objects"],
            "safety_status": "Safe/Violation/Unknown"
        }
        """
        
        try:
            response = model.generate_content(frames + [full_prompt], safety_settings=safety_settings)
            
            raw_text = response.text.strip()
            if "```" in raw_text:
                raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
            
            analysis = json.loads(raw_text)
            
            scene_id = str(uuid.uuid4())
            
            # Pass MODE to the database
            self.db.add_scene_node(
                self.video_name, 
                self.mode, # <--- Passing Mode
                scene_id, 
                start, 
                end, 
                analysis.get('summary', 'No summary provided'), 
                list(person_ids),
                list(object_tags)
            )
            logger.info(f"Saved Scene {scene_id}: {analysis.get('summary', '')[:50]}...")
            
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            if 'response' in locals() and response.text:
                logger.error(f"Raw Gemini Output: {response.text[:100]}...")

if __name__ == "__main__":
    # Test run
    ingestor = VideoIngestor("test_video.mp4", mode="traffic")
    ingestor.process_video()