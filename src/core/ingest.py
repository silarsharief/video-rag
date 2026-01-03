"""
Video Ingestion Module
Processes video files, extracts frames, detects objects/faces, and stores metadata.
"""
import cv2
import time
import uuid
import json
import logging
import re
import numpy as np
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from insightface.app import FaceAnalysis
import os

# Import centralized configurations
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_BUFFER_DELAY,
    MODELS_DIR,
    FRAME_SKIP_INTERVAL,
    MIN_DETECTION_CONFIDENCE,
    SCENE_CAPTURE_INTERVAL,
    BATCH_SIZE,
    FACE_DETECTION_SIZE,
    FACE_CTX_ID,
    LOG_LEVEL
)
from config.modes import get_mode_config
from src.core.database import ForensicDB

# Setup logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


class VideoIngestor:
    """
    Processes video files and stores scene analysis in databases.
    Supports multiple processing modes (traffic, factory, kitchen, general).
    """
    
    def __init__(self, video_path, mode="general"):
        """
        Initialize video ingestor.
        
        Args:
            video_path (str): Path to video file
            mode (str): Processing mode (traffic, factory, kitchen, general)
        """
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.mode = mode
        self.config = get_mode_config(mode)
        
        # Log initialization details
        logger.info(f"🎬 Initializing VideoIngestor for: {self.video_name}")
        logger.info(f"📋 Mode: {mode}")
        logger.info(f"🎯 Target Classes: {self.config['classes']}")
        
        self.db = ForensicDB()
        
        # Load YOLO model from models directory
        model_name = self.config['model']
        model_path = MODELS_DIR / model_name
        logger.info(f"🔍 Loading YOLO Model: {model_name}")
        logger.info(f"📂 Model Path: {model_path}")
        
        if not model_path.exists():
            logger.warning(f"⚠️ Model file not found at {model_path}. YOLO will attempt to download it.")
        
        self.yolo = YOLO(str(model_path))
        logger.info(f"✅ YOLO Model Loaded Successfully: {model_name}")
        
        # Initialize face recognition
        logger.info(f"👤 Initializing Face Recognition (InsightFace)...")
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=FACE_CTX_ID, det_size=FACE_DETECTION_SIZE)
        logger.info(f"✅ Face Recognition Initialized")

    def process_video(self):
        """
        Main video processing pipeline.
        Extracts frames, detects objects, recognizes faces, and generates scene summaries.
        """
        logger.info(f"▶️ Starting video processing...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"📹 Video Info: {total_frames} frames @ {fps:.2f} fps ({duration:.1f}s)")
        logger.info(f"⚙️ Processing every {FRAME_SKIP_INTERVAL}th frame")
        
        buffer_frames = []
        buffer_person_ids = set()
        buffer_object_tags = set()
        last_capture_time = -10
        scene_start_time = 0
        target_classes = set(self.config['classes'])
        frame_idx = 0
        total_detections = 0
        total_faces = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            # Inference every Nth frame (Optimized for speed)
            if frame_idx % FRAME_SKIP_INTERVAL == 0:
                results = self.yolo(frame, verbose=False)
                detections = results[0].boxes
                
                found_interesting = False
                frame_detections = []
                
                for box in detections:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if cls_id in target_classes and confidence > MIN_DETECTION_CONFIDENCE:
                        found_interesting = True
                        obj_name = results[0].names[cls_id]
                        frame_detections.append(f"{obj_name}({confidence:.2f})")
                        total_detections += 1
                        
                        # Skip person class for object tags (handled separately by face detection)
                        if cls_id != 0: 
                            buffer_object_tags.add(obj_name)
                
                # Log detections if found
                if frame_detections:
                    logger.debug(f"🔍 Frame {frame_idx} ({current_time:.1f}s): Detected {', '.join(frame_detections)}")
                
                # Capture frame if interesting and enough time has passed
                if found_interesting and (current_time - last_capture_time > SCENE_CAPTURE_INTERVAL):
                    last_capture_time = current_time
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buffer_frames.append(Image.fromarray(rgb))
                    logger.debug(f"📸 Captured frame at {current_time:.1f}s")
                    
                    # Face detection (only if person detected)
                    if 0 in detections.cls:
                        faces = self.face_app.get(frame)
                        if faces:
                            logger.debug(f"👤 Detected {len(faces)} face(s) at {current_time:.1f}s")
                            total_faces += len(faces)
                        for face in faces:
                            # Generate unique person ID from face embedding
                            buffer_person_ids.add(f"PID_{abs(hash(face.embedding.tobytes())) % 100000}")

            # Flush buffer when batch size reached
            if len(buffer_frames) >= BATCH_SIZE: 
                self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, current_time)
                # Clear buffer
                buffer_frames = []
                buffer_person_ids = set()
                buffer_object_tags = set()
                scene_start_time = current_time
                
            frame_idx += 1
            
        # Process remaining frames in buffer
        if buffer_frames:
            logger.info(f"📦 Flushing remaining {len(buffer_frames)} frames...")
            self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, scene_start_time, frame_idx / fps)
        
        cap.release()
        self.db.close()
        
        # Log final statistics
        logger.info(f"✅ Finished processing: {self.video_name}")
        logger.info(f"📊 Total Statistics:")
        logger.info(f"   - Total Detections: {total_detections}")
        logger.info(f"   - Total Faces: {total_faces}")
        logger.info(f"   - Frames Processed: {frame_idx}")
        logger.info(f"   - Duration: {duration:.1f}s")

    def _flush_buffer(self, frames, pids, tags, start, end):
        """
        Send buffered frames to Gemini for analysis and store results.
        
        Args:
            frames (list): List of PIL Image objects
            pids (set): Set of detected person IDs
            tags (set): Set of detected object tags
            start (float): Scene start time
            end (float): Scene end time
        """
        logger.info(f"🤖 Analyzing scene: {start:.1f}s - {end:.1f}s")
        logger.info(f"   - Frames: {len(frames)}, PIDs: {len(pids)}, Objects: {list(tags)}")
        
        # Enhanced prompt with strict JSON formatting
        prompt = (
            self.config["prompt"] + 
            "\n\nIMPORTANT: Respond ONLY with valid JSON in this exact format:\n"
            '{"summary": "brief description here", "objects": ["object1", "object2"]}\n'
            "Do not include any markdown formatting or additional text."
        )
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(1, max_retries + 1):
            try:
                # Rate limiting (reduced since user upgraded limits)
                if attempt > 1:
                    logger.info(f"♻️ Retry attempt {attempt}/{max_retries}...")
                    time.sleep(retry_delay * attempt)
                else:
                    time.sleep(GEMINI_BUFFER_DELAY)
                
                # Call Gemini API
                logger.debug(f"📤 Sending {len(frames)} frames to Gemini...")
                response = model.generate_content(frames + [prompt])
                
                # Check if response is valid
                if not response or not hasattr(response, 'text'):
                    raise ValueError("Empty or invalid response from Gemini")
                
                raw_text = response.text
                logger.debug(f"📥 Gemini Raw Response: {raw_text[:200]}...")  # Log first 200 chars
                
                # Clean response using regex (more robust than simple replace)
                # Remove markdown code blocks
                cleaned_text = re.sub(r'```json\s*', '', raw_text)
                cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                cleaned_text = cleaned_text.strip()
                
                # Try to find JSON object in the response
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(0)
                
                # Parse JSON
                data = json.loads(cleaned_text)
                logger.info(f"✅ Gemini Response Parsed Successfully")
                logger.info(f"   Summary: {data.get('summary', 'N/A')[:100]}...")  # Log first 100 chars
                
                # Generate unique scene ID
                sid = str(uuid.uuid4())
                
                # Store in databases
                self.db.add_scene_node(
                    self.video_name, 
                    self.mode, 
                    sid, 
                    start, 
                    end, 
                    data.get('summary', 'No summary'), 
                    list(pids), 
                    list(tags)
                )
                logger.info(f"💾 Scene data stored successfully (ID: {sid[:8]}...)")
                
                # Success - break retry loop
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON Parsing Error (attempt {attempt}/{max_retries}): {e}")
                logger.error(f"   Problematic text: {cleaned_text[:200] if 'cleaned_text' in locals() else 'N/A'}")
                
                if attempt == max_retries:
                    # Final attempt failed - store with fallback data
                    logger.warning(f"⚠️ Using fallback data for scene {start:.1f}s - {end:.1f}s")
                    sid = str(uuid.uuid4())
                    self.db.add_scene_node(
                        self.video_name, 
                        self.mode, 
                        sid, 
                        start, 
                        end, 
                        f"[Parse Error] Scene from {start:.1f}s to {end:.1f}s with objects: {', '.join(tags)}", 
                        list(pids), 
                        list(tags)
                    )
                    
            except Exception as e:
                logger.error(f"❌ Gemini API Error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
                
                if attempt == max_retries:
                    # Final attempt failed - store with fallback data
                    logger.warning(f"⚠️ Using fallback data for scene {start:.1f}s - {end:.1f}s")
                    sid = str(uuid.uuid4())
                    self.db.add_scene_node(
                        self.video_name, 
                        self.mode, 
                        sid, 
                        start, 
                        end, 
                        f"[API Error] Scene from {start:.1f}s to {end:.1f}s with objects: {', '.join(tags)}", 
                        list(pids), 
                        list(tags)
                    )

