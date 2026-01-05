"""
Video Ingestion Module
Processes video files, extracts frames, detects objects/faces, and stores metadata.
"""
# Workaround for PyTorch 2.7+ weights_only default change (needed for ppe.pt)
import torch
_original_load = torch.load

def _patched_load(*args, **kwargs):
    """Patch torch.load to use weights_only=False for model loading"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load

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
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BASE_DELAY,
    GEMINI_RETRY_MULTIPLIER,
    GEMINI_RETRY_MAX_DELAY,
    ENABLE_BATCH_QUEUE,
    QUEUE_RETRY_DELAY,
    MAX_FRAME_WIDTH,
    MAX_FRAME_HEIGHT,
    JPEG_QUALITY,
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
        
        # Initialize batch queue for failed API calls
        self.failed_batch_queue = []  # Queue to store failed batches for retry
        
        # Log model class names to verify correct model is loaded (especially for PPE model)
        if hasattr(self.yolo, 'names') and self.yolo.names:
            class_count = len(self.yolo.names)
            logger.info(f"📋 Model has {class_count} classes")
            # For PPE model, log all class names to verify
            if "ppe" in model_name.lower() or mode == "factory":
                logger.info(f"🦺 PPE Model Classes: {dict(self.yolo.names)}")
            else:
                # For other models, just log first few classes
                sample_classes = {k: v for k, v in list(self.yolo.names.items())[:5]}
                logger.info(f"📦 Sample Classes: {sample_classes}...")
        
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
        logger.info(f"🎯 Using Model: {self.config['model']} for mode: {self.mode}")
        logger.info(f"🎯 Target Classes: {self.config['classes']}")
        logger.info(f"🎯 Model Classes Available: {dict(self.yolo.names) if hasattr(self.yolo, 'names') else 'N/A'}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Get frame skip interval from mode config, fallback to default
        frame_skip_interval = self.config.get('frame_skip_interval', FRAME_SKIP_INTERVAL)
        
        # Get confidence threshold from mode config, fallback to default
        conf_threshold = self.config.get('confidence_threshold', MIN_DETECTION_CONFIDENCE)
        
        logger.info(f"📹 Video Info: {total_frames} frames @ {fps:.2f} fps ({duration:.1f}s)")
        logger.info(f"⚙️ Processing every {frame_skip_interval}th frame (mode-specific)")
        logger.info(f"🎯 Confidence Threshold: {conf_threshold} (mode-specific: {self.config.get('confidence_threshold') is not None})")
        
        # Calculate how many frames will be processed
        frames_to_process = (total_frames // frame_skip_interval) + (1 if total_frames % frame_skip_interval > 0 else 0)
        logger.info(f"📊 Will process approximately {frames_to_process} frames out of {total_frames} total frames")
        
        buffer_frames = []
        buffer_person_ids = set()
        buffer_object_tags = set()
        buffer_ppe_detections = {}  # Track detailed PPE detections: {class_name: {'count': int, 'confidences': [float]}}
        last_capture_time = -10
        scene_start_time = 0
        target_classes = set(self.config['classes'])
        frame_idx = 0
        total_detections = 0
        total_faces = 0
        
        frames_processed_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Verify frame is valid
            if frame is None or frame.size == 0:
                logger.warning(f"⚠️ Frame {frame_idx} is None or empty, skipping")
                frame_idx += 1
                continue
            
            current_time = frame_idx / fps
            
            # Inference every Nth frame (Optimized for speed)
            if frame_idx % frame_skip_interval == 0:
                frames_processed_count += 1
                logger.info(f"🖼️ Processing frame {frame_idx} ({current_time:.1f}s) - Frame shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Pass confidence threshold to YOLO to filter low-confidence detections early
                # Use mode-specific threshold if available, otherwise use default
                try:
                    results = self.yolo(frame, conf=conf_threshold, verbose=False)
                    detections = results[0].boxes
                    logger.info(f"✅ YOLO inference completed for frame {frame_idx}")
                except Exception as e:
                    logger.error(f"❌ YOLO inference failed for frame {frame_idx}: {e}")
                    frame_idx += 1
                    continue
                
                # Log raw detection count for first few frames (especially for PPE model debugging)
                raw_detection_count = len(detections)
                if frame_idx < frame_skip_interval * 3 or self.mode == "factory":  # Log first 3 processed frames or all factory frames
                    logger.info(f"🔍 Frame {frame_idx} ({current_time:.1f}s): Model returned {raw_detection_count} raw detections")
                
                found_interesting = False
                frame_detections = []
                all_detections = []  # Track all detections for debugging
                
                # Track detailed PPE detections for factory mode
                frame_ppe_detections = {} if self.mode == "factory" else None
                
                for box in detections:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    obj_name = results[0].names.get(cls_id, f"unknown_{cls_id}")
                    
                    # Log all detections for debugging (especially for PPE model)
                    all_detections.append(f"{obj_name}({confidence:.2f})")
                    
                    if cls_id in target_classes and confidence > conf_threshold:
                        found_interesting = True
                        frame_detections.append(f"{obj_name}({confidence:.2f})")
                        total_detections += 1
                        
                        # Track detailed PPE detections for factory mode
                        if self.mode == "factory":
                            if obj_name not in frame_ppe_detections:
                                frame_ppe_detections[obj_name] = {'count': 0, 'confidences': []}
                            frame_ppe_detections[obj_name]['count'] += 1
                            frame_ppe_detections[obj_name]['confidences'].append(confidence)
                        
                        # Skip person class for object tags (handled separately by face detection)
                        if cls_id != 0: 
                            buffer_object_tags.add(obj_name)
                
                # Aggregate PPE detections into buffer
                if self.mode == "factory" and frame_ppe_detections:
                    for obj_name, det_data in frame_ppe_detections.items():
                        if obj_name not in buffer_ppe_detections:
                            buffer_ppe_detections[obj_name] = {'count': 0, 'confidences': []}
                        buffer_ppe_detections[obj_name]['count'] += det_data['count']
                        buffer_ppe_detections[obj_name]['confidences'].extend(det_data['confidences'])
                
                # Log all detections if any found (helpful for debugging PPE model)
                if all_detections:
                    # Use INFO level for factory mode to see what's happening
                    log_level = logger.info if self.mode == "factory" else logger.debug
                    log_level(f"🔍 Frame {frame_idx} ({current_time:.1f}s): All detections: {', '.join(all_detections)}")
                    if frame_detections:
                        log_level(f"✅ Frame {frame_idx} ({current_time:.1f}s): Matched detections: {', '.join(frame_detections)}")
                    elif self.mode == "factory":
                        # For factory mode, log when detections are found but filtered out
                        logger.info(f"⚠️ Frame {frame_idx}: Detections found but filtered out:")
                        logger.info(f"   - Target classes: {target_classes}")
                        logger.info(f"   - Confidence threshold: {conf_threshold}")
                        logger.info(f"   - All detections: {', '.join(all_detections)}")
                elif raw_detection_count == 0 and (frame_idx < frame_skip_interval * 3 or self.mode == "factory"):
                    # Log when no detections at all (first few frames or factory mode)
                    logger.info(f"⚠️ Frame {frame_idx}: No detections found by model")
                
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
                self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, buffer_ppe_detections, scene_start_time, current_time)
                # Clear buffer
                buffer_frames = []
                buffer_person_ids = set()
                buffer_object_tags = set()
                buffer_ppe_detections = {}
                scene_start_time = current_time
                
            frame_idx += 1
            
        # Process remaining frames in buffer
        if buffer_frames:
            logger.info(f"📦 Flushing remaining {len(buffer_frames)} frames...")
            self._flush_buffer(buffer_frames, buffer_person_ids, buffer_object_tags, buffer_ppe_detections, scene_start_time, frame_idx / fps)
        
        cap.release()
        self.db.close()
        
        # Log final statistics
        logger.info(f"✅ Finished processing: {self.video_name}")
        logger.info(f"📊 Total Statistics:")
        logger.info(f"   - Total Frames Read: {frame_idx}")
        logger.info(f"   - Frames Passed to YOLO: {frames_processed_count}")
        logger.info(f"   - Expected Frames (every {frame_skip_interval}th): ~{frames_to_process}")
        logger.info(f"   - Total Detections: {total_detections}")
        logger.info(f"   - Total Faces: {total_faces}")
        logger.info(f"   - Duration: {duration:.1f}s")
        
        # Process any queued batches that failed during main processing
        if ENABLE_BATCH_QUEUE and self.failed_batch_queue:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 PROCESSING QUEUED BATCHES")
            logger.info(f"{'='*60}")
            logger.info(f"📦 {len(self.failed_batch_queue)} batch(es) in queue")
            
            for idx, batch_data in enumerate(self.failed_batch_queue, 1):
                logger.info(f"\n🔄 Retrying queued batch {idx}/{len(self.failed_batch_queue)}")
                logger.info(f"   Scene: {batch_data['start']:.1f}s - {batch_data['end']:.1f}s")
                time.sleep(QUEUE_RETRY_DELAY)  # Longer delay for queued batches
                self._flush_buffer(
                    batch_data['frames'],
                    batch_data['pids'],
                    batch_data['tags'],
                    batch_data['ppe_detections'],
                    batch_data['start'],
                    batch_data['end'],
                    is_retry_from_queue=True
                )
            
            logger.info(f"\n✅ Finished processing queued batches")
            self.failed_batch_queue.clear()

    def _compress_frames(self, frames):
        """
        Compress frames to reduce payload size and avoid API limits.
        
        Args:
            frames (list): List of PIL Image objects
            
        Returns:
            list: List of compressed PIL Image objects
        """
        compressed_frames = []
        
        for idx, frame in enumerate(frames):
            try:
                # Get original dimensions
                original_width, original_height = frame.size
                
                # Calculate new dimensions maintaining aspect ratio
                if original_width > MAX_FRAME_WIDTH or original_height > MAX_FRAME_HEIGHT:
                    # Calculate scaling factor
                    width_scale = MAX_FRAME_WIDTH / original_width
                    height_scale = MAX_FRAME_HEIGHT / original_height
                    scale = min(width_scale, height_scale)
                    
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    
                    # Resize with high-quality resampling
                    resized_frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    logger.debug(f"   Frame {idx+1}: Resized from {original_width}x{original_height} to {new_width}x{new_height}")
                else:
                    resized_frame = frame
                    logger.debug(f"   Frame {idx+1}: No resize needed ({original_width}x{original_height})")
                
                # Convert to RGB if needed (remove alpha channel)
                if resized_frame.mode in ('RGBA', 'LA', 'P'):
                    rgb_frame = Image.new('RGB', resized_frame.size, (255, 255, 255))
                    if resized_frame.mode == 'P':
                        resized_frame = resized_frame.convert('RGBA')
                    rgb_frame.paste(resized_frame, mask=resized_frame.split()[-1] if resized_frame.mode in ('RGBA', 'LA') else None)
                    resized_frame = rgb_frame
                
                # Apply JPEG compression by saving to bytes and reloading
                import io
                buffer = io.BytesIO()
                resized_frame.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                buffer.seek(0)
                compressed_frame = Image.open(buffer)
                
                compressed_frames.append(compressed_frame)
                
            except Exception as e:
                logger.error(f"❌ Error compressing frame {idx+1}: {e}")
                # Fallback: use original frame
                compressed_frames.append(frame)
        
        logger.info(f"📦 Compressed {len(frames)} frames (max: {MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}, quality: {JPEG_QUALITY}%)")
        return compressed_frames

    def _flush_buffer(self, frames, pids, tags, ppe_detections, start, end, is_retry_from_queue=False):
        """
        Send buffered frames to Gemini for analysis and store results.
        
        Args:
            frames (list): List of PIL Image objects
            pids (set): Set of detected person IDs
            tags (set): Set of detected object tags
            ppe_detections (dict): Detailed PPE detection results {class_name: {'count': int, 'confidences': [float]}}
            start (float): Scene start time
            end (float): Scene end time
            is_retry_from_queue (bool): Whether this is a retry from the failed batch queue
        """
        logger.info(f"🤖 Analyzing scene: {start:.1f}s - {end:.1f}s")
        logger.info(f"   - Frames: {len(frames)}, PIDs: {len(pids)}, Objects: {list(tags)}")
        
        # Step 1: Compress frames to reduce payload size
        logger.info(f"🗜️  Compressing frames to reduce API payload...")
        compressed_frames = self._compress_frames(frames)
        
        # Build detailed PPE detection summary for factory mode
        ppe_summary_text = ""
        if self.mode == "factory" and ppe_detections:
            logger.info(f"🦺 PPE Detection Results:")
            compliance_items = []
            violation_items = []
            
            for obj_name, det_data in sorted(ppe_detections.items()):
                count = det_data['count']
                avg_conf = sum(det_data['confidences']) / len(det_data['confidences']) if det_data['confidences'] else 0
                max_conf = max(det_data['confidences']) if det_data['confidences'] else 0
                
                # Categorize as compliance or violation
                if obj_name.startswith('NO-'):
                    violation_items.append(f"{obj_name}: {count} detection(s) (avg confidence: {avg_conf:.3f}, max: {max_conf:.3f})")
                elif obj_name in ['Hardhat', 'Mask', 'Safety Vest', 'Safety Cone']:
                    compliance_items.append(f"{obj_name}: {count} detection(s) (avg confidence: {avg_conf:.3f}, max: {max_conf:.3f})")
                else:
                    # Other objects (Person, machinery, vehicle)
                    logger.info(f"   - {obj_name}: {count} detection(s) (avg: {avg_conf:.3f}, max: {max_conf:.3f})")
            
            if compliance_items:
                logger.info(f"   ✅ PPE Compliance: {', '.join(compliance_items)}")
                ppe_summary_text += "\n\nPPE COMPLIANCE DETECTED:\n" + "\n".join(f"  - {item}" for item in compliance_items)
            
            if violation_items:
                logger.info(f"   ❌ PPE VIOLATIONS: {', '.join(violation_items)}")
                ppe_summary_text += "\n\nPPE VIOLATIONS DETECTED:\n" + "\n".join(f"  - {item}" for item in violation_items)
        
        # Enhanced prompt with strict JSON formatting and explicit detection list
        detected_objects_list = ", ".join(sorted(tags)) if tags else "None detected"
        prompt = (
            self.config["prompt"] + 
            f"\n\nDETECTED OBJECTS IN THIS SCENE: {detected_objects_list}" +
            ppe_summary_text +
            "\n\nIMPORTANT: Respond ONLY with valid JSON in this exact format:\n"
            '{"summary": "detailed description including ALL violations and compliance items", "objects": ["object1", "object2"]}\n'
            "Do not include any markdown formatting or additional text.\n"
            "CRITICAL: List EVERY violation and compliance item detected in the summary. Use the PPE detection results above."
        )
        
        # Retry logic with exponential backoff
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                # Calculate delay with exponential backoff
                if attempt > 1:
                    # Exponential backoff: base_delay * (multiplier ^ (attempt - 1))
                    delay = min(
                        GEMINI_RETRY_BASE_DELAY * (GEMINI_RETRY_MULTIPLIER ** (attempt - 2)),
                        GEMINI_RETRY_MAX_DELAY
                    )
                    logger.info(f"♻️ Retry attempt {attempt}/{GEMINI_MAX_RETRIES}...")
                    logger.info(f"⏳ Waiting {delay:.1f}s before retry (exponential backoff)...")
                    time.sleep(delay)
                else:
                    # First attempt: use standard buffer delay
                    time.sleep(GEMINI_BUFFER_DELAY)
                
                # Call Gemini API with compressed frames
                logger.debug(f"📤 Sending {len(compressed_frames)} compressed frames to Gemini...")
                response = model.generate_content(compressed_frames + [prompt])
                
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
                
                # Store in databases with PPE detection results
                self.db.add_scene_node(
                    self.video_name, 
                    self.mode, 
                    sid, 
                    start, 
                    end, 
                    data.get('summary', 'No summary'), 
                    list(pids), 
                    list(tags),
                    ppe_detections if self.mode == "factory" else None
                )
                logger.info(f"💾 Scene data stored successfully (ID: {sid[:8]}...)")
                
                # Success - break retry loop
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON Parsing Error (attempt {attempt}/{GEMINI_MAX_RETRIES}): {e}")
                logger.error(f"   Problematic text: {cleaned_text[:200] if 'cleaned_text' in locals() else 'N/A'}")
                
                if attempt == GEMINI_MAX_RETRIES:
                    # Final attempt failed - queue for retry or use fallback
                    if ENABLE_BATCH_QUEUE and not is_retry_from_queue:
                        # Add to queue for retry at the end
                        logger.warning(f"📥 Adding scene {start:.1f}s - {end:.1f}s to retry queue (JSON parse error)")
                        self.failed_batch_queue.append({
                            'frames': frames,  # Use original frames (will be compressed on retry)
                            'pids': pids,
                            'tags': tags,
                            'ppe_detections': ppe_detections,
                            'start': start,
                            'end': end
                        })
                    else:
                        # Either queue is disabled or this is already a retry from queue - use fallback
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
                            list(tags),
                            ppe_detections if self.mode == "factory" else None
                        )
                    
            except Exception as e:
                logger.error(f"❌ Gemini API Error (attempt {attempt}/{GEMINI_MAX_RETRIES}): {type(e).__name__}: {e}")
                
                if attempt == GEMINI_MAX_RETRIES:
                    # Final attempt failed - queue for retry or use fallback
                    if ENABLE_BATCH_QUEUE and not is_retry_from_queue:
                        # Add to queue for retry at the end
                        logger.warning(f"📥 Adding scene {start:.1f}s - {end:.1f}s to retry queue")
                        self.failed_batch_queue.append({
                            'frames': frames,  # Use original frames (will be compressed on retry)
                            'pids': pids,
                            'tags': tags,
                            'ppe_detections': ppe_detections,
                            'start': start,
                            'end': end
                        })
                    else:
                        # Either queue is disabled or this is already a retry from queue - use fallback
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
                            list(tags),
                            ppe_detections if self.mode == "factory" else None
                        )

