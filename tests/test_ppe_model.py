"""
Test script to verify PPE model is working correctly.
Tests the ppe.pt model with sample images or video frames.
"""
# Workaround for PyTorch 2.7+ weights_only default change
import torch
_original_load = torch.load

def _patched_load(*args, **kwargs):
    """Patch torch.load to use weights_only=False for model loading"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import MODELS_DIR, YOLO_PPE_MODEL


def test_model_loading(model_path=None):
    """Load PPE model - minimal logging."""
    if model_path is None:
        model_path = MODELS_DIR / YOLO_PPE_MODEL
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        model = YOLO(str(model_path))
        # Show model classes
        print(f"✅ Model loaded: {model_path.name} ({len(model.names)} classes)")
        print(f"   Classes: {', '.join(model.names.values())}")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_on_image(model, image_path=None):
    """Test the model on an image file - PPE model only, minimal logging."""
    if image_path is None:
        return
    
    # Resolve path relative to project root if it's a relative path
    image_path = Path(image_path)
    if not image_path.is_absolute():
        project_root = Path(__file__).parent.parent
        potential_path = project_root / image_path
        if potential_path.exists():
            image_path = potential_path
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        # PPE model - use conf=0 to get all detections
        results = model(frame, conf=0, verbose=False)
        detections = results[0].boxes
        
        if len(detections) > 0:
            # Group by class
            detections_by_class = {}
            for box in detections:
                cls_id = int(box.cls[0])
                obj_name = model.names[cls_id]
                if obj_name not in detections_by_class:
                    detections_by_class[obj_name] = 0
                detections_by_class[obj_name] += 1
            
            print(f"✅ Image: {len(detections)} detection(s) found")
            for obj_name, count in sorted(detections_by_class.items()):
                print(f"   - {obj_name}: {count}")
        else:
            print(f"❌ Image: No detections")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_model_on_video_frame(model, video_path=None, frame_number=0):
    """Test PPE model on a specific video frame - minimal logging."""
    if video_path is None:
        return
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Could not read frame {frame_number}")
            cap.release()
            return
        
        # PPE model - use conf=0 to get all detections
        results = model(frame, conf=0, verbose=False)
        detections = results[0].boxes
        
        if len(detections) > 0:
            detections_by_class = {}
            for box in detections:
                cls_id = int(box.cls[0])
                obj_name = model.names[cls_id]
                if obj_name not in detections_by_class:
                    detections_by_class[obj_name] = 0
                detections_by_class[obj_name] += 1
            
            print(f"✅ Frame {frame_number}: {len(detections)} detection(s)")
            for obj_name, count in sorted(detections_by_class.items()):
                print(f"   - {obj_name}: {count}")
        else:
            print(f"❌ Frame {frame_number}: No detections")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_model_on_all_video_frames(model, video_path=None, frame_skip=10, max_frames=None):
    """Test PPE model on video frames - minimal logging, only show detections."""
    if video_path is None:
        return
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video: {total_frames} frames @ {fps:.1f} fps | Checking every {frame_skip}th frame")
        
        frames_with_detections = []
        frames_checked = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only check every Nth frame
            if frame_idx % frame_skip == 0:
                frames_checked += 1
                
                # PPE model - use conf=0 to get all detections
                results = model(frame, conf=0, verbose=False)
                detections = results[0].boxes
                
                if len(detections) > 0:
                    current_time = frame_idx / fps if fps > 0 else 0
                    # Group by class
                    detections_by_class = {}
                    for box in detections:
                        cls_id = int(box.cls[0])
                        obj_name = model.names[cls_id]
                        if obj_name not in detections_by_class:
                            detections_by_class[obj_name] = 0
                        detections_by_class[obj_name] += 1
                    
                    detections_str = ", ".join([f"{name}({count})" for name, count in sorted(detections_by_class.items())])
                    print(f"✅ Frame {frame_idx} ({current_time:.1f}s): {detections_str}")
                    
                    frames_with_detections.append({
                        'frame': frame_idx,
                        'time': current_time,
                        'detections': detections_by_class
                    })
                
                if max_frames and frames_checked >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        # Summary
        print(f"\n📊 Summary: {len(frames_with_detections)}/{frames_checked} frames had detections")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")




def main():
    """Run PPE model tests - minimal logging."""
    
    # Check for command line model argument
    import sys
    custom_model = None
    test_all_models = False
    models_directory = None
    
    # Check for model argument
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pt'):
        custom_model = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    # ============================================================================
    # MANUAL CONFIGURATION - Edit these paths as needed
    # ============================================================================
    VIDEO_PATH = None#"data/videos/3_te2.mp4"  # Set to your video path, e.g., "data/videos/test.mp4" or "/full/path/to/video.mp4"
    IMAGE_PATH = "data/videos/Screenshot 2026-01-03 at 10.28.10 PM.png"  # Set to your image path, e.g., "data/test_image.jpg"
    FRAME_NUMBER = 0   # Which frame to test from video (0 = first frame)
    
    # Model path - use ppe.pt from models directory
    if custom_model:
        CUSTOM_MODEL_PATH = custom_model
    else:
        # Try models/ppe.pt relative to project root
        potential_path = project_root / "models" / "ppe.pt"
        if potential_path.exists():
            CUSTOM_MODEL_PATH = str(potential_path)
        else:
            CUSTOM_MODEL_PATH = "models/ppe.pt"  # Will be resolved in test_model_loading
    
    # Options for testing all frames
    TEST_ALL_FRAMES = False   # Set to True to scan all frames, False to test single frame
    FRAME_SKIP = 10           # Check every Nth frame (10 = every 10th frame)
    MAX_FRAMES = None        # Maximum frames to check (None = check all frames)
    # ============================================================================
    
    # Load PPE model
    model = test_model_loading(CUSTOM_MODEL_PATH)
    if model is None:
        return
    
    # Test on image (command line arg or manual config)
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_model_on_image(model, image_path)
    elif IMAGE_PATH:
        test_model_on_image(model, IMAGE_PATH)
    
    # Test on video frame(s)
    if len(sys.argv) > 2:
        video_path = sys.argv[2]
        if len(sys.argv) > 3 and sys.argv[3].lower() == "all":
            frame_skip = int(sys.argv[4]) if len(sys.argv) > 4 else 10
            max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else None
            test_model_on_all_video_frames(model, video_path, frame_skip, max_frames)
        else:
            frame_number = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            test_model_on_video_frame(model, video_path, frame_number)
    elif VIDEO_PATH:
        if TEST_ALL_FRAMES:
            test_model_on_all_video_frames(model, VIDEO_PATH, FRAME_SKIP, MAX_FRAMES)
        else:
            test_model_on_video_frame(model, VIDEO_PATH, FRAME_NUMBER)


if __name__ == "__main__":
    main()

