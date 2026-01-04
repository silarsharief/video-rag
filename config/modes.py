"""
Video Processing Modes Configuration
Each mode defines specific detection classes, prompts, and models for different use cases.
"""

from config.settings import YOLO_HEAVY_MODEL, YOLO_PPE_MODEL
from config.prompts import get_prompt
from config.detection_classes import get_classes

# ============================================================================
# VIDEO PROCESSING MODES
# ============================================================================
# Each mode defines:
# - model: YOLO model file to use
# - prompt: Instruction for Gemini analysis (see config/prompts.py)
# - classes: List of COCO class IDs to detect (see config/detection_classes.py for mappings)

# Helper: Standard COCO Class IDs for reference
# 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light, 
# 10:fire hydrant, 11:stop sign, 24:backpack, 26:handbag, 28:suitcase, 
# 39:bottle, 40:wine glass, 41:cup, 42:fork, 43:knife, 44:spoon, 45:bowl, 
# 46-55: foods (banana, apple, sandwich, orange, broccoli...), 
# 66:mouse (pest), 67:remote, 68:microwave, 69:oven, 70:toaster, 71:sink, 72:refrigerator

MODES = {
    "traffic": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for better vehicle detection
        "prompt": get_prompt("traffic"),
        "classes": get_classes("traffic"),
        "frame_skip_interval": 10  # Default: process every 10th frame
    },
    
    "factory": {
        "model": YOLO_PPE_MODEL,  # ppe.pt - Safety Detection Model
        "prompt": get_prompt("factory"),
        "classes": get_classes("factory"),
        "frame_skip_interval": 10,  # Process every 10th frame (optimized for performance)
        "confidence_threshold": 0.15  # Lower threshold for PPE model (detects low-confidence violations)
    },
    
    "kitchen": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for better small object detection (knives/pests)
        "prompt": get_prompt("kitchen"),
        "classes": get_classes("kitchen"),
        "frame_skip_interval": 10  # Default: process every 10th frame
    },
    
    "general": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for comprehensive detection
        "prompt": get_prompt("general"),
        "classes": get_classes("general"),
        "frame_skip_interval": 10  # Default: process every 10th frame
    }
}

def get_mode_config(mode_name):
    """
    Get configuration for a specific mode.
    
    Args:
        mode_name (str): Name of the mode (traffic, factory, kitchen, general)
        
    Returns:
        dict: Mode configuration or general mode if not found
    """
    return MODES.get(mode_name, MODES["general"])


def list_available_modes():
    """
    Get list of all available mode names.
    
    Returns:
        list: List of mode names
    """
    return list(MODES.keys())

