"""
Video Processing Modes Configuration
Each mode defines specific detection classes, prompts, and models for different use cases.
"""

from config.settings import YOLO_HEAVY_MODEL, YOLO_PPE_MODEL

# ============================================================================
# VIDEO PROCESSING MODES
# ============================================================================
# Each mode defines:
# - model: YOLO model file to use
# - prompt: Instruction for Gemini analysis
# - classes: List of COCO class IDs to detect (see COCO_CLASSES below for mapping)

# Helper: Standard COCO Class IDs for reference
# 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light, 
# 10:fire hydrant, 11:stop sign, 24:backpack, 26:handbag, 28:suitcase, 
# 39:bottle, 40:wine glass, 41:cup, 42:fork, 43:knife, 44:spoon, 45:bowl, 
# 46-55: foods (banana, apple, sandwich, orange, broccoli...), 
# 66:mouse (pest), 67:remote, 68:microwave, 69:oven, 70:toaster, 71:sink, 72:refrigerator

MODES = {
    "traffic": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for better vehicle detection
        "prompt": """
            You are a Traffic Analyst.
            1. Identify vehicle types, COLORS, and pedestrians.
            2. Note any traffic violations (running lights, jaywalking).
            3. Describe the flow and density of the traffic.
            Output pure JSON with summary.
        """,
        # EXPANDED: Vehicles + Pedestrians + Signs + Lights + Bags (for finding specific people)
        "classes": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 24, 26, 28],
        "frame_skip_interval": 10  # Default: process every 10th frame
    },
    
    "factory": {
        "model": YOLO_PPE_MODEL,  # ppe.pt - Safety Detection Model
        "prompt": """
            You are a Safety Inspector analyzing a workplace scene.
            The system has detected PPE items and violations in the video frames.
            
            CRITICAL: Report ALL detections - both compliance and violations.
            
            1. PPE COMPLIANCE (Good - report all detected):
               - Hardhat: Workers wearing hardhats/helmets
               - Mask: Workers wearing masks
               - Safety Vest: Workers wearing safety vests
               - Safety Cone: Safety cones present (environmental safety)
            
            2. PPE VIOLATIONS (Bad - report ALL detected):
               - NO-Hardhat: Workers NOT wearing hardhats/helmets (SAFETY VIOLATION)
               - NO-Mask: Workers NOT wearing masks (SAFETY VIOLATION)
               - NO-Safety Vest: Workers NOT wearing safety vests (SAFETY VIOLATION)
            
            3. CONTEXT AND HAZARDS:
               - Person: Number of people in scene
               - machinery: Machinery present (potential hazard)
               - vehicle: Vehicles present (potential hazard)
               - Describe worker activities and environmental conditions
            
            4. SAFETY ASSESSMENT:
               - List ALL violations found (be specific: "Worker without hardhat", "Worker without safety vest", etc.)
               - List ALL compliance items found (be specific: "Worker with hardhat", "Worker with safety vest", etc.)
               - Assess overall safety status and risk level
               - Note any environmental hazards or unsafe conditions
            
            IMPORTANT: Do not skip any violations. If multiple workers are present, note violations for each.
            Output pure JSON with summary.
        """,
        # PPE model classes (ppe.pt has 10 classes: 0-9)
        # 0:Hardhat, 1:Mask, 2:NO-Hardhat, 3:NO-Mask, 4:NO-Safety Vest,
        # 5:Person, 6:Safety Cone, 7:Safety Vest, 8:machinery, 9:vehicle
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All 10 classes from ppe.pt
        "frame_skip_interval": 10,  # Process every 10th frame (optimized for performance)
        "confidence_threshold": 0.15  # Lower threshold for PPE model (detects low-confidence violations)
    },
    
    "kitchen": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for better small object detection (knives/pests)
        "prompt": """
            You are a Health Inspector. Analyze this kitchen.
            1. HYGIENE: Are staff detected? Are they wearing gloves/hairnets?
            2. SAFETY: Look for 'knife' objects left unsafe. Check for fire/smoke.
            3. PESTS: Look for mice/rats.
            4. CLEANLINESS: Check for spills or clutter on tables.
            Output pure JSON with summary.
        """,
        # EXPANDED: People + Bottles + Cups + Cutlery + Food + Pests + Appliances
        "classes": [0, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 66, 68, 69, 70, 71, 72],
        "frame_skip_interval": 10  # Default: process every 10th frame
    },
    
    "general": {
        "model": YOLO_HEAVY_MODEL,  # yolo11x.pt for comprehensive detection
        "prompt": """
            You are a General Video Observer. 
            1. Describe the scene, location, and mood.
            2. List the main objects and actions occurring.
            3. Note any anomalies or interesting events.
            Output pure JSON with summary.
        """,
        # ENABLE EVERYTHING: 0 to 79 (All COCO classes)
        "classes": list(range(80)),
        "frame_skip_interval": 10  # Default: process every 10th frame
    }
}

# ============================================================================
# COCO DATASET CLASS MAPPINGS (for reference)
# ============================================================================
# Complete list of YOLO COCO classes for documentation
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
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

