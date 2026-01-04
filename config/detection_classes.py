"""
Detection Classes Configuration
Centralized class ID definitions for all video processing modes.
"""

# ============================================================================
# COCO DATASET CLASS MAPPINGS (for reference)
# ============================================================================
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

# ============================================================================
# MODE-SPECIFIC CLASS CONFIGURATIONS
# ============================================================================

# Traffic Mode Classes
# Vehicles + Pedestrians + Signs + Lights + Bags (for finding specific people)
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 24, 26, 28]

# Factory Mode Classes (PPE model - ppe.pt has 10 classes: 0-9)
# 0:Hardhat, 1:Mask, 2:NO-Hardhat, 3:NO-Mask, 4:NO-Safety Vest,
# 5:Person, 6:Safety Cone, 7:Safety Vest, 8:machinery, 9:vehicle
FACTORY_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Kitchen Mode Classes
# People + Bottles + Cups + Cutlery + Food + Pests + Appliances
KITCHEN_CLASSES = [0, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 66, 68, 69, 70, 71, 72]

# General Mode Classes
# All COCO classes (0 to 79)
GENERAL_CLASSES = list(range(80))

# ============================================================================
# CLASS MAPPING BY MODE
# ============================================================================
MODE_CLASSES = {
    "traffic": TRAFFIC_CLASSES,
    "factory": FACTORY_CLASSES,
    "kitchen": KITCHEN_CLASSES,
    "general": GENERAL_CLASSES
}


def get_classes(mode_name):
    """
    Get detection classes for a specific mode.
    
    Args:
        mode_name (str): Name of the mode (traffic, factory, kitchen, general)
        
    Returns:
        list: List of class IDs for the mode, or general classes if mode not found
    """
    return MODE_CLASSES.get(mode_name, GENERAL_CLASSES)


def get_class_name(class_id):
    """
    Get class name from COCO class ID.
    
    Args:
        class_id (int): COCO class ID (0-79)
        
    Returns:
        str: Class name or "unknown" if ID not found
    """
    return COCO_CLASSES.get(class_id, "unknown")

