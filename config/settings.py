"""
Centralized Configuration for Forensic RAG System
All paths, constants, and settings are defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
# Get the project root directory (parent of config/)
_config_dir = Path(__file__).parent
_project_root = _config_dir.parent
_env_file = _project_root / ".env"

# Load .env file explicitly from project root
load_dotenv(dotenv_path=_env_file, override=True)

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Base directories (use the same project root we calculated above)
PROJECT_ROOT = _project_root.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data subdirectories
VIDEO_STORAGE_DIR = DATA_DIR / "videos"
CHROMADB_PATH = DATA_DIR / "chromadb"
TEMP_DIR = DATA_DIR / "temp"

# Ensure critical directories exist
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CREDENTIALS (from .env)
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# Gemini model to use
GEMINI_MODEL = "gemini-2.0-flash"

# YOLO model files (relative to MODELS_DIR)
YOLO_DEFAULT_MODEL = "yolo11n.pt"
YOLO_HEAVY_MODEL = "yolo11x.pt"
YOLO_PPE_MODEL = "yolo_ppe.pt"

# ============================================================================
# VIDEO PROCESSING PARAMETERS
# ============================================================================
# Frame processing
FRAME_SKIP_INTERVAL = 10  # Process every Nth frame
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for YOLO detections
SCENE_CAPTURE_INTERVAL = 0.5  # Minimum seconds between frame captures
BATCH_SIZE = 10  # Number of frames to batch before sending to Gemini

# Face recognition
FACE_DETECTION_SIZE = (640, 640)
FACE_CTX_ID = 0  # CPU context ID for insightface

# ============================================================================
# API RATE LIMITING
# ============================================================================
# Gemini API rate limits (adjust based on your tier)
GEMINI_RPM_LIMIT = 60  # Requests per minute
GEMINI_BUFFER_DELAY = 0.2  # Seconds to wait between Gemini calls during ingestion
GEMINI_SEARCH_DELAY = 0.5  # Seconds to wait for search operations

# ============================================================================
# DATABASE CONFIGURATIONS
# ============================================================================
# ChromaDB collection settings
CHROMA_COLLECTION_NAME = "forensic_scenes"
CHROMA_DISTANCE_METRIC = "cosine"  # Options: "cosine", "l2", "ip"

# Search parameters
DEFAULT_SEARCH_RESULTS = 10  # Number of results to retrieve from vector DB

# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================
APP_TITLE = "Forensic RAG"
APP_ICON = "🕵️"
APP_LAYOUT = "wide"
DISPLAY_COLUMNS = 2  # Number of columns for displaying video evidence

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# SSL CONFIGURATION (Mac fix)
# ============================================================================
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

