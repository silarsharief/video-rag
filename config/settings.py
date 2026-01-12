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
YOLO_PPE_MODEL = "ppe.pt"  # Updated to use ppe.pt model

# ============================================================================
# VIDEO PROCESSING PARAMETERS
# ============================================================================
# Frame processing
FRAME_SKIP_INTERVAL = 10  # Process every Nth frame
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for YOLO detections
SCENE_CAPTURE_INTERVAL = 0.5  # Minimum seconds between frame captures
BATCH_SIZE = 5  # Number of frames to batch before sending to Gemini

# Frame quality settings (to reduce payload size and avoid API limits)
MAX_FRAME_WIDTH = 512  # Maximum width for frames sent to Gemini (lower = smaller payload)
MAX_FRAME_HEIGHT = 512  # Maximum height for frames sent to Gemini
JPEG_QUALITY = 60  # JPEG compression quality (1-100, lower = smaller file, 60 is good balance)

# Face recognition
FACE_DETECTION_SIZE = (640, 640)
FACE_CTX_ID = 0  # CPU context ID for insightface

# ============================================================================
# API RATE LIMITING
# ============================================================================
# Gemini API rate limits (based on your account tier)
# gemini-2.0-flash: 2000 RPM (requests per minute)
GEMINI_RPM_LIMIT = 60  # Your actual limit from API dashboard
GEMINI_BUFFER_DELAY = 3.5  # Base delay between Gemini calls during ingestion (increased to avoid 429 errors)
GEMINI_SEARCH_DELAY = 0.5  # Seconds for search operations (query rewrite + filtering + summary)

# Adaptive rate limiting (automatically adjusts delays based on errors)
ENABLE_ADAPTIVE_RATE_LIMITING = True  # Enable dynamic delay adjustment
ADAPTIVE_SLOWDOWN_FACTOR = 1.5  # Multiply delay by this when hitting errors
ADAPTIVE_SPEEDUP_FACTOR = 0.9  # Multiply delay by this after successful requests
ADAPTIVE_MIN_DELAY = 1.0  # Minimum delay (never go below this)
ADAPTIVE_MAX_DELAY = 10.0  # Maximum delay (never go above this)
ADAPTIVE_SUCCESS_THRESHOLD = 5  # Number of successful requests before speeding up

# Retry configuration for API failures
GEMINI_MAX_RETRIES = 5  # Maximum retry attempts when hitting rate limits
GEMINI_RETRY_BASE_DELAY = 3.0  # Base delay for first retry (seconds)
GEMINI_RETRY_MULTIPLIER = 2.0  # Exponential backoff multiplier (2.0 = double each time)
GEMINI_RETRY_MAX_DELAY = 30.0  # Maximum delay between retries (seconds)

# Queue configuration for failed batches
ENABLE_BATCH_QUEUE = True  # Enable queueing of failed batches for retry at end
QUEUE_RETRY_DELAY = 10.0  # Delay between queued batch retries (seconds)

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================
# Cache query rewrites to reduce API calls and improve performance
ENABLE_QUERY_CACHE = True  # Enable caching of query rewrites
QUERY_CACHE_MAX_SIZE = 100  # Maximum number of cached queries (LRU eviction)
QUERY_CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# ============================================================================
# MONITORING & METRICS
# ============================================================================
# Track API performance and system metrics
ENABLE_METRICS = True  # Enable metrics tracking
METRICS_LOG_INTERVAL = 10  # Log metrics summary every N requests
ENABLE_ERROR_SUMMARY = True  # Show error summary after video processing

# ============================================================================
# DATABASE CONFIGURATIONS
# ============================================================================
# ChromaDB collection settings
CHROMA_COLLECTION_NAME = "forensic_scenes"
CHROMA_DISTANCE_METRIC = "cosine"  # Options: "cosine", "l2", "ip"

# Search parameters
SEARCH_TOP_K = 5  # Top K results to return to user (after filtering)
SEARCH_FETCH_LIMIT = 15  # Fetch more from vector DB, then filter by threshold
SIMILARITY_THRESHOLD = 0.5  # Maximum distance to include (0.0 = perfect match, 1.0 = no match)
MIN_RESULTS = 1  # Always return at least this many results (even if threshold not met)

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

# ============================================================================
# ENVIRONMENT-BASED CONFIGURATION
# ============================================================================
# Override settings based on environment (dev/staging/prod)
ENVIRONMENT = os.getenv("FORENSIC_ENV", "production").lower()  # dev, staging, production

# Development environment overrides (faster, more verbose)
if ENVIRONMENT == "dev":
    LOG_LEVEL = "DEBUG"
    GEMINI_BUFFER_DELAY = 1.0  # Faster for development
    ENABLE_METRICS = True
    METRICS_LOG_INTERVAL = 5  # More frequent logging

# Staging environment overrides (balanced)
elif ENVIRONMENT == "staging":
    LOG_LEVEL = "INFO"
    GEMINI_BUFFER_DELAY = 2.5
    ENABLE_METRICS = True

# Production environment (stable, optimized)
elif ENVIRONMENT == "production":
    # Use settings as defined above
    pass

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration settings on startup."""
    errors = []
    warnings = []
    
    # Check API keys
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set in environment variables")
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        errors.append("Neo4j credentials are not fully configured")
    
    # Check rate limiting makes sense
    if GEMINI_BUFFER_DELAY < 0.1:
        warnings.append(f"GEMINI_BUFFER_DELAY ({GEMINI_BUFFER_DELAY}s) is very low, may hit rate limits")
    if BATCH_SIZE > 10:
        warnings.append(f"BATCH_SIZE ({BATCH_SIZE}) is high, may create large payloads")
    
    # Check retry configuration
    if GEMINI_MAX_RETRIES < 3:
        warnings.append(f"GEMINI_MAX_RETRIES ({GEMINI_MAX_RETRIES}) is low, may fail on temporary errors")
    
    # Check frame quality settings
    if JPEG_QUALITY < 30:
        warnings.append(f"JPEG_QUALITY ({JPEG_QUALITY}%) is very low, may affect analysis accuracy")
    if MAX_FRAME_WIDTH < 256 or MAX_FRAME_HEIGHT < 256:
        warnings.append(f"Frame dimensions ({MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}) are very small")
    
    # Print validation results
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError("Configuration validation failed. Please fix the errors above.")
    
    if warnings:
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"🔧 CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Gemini Model: {GEMINI_MODEL}")
    print(f"Rate Limiting: {GEMINI_BUFFER_DELAY}s delay, {GEMINI_MAX_RETRIES} retries")
    print(f"Frame Quality: {MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT} @ {JPEG_QUALITY}% JPEG")
    print(f"Batch Size: {BATCH_SIZE} frames")
    print(f"Adaptive Rate Limiting: {'Enabled' if ENABLE_ADAPTIVE_RATE_LIMITING else 'Disabled'}")
    print(f"Query Cache: {'Enabled' if ENABLE_QUERY_CACHE else 'Disabled'}")
    print(f"Metrics: {'Enabled' if ENABLE_METRICS else 'Disabled'}")
    print(f"{'='*60}\n")

# Run validation on import (can be disabled by setting env var)
if os.getenv("SKIP_CONFIG_VALIDATION", "false").lower() != "true":
    validate_config()

