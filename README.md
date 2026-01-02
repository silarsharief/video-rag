# Forensic Video RAG System

A production-ready, edge-first forensic video analysis system that processes CCTV footage and enables natural language queries for security investigations. The system combines computer vision, face recognition, vector search, and graph databases to provide intelligent video analysis capabilities.
<img width="1667" height="1042" alt="Screenshot 2026-01-02 at 5 55 59 PM" src="https://github.com/user-attachments/assets/d3dc9592-7b70-4a32-b1e9-bffd97b10759" />

## 🎯 Overview

This system processes video footage to extract meaningful insights, detect objects and people, recognize faces, and allows investigators to search through hours of footage using natural language queries. It's designed for security teams, law enforcement, and forensic analysts who need to quickly find evidence in video recordings.

### Key Capabilities

- **Multi-Modal Video Analysis**: Processes videos using YOLO object detection, InsightFace face recognition, and Gemini Vision Language Model
- **Intelligent Frame Capture**: Automatically captures relevant frames based on events and time intervals
- **Semantic Search**: Query videos using natural language (e.g., "Show me all cars" or "Find the person in the red jacket")
- **Graph-Based Timeline**: Maintains temporal relationships and person appearances across scenes using Neo4j
- **Multi-Use Case Support**: Pre-configured modes for traffic, factory safety, kitchen inspection, and general surveillance
- **Face Re-identification**: Track the same person across multiple videos and camera feeds
- **Evidence Compilation**: Generate evidence reports with timestamps and visual references

## ✨ Features

### Video Processing

- Automatic frame extraction based on object detection events
- Batch processing of multiple video files
- Support for MP4 and MOV formats
- Intelligent scene segmentation (1-second minimum intervals, 60-second maximum scenes)

### Object Detection

- YOLO11 models for general object detection (80 COCO classes)
- Custom PPE (Personal Protective Equipment) model for factory safety inspections
- Configurable class filtering per use case mode
- Real-time detection with confidence thresholds

### Face Recognition

- InsightFace-based face embedding extraction
- Person ID generation for tracking individuals
- Cross-video person re-identification
- Privacy-preserving face hashing

### Search & Retrieval

- Vector-based semantic search using ChromaDB
- Graph-based relationship queries using Neo4j
- Mode-filtered search (traffic, factory, kitchen, general)
- Smart candidate filtering using LLM
- Evidence synthesis with timestamps and descriptions

### User Interface

- Streamlit-based web interface
- Multi-file upload support
- Real-time processing progress
- Video playback with timestamp navigation
- Evidence visualization with metadata

## 🏗️ Architecture

The system follows a hybrid architecture combining vector search, graph databases, and vision AI:

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Ingest │──► YOLO Detection
    │  Module │──► InsightFace
    └────┬────┘──► Gemini VLM
         │
    ┌────▼──────────────────┐
    │   Database Layer      │
    │  ┌─────────┐ ┌──────┐ │
    │  │ ChromaDB│ │Neo4j │ │
    │  │(Vectors)│ │(Graph)│ │
    │  └─────────┘ └──────┘ │
    └────────┬──────────────┘
         │
    ┌────▼────┐
    │Retrieval│──► Vector Search
    │ Module  │──► Graph Traversal
    └─────────┘──► LLM Synthesis
```

### Data Flow

1. **Ingestion**: Video → Frame Extraction → YOLO Detection → Face Recognition → Gemini Analysis → Storage
2. **Storage**: Scene summaries stored in ChromaDB (vectors) and Neo4j (graph relationships)
3. **Retrieval**: User Query → Vector Search → Graph Enrichment → LLM Filtering → Evidence Synthesis

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.10+**: Main programming language
- **Streamlit**: Web application framework
- **OpenCV**: Video processing and frame extraction
- **Ultralytics YOLO**: Object detection models
- **InsightFace**: Face recognition and embedding extraction
- **Google Gemini 1.5 Flash**: Vision Language Model for scene analysis

### Databases

- **ChromaDB**: Vector database for semantic search
- **Neo4j**: Graph database for timeline and relationship tracking

### Additional Libraries

- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **python-dotenv**: Environment variable management
- **certifi**: SSL certificate handling

## 📋 Prerequisites

- Python 3.10 or higher
- Neo4j database instance (local or Neo4j Aura cloud)
- Google Gemini API key (free tier supported)
- macOS (for MPS device support) or compatible system
- Minimum 8GB RAM recommended
- GPU support optional but recommended for faster processing

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd videorag
```

### 2. Create Virtual Environment

```bash
cd forensic-rag
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Models

The system requires YOLO model files. Ensure the following models are present in the `forensic-rag` directory:

- `yolo11n.pt` or `yolo11x.pt` (for general/traffic/kitchen modes)
- `yolo_ppe.pt` (for factory safety mode)

Models will be auto-downloaded on first use if not present, or you can download them manually.

### 5. Configure Environment Variables

Create a `.env` file in the `forensic-rag` directory:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Database
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### 6. Test Connections

Run the connection test script to verify all services are properly configured:

```bash
python test_connections.py
```

This will test:

- Gemini API connectivity
- Neo4j database connection
- ChromaDB local storage
- YOLO model loading

## ⚙️ Configuration

### Use Case Modes

The system supports four pre-configured modes defined in `ingest.py`:

1. **Traffic Mode** (`traffic`)

   - Detects vehicles, pedestrians, traffic signs, and violations
   - Uses YOLO11x model
   - Classes: vehicles, pedestrians, traffic lights, signs, bags

2. **Factory Mode** (`factory`)

   - PPE (Personal Protective Equipment) compliance checking
   - Uses custom `yolo_ppe.pt` model
   - Detects: helmets, masks, gloves, goggles, and violations

3. **Kitchen Mode** (`kitchen`)

   - Health and safety inspection
   - Detects: hygiene violations, pests, unsafe practices
   - Classes: people, cutlery, food items, appliances, pests

4. **General Mode** (`general`)
   - Catch-all for any surveillance scenario
   - Uses all 80 COCO classes
   - Flexible scene analysis

### Model Configuration

Edit `MODES` dictionary in `ingest.py` to customize:

- Model selection per mode
- Detection classes
- Analysis prompts
- Confidence thresholds

## 📖 Usage

### Starting the Application

```bash
cd forensic-rag
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Processing Videos

1. **Select Use Case Mode**: Choose from traffic, factory, kitchen, or general
2. **Upload Videos**: Click "Upload CCTV Footage" and select one or more MP4/MOV files
3. **Process**: Click "Process All Videos" and monitor progress
4. Videos are automatically:
   - Saved to `stored_videos/` directory
   - Analyzed frame-by-frame
   - Indexed in ChromaDB and Neo4j

### Searching Videos

1. Enter a natural language query in the search box (e.g., "Show me all cars")
2. The system will:
   - Search vector embeddings for semantic matches
   - Enrich results with graph relationships
   - Filter candidates using LLM
   - Synthesize a comprehensive answer
3. View results:
   - Analysis summary
   - Matching video clips with timestamps
   - Detected objects and person IDs
   - Playable video segments

### Example Queries

- "Show me all cars in the parking lot"
- "Find the person who entered after 9 PM"
- "Track Person_02 across all videos"
- "Show me safety violations in the factory"
- "What happened before the incident at 3:45 PM?"

## 📁 Project Structure

```
videorag/
├── README.md                 # This file
├── USE_CASES.md              # Detailed use case scenarios
├── requirements.txt          # Root dependencies
└── forensic-rag/
    ├── app.py                # Streamlit application
    ├── ingest.py             # Video ingestion and processing
    ├── retrieval.py          # Search and retrieval logic
    ├── database.py           # Database abstraction layer
    ├── finding_classes.py    # YOLO class utilities
    ├── test_connections.py   # Connection testing script
    ├── requirements.txt      # Python dependencies
    ├── .env                  # Environment variables (create this)
    ├── chromadb/             # ChromaDB storage directory
    ├── stored_videos/        # Uploaded video storage
    ├── yolo11n.pt           # YOLO nano model
    ├── yolo11x.pt           # YOLO extra-large model
    └── yolo_ppe.pt          # Custom PPE detection model
```

## 🎯 Use Cases

The system is designed for various forensic and security scenarios:

- **Package Theft Investigation**: Find suspects and track movements
- **Unauthorized Access Detection**: Identify after-hours entries
- **Person Re-identification**: Track individuals across multiple cameras
- **Timeline Reconstruction**: Understand event sequences
- **Loitering Detection**: Identify suspicious behavior patterns
- **Vehicle Tracking**: Monitor vehicle movements and access
- **Evidence Compilation**: Generate comprehensive evidence reports
- **Behavior Pattern Analysis**: Detect anomalies and violations
- **Quick Person Search**: Find specific individuals in hours of footage
- **Multi-Camera Investigation**: Unified view across camera feeds

For detailed use case scenarios, see [USE_CASES.md](USE_CASES.md).

## 🧪 Testing

### Test Database Connections

```bash
python test_connections.py
```

This script verifies:

- ✅ Gemini API access and model availability
- ✅ Neo4j connection and authentication
- ✅ ChromaDB local storage functionality
- ✅ YOLO model loading

### Test Video Ingestion

```python
from ingest import VideoIngestor

ingestor = VideoIngestor("path/to/video.mp4", mode="traffic")
ingestor.process_video()
```

## 🔧 Troubleshooting

### Common Issues

**1. SSL Certificate Errors (macOS)**

- The system automatically sets SSL certificate path using `certifi`
- If issues persist, ensure `certifi` is installed: `pip install certifi`

**2. Gemini API Rate Limits**

- The system includes automatic retry with exponential backoff
- For production use, consider upgrading from free tier
- Check API quota in Google Cloud Console

**3. Neo4j Connection Failures**

- Verify credentials in `.env` file
- Check network connectivity to Neo4j instance
- Ensure Neo4j Aura instance is running (if using cloud)

**4. YOLO Model Not Found**

- Models auto-download on first use
- For offline use, manually download models and place in `forensic-rag/` directory
- Check model file names match exactly: `yolo11n.pt`, `yolo11x.pt`, `yolo_ppe.pt`

**5. ChromaDB Permission Errors**

- Ensure write permissions in `forensic-rag/` directory
- Check disk space availability

**6. Face Recognition Not Working**

- InsightFace requires proper initialization
- Ensure ONNX runtime is correctly installed
- Check that InsightFace models are downloaded (auto-downloaded on first use)

### Performance Optimization

- **GPU Acceleration**: Use MPS (Metal Performance Shaders) on macOS or CUDA on Linux/Windows
- **Batch Processing**: Process multiple videos sequentially to avoid memory issues
- **Frame Sampling**: Adjust frame sampling rate in `ingest.py` (currently every 5th frame)
- **Scene Duration**: Modify scene buffer size (currently 5 frames or 60 seconds)

## 🔒 Privacy & Security

- **Local Processing**: Video files are stored locally in `stored_videos/` directory
- **Cloud API**: Only frame images (not full videos) are sent to Gemini API for analysis
- **Face Hashing**: Person IDs are generated using hash functions, not raw face data
- **Data Storage**: All metadata stored locally (ChromaDB) or in your Neo4j instance
- **No Video Upload**: Original videos never leave your system

## 📝 Development Notes

### Code Standards

- Follows Python PEP 8 style guidelines
- Industry-standard error handling and logging
- Configuration-driven design (no hardcoded values)
- Modular architecture for maintainability

### Adding New Use Cases

1. Add mode configuration to `MODES` dictionary in `ingest.py`
2. Define appropriate YOLO classes
3. Create custom analysis prompt
4. Test with sample videos

### Extending Functionality

- **Custom Models**: Replace YOLO models with domain-specific detectors
- **Additional Databases**: Extend `database.py` for other storage backends
- **API Integration**: Replace Streamlit with FastAPI for programmatic access
- **Real-time Processing**: Modify ingestion for live video streams

## 🤝 Contributing

This is a production-level codebase. When contributing:

1. Ensure all code passes linting checks
2. Follow existing code patterns and structure
3. Add appropriate comments and documentation
4. Test changes with `test_connections.py`
5. Verify no breaking changes to existing functionality

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Ultralytics for YOLO models
- InsightFace for face recognition
- Google for Gemini Vision Language Model
- ChromaDB and Neo4j communities

## 📞 Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Note**: This system is designed for legitimate security and forensic purposes. Ensure compliance with local privacy laws and regulations when processing video footage.
