"""
System Connections Test Suite
Tests Gemini API, Neo4j, ChromaDB, and YOLO model connectivity.
"""
import os
import sys

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import google.generativeai as genai
from neo4j import GraphDatabase
import chromadb
from ultralytics import YOLO

# Import configurations
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    CHROMADB_PATH,
    MODELS_DIR,
    YOLO_DEFAULT_MODEL
)


def test_gemini():
    """Test Gemini API connection and generation."""
    print("\n--- 🤖 Testing Gemini API ---")
    
    if not GEMINI_API_KEY:
        print("❌ FAIL: GEMINI_API_KEY not found in .env")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        print(f"Testing generation with {GEMINI_MODEL}...")
        model = genai.GenerativeModel(GEMINI_MODEL) 
        response = model.generate_content("Hello, represent the number 5.")
        print(f"✅ SUCCESS: Gemini replied: {response.text.strip()}")
            
    except Exception as e:
        print(f"❌ FAIL: Gemini Error: {e}")
        print("\n👇 HINT: Here are the models you actually have access to:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   - {m.name}")


def test_neo4j():
    """Test Neo4j graph database connection."""
    print("\n--- 🕸️ Testing Neo4j Graph DB ---")
    
    if not NEO4J_URI or not NEO4J_USERNAME:
        print("❌ FAIL: Neo4j keys missing in .env")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ SUCCESS: Connected to Neo4j Aura!")
        driver.close()
    except Exception as e:
        print(f"❌ FAIL: Neo4j Connection Error: {e}")


def test_chroma():
    """Test ChromaDB vector store."""
    print("\n--- 🧠 Testing ChromaDB (Vector Store) ---")
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        col = client.get_or_create_collection("test_collection")
        col.add(documents=["test"], ids=["1"])
        print("✅ SUCCESS: ChromaDB is writing locally!")
    except Exception as e:
        print(f"❌ FAIL: ChromaDB Error: {e}")


def test_vision():
    """Test YOLO vision model loading."""
    print("\n--- 👁️ Testing Local Vision Models ---")
    
    try:
        model_path = MODELS_DIR / YOLO_DEFAULT_MODEL
        
        if model_path.exists():
            print(f"Found local {YOLO_DEFAULT_MODEL} file.")
            model = YOLO(str(model_path))
            print(f"✅ SUCCESS: {YOLO_DEFAULT_MODEL} loaded from local file!")
        else:
            print(f"⚠️ WARNING: {YOLO_DEFAULT_MODEL} not found locally. Attempting download (might fail on Restricted Wifi)...")
            model = YOLO(YOLO_DEFAULT_MODEL)
            print(f"✅ SUCCESS: {YOLO_DEFAULT_MODEL} downloaded and loaded!")
    except Exception as e:
        print(f"❌ FAIL: YOLO Error: {e}")
        print(f"👉 SOLUTION: Download '{YOLO_DEFAULT_MODEL}' manually and place it in {MODELS_DIR}/")


if __name__ == "__main__":
    print("🚀 STARTING SYSTEM CHECK...\n")
    test_gemini()
    test_neo4j()
    test_chroma()
    test_vision()
    print("\n🏁 CHECK COMPLETE.")

