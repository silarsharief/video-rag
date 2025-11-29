import os
import sys
import certifi

# 1. Apply SSL Fix immediately (Must be before other imports)
os.environ['SSL_CERT_FILE'] = certifi.where()
print(f"🔐 SSL Cert Path set to: {certifi.where()}")

from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase
import chromadb
from ultralytics import YOLO

# Load keys
load_dotenv()

def test_gemini():
    print("\n--- 🤖 Testing Gemini API ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ FAIL: GEMINI_API_KEY not found in .env")
        return

    genai.configure(api_key=api_key)
    
    try:
        # Use a model we KNOW exists in your list
        target_model = 'gemini-2.0-flash'
        print(f"Testing generation with {target_model}...")
        
        model = genai.GenerativeModel(target_model) 
        response = model.generate_content("Hello, represent the number 5.")
        print(f"✅ SUCCESS: Gemini replied: {response.text.strip()}")
            
    except Exception as e:
        print(f"❌ FAIL: Gemini Error: {e}")
        print("\n👇 HINT: Here are the models you actually have access to:")
        for m in genai.list_models():
             if 'generateContent' in m.supported_generation_methods:
                print(f"   - {m.name}")

def test_neo4j():
    print("\n--- 🕸️ Testing Neo4j Graph DB ---")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not user:
        print("❌ FAIL: Neo4j keys missing in .env")
        return

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ SUCCESS: Connected to Neo4j Aura!")
        driver.close()
    except Exception as e:
        print(f"❌ FAIL: Neo4j Connection Error: {e}")

def test_chroma():
    print("\n--- 🧠 Testing ChromaDB (Vector Store) ---")
    try:
        client = chromadb.PersistentClient(path="./chromadb")
        col = client.get_or_create_collection("test_collection")
        col.add(documents=["test"], ids=["1"])
        print("✅ SUCCESS: ChromaDB is writing locally!")
    except Exception as e:
        print(f"❌ FAIL: ChromaDB Error: {e}")

def test_vision():
    print("\n--- 👁️ Testing Local Vision Models ---")
    try:
        # Check if file exists first to avoid re-download loop
        if os.path.exists("yolo11n.pt"):
            print("Found local yolo11n.pt file.")
            model = YOLO("yolo11n.pt")
            print("✅ SUCCESS: YOLO11n loaded from local file!")
        else:
            print("⚠️ WARNING: yolo11n.pt not found locally. Attempting download (might fail on Restricted Wifi)...")
            model = YOLO("yolo11n.pt")
            print("✅ SUCCESS: YOLO11n downloaded and loaded!")
    except Exception as e:
        print(f"❌ FAIL: YOLO Error: {e}")
        print("👉 SOLUTION: Download 'yolo11n.pt' manually and drag it into this folder.")

if __name__ == "__main__":
    print("🚀 STARTING SYSTEM CHECK...\n")
    test_gemini()
    test_neo4j()
    test_chroma()
    test_vision()
    print("\n🏁 CHECK COMPLETE.")