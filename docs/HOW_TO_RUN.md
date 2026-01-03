# 🚀 How to Run the Application

## ✅ **FIXED: Import Error Resolved**

The `ModuleNotFoundError: No module named 'src'` error has been fixed!

---

## 📍 **Important: Always Run from Project Root**

```bash
cd /Users/silarsharief/Desktop/videorag
```

Make sure you're in the `videorag/` directory, **NOT** the `forensic-rag/` subdirectory!

---

## 🎯 **3 Ways to Run the App**

### **Method 1: Python Script** (Recommended)
```bash
python run_app.py
```

### **Method 2: Bash Script** (Mac/Linux)
```bash
./start.sh
```

### **Method 3: Direct Streamlit Command**
```bash
export PYTHONPATH=$(pwd)
streamlit run src/app.py
```

---

## ✅ **Complete Workflow**

```bash
# 1. Navigate to project root
cd /Users/silarsharief/Desktop/videorag

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the app (choose your favorite method)
python run_app.py
```

---

## 🔧 **What Was Fixed**

**Problem:** The app couldn't find the `src` module when running.

**Solution:** Added these lines to `src/app.py`:
```python
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

And updated `run_app.py` to set `PYTHONPATH` environment variable.

---

## 📋 **Verify It Works**

After running any of the commands above, you should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Then your browser will open to the app! 🎉

---

## ⚠️ **Common Mistakes to Avoid**

### ❌ **Wrong Directory**
```bash
# DON'T run from inside forensic-rag/
cd forensic-rag  # ❌ WRONG
python run_app.py  # Won't find the new structure
```

### ✅ **Correct Directory**
```bash
# DO run from videorag/ root
cd /Users/silarsharief/Desktop/videorag  # ✅ CORRECT
python run_app.py
```

### ❌ **Wrong Command**
```bash
# DON'T use python directly on app.py
python src/app.py  # ❌ Won't start Streamlit server
```

### ✅ **Correct Commands**
```bash
# DO use one of these
python run_app.py           # ✅ CORRECT
./start.sh                  # ✅ CORRECT
streamlit run src/app.py    # ✅ CORRECT (with PYTHONPATH set)
```

---

## 🐛 **Still Having Issues?**

### Check Your Location
```bash
pwd
# Should output: /Users/silarsharief/Desktop/videorag
```

### Check Project Structure
```bash
ls -la
# Should see: config/, src/, data/, models/, run_app.py, start.sh
```

### Set PYTHONPATH Manually
```bash
export PYTHONPATH=/Users/silarsharief/Desktop/videorag
streamlit run src/app.py
```

---

## 📚 **Additional Commands**

```bash
# Test connections
python tests/test_connections.py

# List available Gemini models
python tests/test_gemini.py

# Test video ingestion
python tests/test_ingester.py
```

---

**Summary:** Always be in `/Users/silarsharief/Desktop/videorag/` and use `python run_app.py` - easiest way! 🚀

