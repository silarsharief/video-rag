# 🔧 Troubleshooting Guide

## ✅ Issues Fixed

### 1. ❌ `ModuleNotFoundError: No module named 'src'`

**Problem:** Python couldn't find the `src` module when importing.

**Solution:** Updated `src/app.py` and `run_app.py` to add project root to Python path.

**Status:** ✅ **FIXED**

---

### 2. ❌ `ValueError: 'NEO4J_URI' missing in .env`

**Problem:** The `.env` file was in `forensic-rag/` but needed to be at project root.

**Solution:**
1. Copied `.env` file from `forensic-rag/.env` to `/Users/silarsharief/Desktop/videorag/.env`
2. Updated `config/settings.py` to explicitly load `.env` from project root

**Status:** ✅ **FIXED**

---

## 🎯 Current Setup

Your `.env` file is now at: `/Users/silarsharief/Desktop/videorag/.env`

It contains:
- `GEMINI_API_KEY` - Google Gemini API credentials
- `NEO4J_URI` - Neo4j Aura database URI
- `NEO4J_USERNAME` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password

---

## 🚀 How to Run (After Fixes)

```bash
cd /Users/silarsharief/Desktop/videorag
python run_app.py
```

Should now work without errors! 🎉

---

## 🔍 Common Issues & Solutions

### Issue: App can't find .env file
**Check:**
```bash
ls -la /Users/silarsharief/Desktop/videorag/.env
```

**Solution:** Make sure `.env` exists at project root (not in subdirectories)

---

### Issue: Import errors
**Check:**
```bash
pwd  # Should output: /Users/silarsharief/Desktop/videorag
```

**Solution:** Always run from project root directory

---

### Issue: Neo4j connection fails
**Check your .env file has:**
```
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

**Test connection:**
```bash
python tests/test_connections.py
```

---

### Issue: Gemini API fails
**Check your .env file has:**
```
GEMINI_API_KEY="your_api_key_here"
```

**Test API:**
```bash
python tests/test_gemini.py
```

---

## 📝 File Locations Reference

| File/Folder | Location | Purpose |
|-------------|----------|---------|
| `.env` | `/Users/silarsharief/Desktop/videorag/.env` | Environment variables |
| `run_app.py` | `/Users/silarsharief/Desktop/videorag/run_app.py` | App launcher |
| `src/app.py` | `/Users/silarsharief/Desktop/videorag/src/app.py` | Main application |
| `config/` | `/Users/silarsharief/Desktop/videorag/config/` | All configuration |
| `data/` | `/Users/silarsharief/Desktop/videorag/data/` | Videos, databases |
| `models/` | `/Users/silarsharief/Desktop/videorag/models/` | YOLO models |

---

## ✅ Quick Health Check

Run this to test everything:

```bash
cd /Users/silarsharief/Desktop/videorag
python tests/test_connections.py
```

Should show:
- ✅ Gemini API connected
- ✅ Neo4j connected
- ✅ ChromaDB working
- ✅ YOLO models loaded

---

## 🆘 Still Having Issues?

1. **Verify you're in the right directory:**
   ```bash
   pwd
   # Should output: /Users/silarsharief/Desktop/videorag
   ```

2. **Check .env file exists:**
   ```bash
   cat .env
   # Should show your credentials
   ```

3. **Check Python path:**
   ```bash
   echo $PYTHONPATH
   # Should include project root or be empty (script sets it)
   ```

4. **Reactivate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

---

## 📚 Related Documentation

- **How to Run:** See `HOW_TO_RUN.md`
- **Quick Start:** See `QUICK_START.md`
- **Migration Guide:** See `MIGRATION_GUIDE.md`
- **Full Summary:** See `REORGANIZATION_SUMMARY.md`

---

**Everything should work now!** The two main issues (imports and .env location) are fixed. 🎉

