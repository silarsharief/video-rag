# Quick Start Guide

## ⚡ Start Using the New Structure in 3 Steps

### 1️⃣ Navigate to Project Root
```bash
cd /Users/silarsharief/Desktop/videorag
```

### 2️⃣ Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3️⃣ Run the Application
```bash
python run_app.py
```

That's it! 🎉

---

## 🧪 Optional: Test Everything Works

```bash
# Test all connections (Gemini, Neo4j, ChromaDB, YOLO)
python tests/test_connections.py
```

---

## 📝 Make Configuration Changes

Edit this file for any settings:
```bash
# Open in your editor
vim config/settings.py
# or
code config/settings.py
```

Common things to configure:
- API rate limits
- Frame processing intervals
- Batch sizes
- Model selection
- Path locations

---

## 📖 More Information

- **Full Details:** See `REORGANIZATION_SUMMARY.md`
- **Migration Guide:** See `MIGRATION_GUIDE.md`
- **Settings Reference:** See `config/settings.py`

---

## 🆘 Having Issues?

1. Make sure you're in the correct directory: `/Users/silarsharief/Desktop/videorag`
2. Make sure your `.env` file is at the project root
3. Make sure your virtual environment is activated
4. Run `python tests/test_connections.py` to diagnose

---

**Everything is ready to use!** Your data is safe, functionality is unchanged, just better organized. 🚀

