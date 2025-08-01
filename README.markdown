# MeetMind RAG

## Overview
MeetMind RAG is a Streamlit-based application designed for Retrieval-Augmented Generation (RAG). It enables text processing, vector-based retrieval, similarity search, and audio transcription. The project leverages `chromadb` for vector storage, `sentence-transformers` for embeddings, `TfidfVectorizer` for text similarity, and FFmpeg-dependent libraries for transcription.

## Features
- **Text Retrieval**: Efficient vector search using `chromadb`.
- **Similarity Search**: TF-IDF and cosine similarity for text analysis.
- **Audio Transcription**: Processes audio inputs (requires FFmpeg).
- **Streamlit Interface**: Web-based UI for user interaction.
- **Extensible**: Modular structure for adding features.

## Prerequisites
- **Python**: 3.11–3.13 (tested on 3.13)
- **Git**: For cloning the repository
- **Virtual Environment**: Recommended
- **FFmpeg**: For audio transcription
- **Streamlit Cloud**: For deployment (optional)

### System Dependencies
- **SQLite**: 3.35.0+ (local; patched for Streamlit Cloud)
  - Check: `sqlite3 --version`
- **FFmpeg**: Required for transcription
  - Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Install via [ffmpeg.org](https://ffmpeg.org) or Chocolatey

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sudhakar-r08/MeetMind.git
cd MeetMind
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Create `requirements.txt`:
```
streamlit
chromadb>=0.5.0
numpy
scikit-learn
sentence-transformers
pysqlite3-binary
speech_recognition  # Adjust if using a different transcription library
pydub              # Adjust as needed
```
Install:
```bash
pip install -r requirements.txt
```

Install FFmpeg locally (see [System Dependencies](#system-dependencies)).

### 4. Run Locally
```bash
streamlit run meetmind_rag.py
```
Access at `http://localhost:8501`.

## Project Structure
```
MeetMind/
├── meetmind_rag.py             # Main application script
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (e.g., ffmpeg)
├── venv/                       # Virtual environment (gitignore)
└── README.md                   # This file
```

## Usage
- Use the Streamlit UI to query text or upload audio for transcription.
- Customize `meetmind_rag.py` to modify retrieval, similarity, or transcription logic.

## Deployment to Streamlit Cloud
1. Push your repository to GitHub.
2. Deploy via [Streamlit Cloud](https://share.streamlit.io).
3. **SQLite Fix**:
   - Add to `requirements.txt`: `pysqlite3-binary`
   - Patch `meetmind_rag.py` at the top:
     ```python
     import sys
     import pysqlite3
     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
     ```
4. **FFmpeg Fix**:
   - Create `packages.txt`:
     ```
     ffmpeg
     ```
5. **ChromaDB Space Import Fix**:
   - Use `chromadb>=0.5.0` in `requirements.txt`.
   - Replace `Space` references with `chromadb.Client` and `create_collection`:
     ```python
     import chromadb
     client = chromadb.Client()
     collection = client.create_collection(name="my_collection")
     ```
6. Redeploy and check logs.

## Troubleshooting
### SQLite Version Error
Error: `Your system has an unsupported version of sqlite3...`
- **Local**: Upgrade SQLite or install `pysqlite3-binary`.
- **Streamlit Cloud**: Apply the SQLite patch above.
- Docs: [ChromaDB Troubleshooting](https://docs.trychroma.com/troubleshooting#sqlite)

### FFmpeg Error
Error: `Transcription failed: [Errno 2] No such file or directory: 'ffmpeg'`
- **Local**: Install FFmpeg.
- **Streamlit Cloud**: Add `ffmpeg` to `packages.txt`.
- Docs: [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

### ChromaDB ImportError
Error: `cannot import name 'Space' from 'chromadb.api.types'`
- Remove `Space` imports; use `chromadb.Client` and `create_collection`.
- Update ChromaDB: `pip install --upgrade chromadb`.
- Docs: [ChromaDB API](https://docs.trychroma.com)

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature-branch`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature-branch`
5. Open a Pull Request.

## License
MIT (assumed; add `LICENSE` file if needed).

## Contact
For issues or feedback, use [GitHub Issues](https://github.com/sudhakar-r08/MeetMind/issues) or contact [sudhakar-r08](https://github.com/sudhakar-r08).