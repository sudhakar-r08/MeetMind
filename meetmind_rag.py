import streamlit as st
import ollama
import numpy as np
from faster_whisper import WhisperModel
import pyaudio
import queue
import threading
import time
import warnings
import tempfile
import os
import json
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import uuid
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from io import BytesIO
import base64

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# Load faster-whisper model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_whisper_model()

# Load embeddings for RAG
@st.cache_resource
def load_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2')

embeddings = load_embeddings()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm MeetMind with RAG. Upload docs, transcribe meetings, or query for code/docs."}]
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "recording" not in st.session_state:
    st.session_state.recording = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "meeting_notes" not in st.session_state:
    st.session_state.meeting_notes = ""
if "code_snippets" not in st.session_state:
    st.session_state.code_snippets = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = None
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = "tiny"
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "llama3:8b"

# Document Processor
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_content):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None

    @staticmethod
    def extract_text_from_docx(file_content):
        try:
            doc = docx.Document(BytesIO(file_content))
            text = "".join(paragraph.text + "\n" for paragraph in doc.paragraphs)
            return text.strip()
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return None

    @staticmethod
    def extract_text_from_txt(file_content):
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return None

# RAG System
class MeetMindRAG:
    def __init__(self, persist_directory="./meetmind_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection("meetings")
        except:
            self.collection = self.client.create_collection(name="meetings")
        self.embedding_model = embeddings

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def add_meeting(self, meeting_id, title, transcript="", notes="", documents=None, timestamp=None):
        if not self.embedding_model:
            return False
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        try:
            documents_to_add = []
            metadatas = []
            ids = []
            if transcript:
                transcript_chunks = self.chunk_text(transcript)
                for i, chunk in enumerate(transcript_chunks):
                    documents_to_add.append(chunk)
                    metadatas.append({
                        "meeting_id": meeting_id,
                        "title": title,
                        "type": "transcript",
                        "chunk_id": i,
                        "timestamp": timestamp,
                        "source": "audio_transcription"
                    })
                    ids.append(f"{meeting_id}_transcript_{i}")
            if notes:
                documents_to_add.append(notes)
                metadatas.append({
                    "meeting_id": meeting_id,
                    "title": title,
                    "type": "notes",
                    "chunk_id": 0,
                    "timestamp": timestamp,
                    "source": "ai_generated"
                })
                ids.append(f"{meeting_id}_notes")
            if documents:
                for doc_name, doc_content in documents.items():
                    doc_chunks = self.chunk_text(doc_content)
                    for i, chunk in enumerate(doc_chunks):
                        documents_to_add.append(chunk)
                        metadatas.append({
                            "meeting_id": meeting_id,
                            "title": title,
                            "type": "document",
                            "chunk_id": i,
                            "timestamp": timestamp,
                            "source": doc_name,
                            "document_name": doc_name
                        })
                        ids.append(f"{meeting_id}_doc_{doc_name}_{i}")
            if documents_to_add:
                embeddings = [self.embedding_model.encode(doc).tolist() for doc in documents_to_add]
                self.collection.add(documents=documents_to_add, embeddings=embeddings, metadatas=metadatas, ids=ids)
            return True
        except Exception as e:
            st.error(f"Failed to add meeting to vector DB: {e}")
            return False

    def search_meetings(self, query, n_results=8, meeting_id=None, content_types=None):
        try:
            where_clause = {}
            if meeting_id:
                where_clause["meeting_id"] = meeting_id
            if content_types:
                where_clause["type"] = {"$in": content_types}
            results = self.collection.query(
                query_embeddings=self.embedding_model.encode(query).tolist(),
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            return results
        except Exception as e:
            st.error(f"Search failed: {e}")
            return None

    def get_meeting_history(self):
        try:
            all_docs = self.collection.get()
            meeting_info = {}
            for metadata in all_docs['metadatas']:
                meeting_id = metadata['meeting_id']
                if meeting_id not in meeting_info:
                    meeting_info[meeting_id] = {
                        'title': metadata['title'],
                        'timestamp': metadata['timestamp'],
                        'content_types': set(),
                        'document_count': 0,
                        'chunk_count': 0
                    }
                meeting_info[meeting_id]['content_types'].add(metadata['type'])
                meeting_info[meeting_id]['chunk_count'] += 1
                if metadata['type'] == 'document':
                    meeting_info[meeting_id]['document_count'] += 1
            for meeting in meeting_info.values():
                meeting['content_types'] = list(meeting['content_types'])
            return meeting_info
        except Exception as e:
            st.error(f"Failed to get meeting history: {e}")
            return {}

# Audio Processing
audio_queue = queue.Queue()

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while st.session_state.recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio():
    buffer = np.array([], dtype=np.float32)
    while st.session_state.recording or not audio_queue.empty():
        if not audio_queue.empty():
            chunk = audio_queue.get()
            buffer = np.append(buffer, chunk)
            if len(buffer) >= RATE * 5:
                segments, _ = model.transcribe(buffer, language="en")
                text = "".join(segment.text for segment in segments).strip()
                if text:
                    st.session_state.transcript += text + " "
                    st.session_state.messages.append({"role": "assistant", "content": f"Live chunk: {text}"})
                    st.rerun()
                buffer = np.array([], dtype=np.float32)
        time.sleep(0.1)

def transcribe_audio(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        segments, _ = model.transcribe(tmp_file_path, language="en")
        os.unlink(tmp_file_path)
        return "".join(segment.text for segment in segments).strip()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# LLM Functions
def stream_response(generator):
    response = ""
    for chunk in generator:
        response += chunk
        yield response

def ollama_response(system_prompt, user_prompt, context="", model="llama3:8b"):
    full_prompt = f"Context from documents/meeting notes: {context}\n\n{user_prompt}"
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': full_prompt[:2000]}
    ])['message']['content']
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def generate_notes():
    context = retrieve_context("Summarize meeting")
    return ollama_response(
        "Summarize concisely into key notes, actions, decisions. Use context if relevant.",
        st.session_state.transcript,
        context
    )

def generate_code_and_explanation():
    context = retrieve_context("Extract logic for code generation")
    return ollama_response(
        "Extract logic from transcript/context and generate Python code with explanation.",
        st.session_state.transcript,
        context,
        model="codellama:7b"
    )

def qa_chat(user_question):
    context = retrieve_context(user_question)
    return ollama_response(
        "Answer query using context from documents and meeting notes. Generate code or documents if asked.",
        user_question,
        context
    )

def parse_code_snippets(response):
    snippets = []
    code_pattern = r'```python\n(.*?)\n```'
    codes = re.findall(code_pattern, response, re.DOTALL)
    title_pattern = r'###\s*[^:]+:\s*(.*?)\n'
    titles = re.findall(title_pattern, response)
    desc_pattern = r'\*\*[^:]+:\*\*\s*(.*?)\n'
    descriptions = re.findall(desc_pattern, response)
    exp_pattern = r'\*\*[^:]+:\*\*\s*(.*?)(?=---|$|\n\n)'
    explanations = re.findall(exp_pattern, response, re.DOTALL)
    for i in range(min(len(codes), len(titles))):
        snippet = {
            'title': titles[i] if i < len(titles) else f"Code Snippet {i + 1}",
            'description': descriptions[i] if i < len(descriptions) else "No description",
            'code': codes[i],
            'explanation': explanations[i].strip() if i < len(explanations) else "No explanation"
        }
        snippets.append(snippet)
    return snippets

def retrieve_context(query, k=3):
    if st.session_state.vector_db is None:
        return ""
    results = st.session_state.vector_db.search_meetings(query, k=k)
    if not results or not results['documents'][0]:
        return ""
    return "\n\n".join(doc for doc in results['documents'][0])

def process_uploaded_documents(uploaded_files):
    processed_docs = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_content = uploaded_file.getvalue()
        if file_name.lower().endswith('.pdf'):
            text = DocumentProcessor.extract_text_from_pdf(file_content)
        elif file_name.lower().endswith('.docx'):
            text = DocumentProcessor.extract_text_from_docx(file_content)
        elif file_name.lower().endswith('.txt'):
            text = DocumentProcessor.extract_text_from_txt(file_content)
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue
        if text:
            processed_docs[file_name] = text
            st.success(f"✅ Processed: {file_name} ({len(text)} characters)")
        else:
            st.error(f"❌ Failed to process: {file_name}")
    return processed_docs

# CSS for Claude-like UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .status-card {
        background: linear-gradient(45deg, #00d4aa, #00a8cc);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success-banner {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        align-self: flex-end;
        background: #dcf8c6;
        padding: 1rem;
        border-radius: 10px 10px 0 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .assistant-message {
        align-self: flex-start;
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px 10px 10px 0;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    st.markdown("### 🧠 MeetMind Menu")
    selected_page = st.selectbox("Navigate", [
        "Upload & Process",
        "Meeting Notes",
        "Code Extraction",
        "Smart Chat",
        "Advanced Search",
        "Analytics"
    ])
    st.markdown("---")
    st.markdown("### 📚 Meeting Library")
    if st.session_state.vector_db:
        meeting_history = st.session_state.vector_db.get_meeting_history()
        if meeting_history:
            total_meetings = len(meeting_history)
            total_chunks = sum(m['chunk_count'] for m in meeting_history.values())
            total_docs = sum(m['document_count'] for m in meeting_history.values())
            st.markdown(f"""
            <div class="status-card">
                <h4>📊 Library Stats</h4>
                <p>🎯 {total_meetings} Meetings</p>
                <p>📄 {total_docs} Documents</p>
                <p>🧩 {total_chunks} Chunks</p>
            </div>
            """, unsafe_allow_html=True)
            for meeting_id, info in meeting_history.items():
                content_types = ", ".join(info['content_types'])
                with st.expander(f"📅 {info['title'][:25]}...", expanded=False):
                    st.markdown(f"**ID:** `{meeting_id[:8]}...`")
                    st.markdown(f"**Date:** {info['timestamp'][:19]}")
                    st.markdown(f"**Content:** {content_types}")
                    st.markdown(f"**Documents:** {info['document_count']}")
                    if st.button(f"🎯 Switch to Meeting", key=f"load_{meeting_id}"):
                        st.session_state.current_meeting_id = meeting_id
                        st.success(f"✅ Switched to: {info['title'][:30]}...")
                        st.rerun()
        else:
            st.info("📝 No meetings stored yet")
    st.markdown("---")
    st.markdown("### ⚙️ Configurations")
    st.session_state.whisper_model = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=0,
        help="Use 'tiny' for fastest transcription"
    )
    st.session_state.ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3:8b", "codellama:7b", "mistral:7b"],
        index=0,
        help="Use smaller models for speed"
    )

# Main Content
st.markdown("""
<div class="main-header">
    <h1>🧠 MeetMind - AI Meeting Assistant</h1>
    <p>Transform meetings with AI-powered transcription, analysis, and RAG search</p>
</div>
""", unsafe_allow_html=True)

# Initialize RAG
if st.session_state.vector_db is None:
    st.session_state.vector_db = MeetMindRAG()

if selected_page == "Upload & Process":
    st.markdown("### 🎯 Process New Meeting")
    col1, col2 = st.columns(2)
    with col1:
        meeting_title = st.text_input("📋 Meeting Title", placeholder="e.g., Q1 Planning Session")
    with col2:
        meeting_date = st.date_input("📅 Meeting Date", value=datetime.now().date())
    st.markdown("""
        <div class="upload-section">
            <h4>🎙️ Upload Meeting Audio</h4>
            <p>Supported formats: WAV, MP3, M4A, FLAC, OGG</p>
        </div>
        """, unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'm4a', 'flac', 'ogg'], label_visibility="collapsed")
    st.markdown("""
        <div class="upload-section">
            <h4>📄 Upload Documents</h4>
            <p>PDF, Word, Text files</p>
        </div>
        """, unsafe_allow_html=True)
    uploaded_docs = st.file_uploader("Choose document files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_audio:
        st.audio(uploaded_audio)
    if uploaded_docs:
        st.markdown("#### 📚 Uploaded Documents:")
        for doc in uploaded_docs:
            st.markdown(f"📄 **{doc.name}** ({doc.size:,} bytes)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Transcribe Audio", type="primary", use_container_width=True):
            with st.spinner("🎯 Transcribing..."):
                transcript = transcribe_audio(uploaded_audio)
                if transcript:
                    st.session_state.transcript = transcript
                    st.markdown('<div class="success-banner">✅ Transcribed!</div>', unsafe_allow_html=True)
                    with st.expander("📄 Transcript Preview", expanded=True):
                        st.text_area("", transcript, height=200, key="transcript_preview")
    with col2:
        if st.button("📋 Generate Notes", use_container_width=True) and st.session_state.transcript:
            with st.spinner("🤖 Generating..."):
                notes = "".join(resp for resp in generate_notes())
                if notes:
                    st.session_state.meeting_notes = notes
                    st.markdown('<div class="success-banner">✅ Notes generated!</div>', unsafe_allow_html=True)
    with col3:
        if st.button("💾 Save Everything", use_container_width=True):
            if meeting_title.strip() and (st.session_state.transcript or uploaded_docs):
                processed_docs = process_uploaded_documents(uploaded_docs) if uploaded_docs else {}
                meeting_id = str(uuid.uuid4())
                success = st.session_state.vector_db.add_meeting(
                    meeting_id=meeting_id,
                    title=meeting_title,
                    transcript=st.session_state.transcript,
                    notes=st.session_state.meeting_notes,
                    documents=processed_docs,
                    timestamp=datetime.combine(meeting_date, datetime.now().time()).isoformat()
                )
                if success:
                    st.session_state.current_meeting_id = meeting_id
                    st.markdown(f'<div class="success-banner">🎉 Meeting "{meeting_title}" saved!</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to save meeting")
            else:
                st.error("⚠️ Provide a title and audio/documents")

elif selected_page == "Meeting Notes":
    st.markdown("### 📝 AI-Generated Meeting Notes")
    if st.session_state.meeting_notes:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.meeting_notes)
        st.markdown('</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="💾 Download as Text",
                data=st.session_state.meeting_notes,
                file_name=f"meetmind_notes_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            markdown_notes = f"# Meeting Notes - {datetime.now().strftime('%Y-%m-%d')}\n\n{st.session_state.meeting_notes}"
            st.download_button(
                label="📄 Download as Markdown",
                data=markdown_notes,
                file_name=f"meetmind_notes_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class="feature-card">
            <h4>📋 No Notes Yet</h4>
            <p>Upload audio and generate notes to see them here.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Code Extraction":
    st.markdown("### 💻 Code & Logic Extraction")
    if st.session_state.transcript:
        if st.button("🔍 Extract Code & Logic", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing..."):
                code_response = "".join(resp for resp in generate_code_and_explanation())
                if code_response:
                    st.session_state.code_snippets = parse_code_snippets(code_response)
                    st.success("✅ Code extracted!")
    if st.session_state.code_snippets:
        for i, snippet in enumerate(st.session_state.code_snippets):
            st.markdown(f"""
            <div class="feature-card">
                <h4>💻 {snippet['title']}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**📝 Description:** {snippet['description']}")
            st.code(snippet['code'], language='python')
            st.markdown(f"**💡 Explanation:** {snippet['explanation']}")
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="💾 Save Code",
                    data=f"# {snippet['title']}\n# {snippet['description']}\n\n{snippet['code']}",
                    file_name=f"meetmind_code_{i + 1}.py",
                    mime="text/plain",
                    key=f"download_code_{i}",
                    use_container_width=True
                )
            st.markdown("---")
    else:
        st.markdown("""
        <div class="feature-card">
            <h4>💻 No Code Yet</h4>
            <p>Process a transcript to extract code.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Smart Chat":
    st.markdown("### 💬 Smart Chat with AI")
    st.markdown('<div class="chat-container" id="chat_container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    prompt = st.chat_input("Command or query (e.g., 'start live transcription', 'generate code', 'What code for X?')")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        prompt_lower = prompt.lower()
        if "start live" in prompt_lower:
            if not st.session_state.recording:
                st.session_state.recording = True
                st.session_state.messages.append({"role": "assistant", "content": "Starting live transcription..."})
                record_thread = threading.Thread(target=record_audio, daemon=True)
                process_thread = threading.Thread(target=process_audio, daemon=True)
                record_thread.start()
                process_thread.start()
        elif "stop live" in prompt_lower:
            if st.session_state.recording:
                st.session_state.recording = False
                st.session_state.messages.append({"role": "assistant", "content": "Stopped. Transcript ready."})
                if st.sidebar.button("Re-index with new notes", key="reindex"):
                    st.session_state.vector_db = MeetMindRAG()  # Rebuild to include new transcript
        elif "generate notes" in prompt_lower:
            with st.container():
                placeholder = st.empty()
                full = ""
                for resp in stream_response(generate_notes()):
                    full = resp
                    placeholder.markdown(f'<div class="assistant-message">{full}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full})
                st.session_state.meeting_notes = full
                st.sidebar.download_button("Export Notes", full, "notes.txt")
        elif "generate code" in prompt_lower:
            with st.container():
                placeholder = st.empty()
                full = ""
                for resp in stream_response(generate_code_and_explanation()):
                    full = resp
                    placeholder.markdown(f'<div class="assistant-message">{full}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full})
                st.session_state.code_snippets = parse_code_snippets(full)
                st.sidebar.download_button("Export Code", full, "code.md")
        else:
            with st.container():
                placeholder = st.empty()
                full = ""
                for resp in stream_response(qa_chat(prompt)):
                    full = resp
                    placeholder.markdown(f'<div class="assistant-message">{full}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full})

elif selected_page == "Advanced Search":
    st.markdown("### 🔍 Advanced Search & Analytics")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search meetings/documents:", placeholder="e.g., budget constraints")
    with col2:
        search_content_types = st.multiselect("Content Filter", ["transcript", "notes", "document"], default=["transcript", "notes", "document"])
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results", 5, 20, 10)
        with col2:
            min_relevance = st.slider("Minimum relevance", 0.0, 1.0, 0.3, 0.1)
    if st.button("🔍 Search", type="primary", use_container_width=True) and search_query:
        with st.spinner("🔍 Searching..."):
            results = st.session_state.vector_db.search_meetings(search_query, n_results=num_results, content_types=search_content_types)
            if results and results['documents'][0]:
                filtered_results = []
                for i, doc in enumerate(results['documents'][0]):
                    relevance = 1 - results['distances'][0][i] if 'distances' in results else 1.0
                    if relevance >= min_relevance:
                        filtered_results.append((doc, results['metadatas'][0][i], relevance))
                if filtered_results:
                    st.markdown(f"### 📊 Found {len(filtered_results)} results")
                    for i, (doc, metadata, relevance) in enumerate(filtered_results):
                        source_info = f"{metadata['title']} - {metadata['type'].title()}"
                        if metadata.get('document_name'):
                            source_info += f" ({metadata['document_name']})"
                        with st.expander(f"📄 {source_info} (Relevance: {relevance:.1%})", expanded=i < 3):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**📅 Date:** {metadata['timestamp'][:19]}")
                                st.markdown(f"**📋 Meeting:** {metadata['title']}")
                                st.markdown(f"**📄 Type:** {metadata['type'].title()}")
                                if metadata.get('document_name'):
                                    st.markdown(f"**📎 Document:** {metadata['document_name']}")
                            with col2:
                                st.markdown(f"**🎯 Relevance**")
                                st.progress(relevance)
                                st.markdown(f"{relevance:.1%}")
                            st.markdown("**📝 Content:**")
                            st.text_area("", doc, height=150, key=f"search_result_{i}", label_visibility="collapsed")
                else:
                    st.info(f"No results with relevance >= {min_relevance:.1%}.")
            else:
                st.info("No results found.")

elif selected_page == "Analytics":
    st.markdown("### 📊 Meeting Analytics & Insights")
    if st.session_state.vector_db:
        meeting_history = st.session_state.vector_db.get_meeting_history()
        if meeting_history:
            total_meetings = len(meeting_history)
            total_chunks = sum(m['chunk_count'] for m in meeting_history.values())
            total_docs = sum(m['document_count'] for m in meeting_history.values())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯</h3>
                    <h2>{total_meetings}</h2>
                    <p>Total Meetings</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📄</h3>
                    <h2>{total_docs}</h2>
                    <p>Documents</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🧩</h3>
                    <h2>{total_chunks}</h2>
                    <p>Content Chunks</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("### 📊 Content Analysis")
            content_type_stats = {'transcript': 0, 'notes': 0, 'document': 0}
            for info in meeting_history.values():
                for content_type in info['content_types']:
                    if content_type in content_type_stats:
                        content_type_stats[content_type] += 1
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📄 Content Types Distribution")
                for content_type, count in content_type_stats.items():
                    percentage = (count / total_meetings * 100) if total_meetings > 0 else 0
                    st.markdown(f"**{content_type.title()}:** {count} meetings ({percentage:.1f}%)")
                    st.progress(percentage / 100)
            st.markdown("### 📋 Detailed Meeting Breakdown")
            for meeting_id, info in meeting_history.items():
                with st.expander(f"📊 {info['title']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**📅 Date:** {info['timestamp'][:19]}")
                        st.markdown(f"**🆔 ID:** `{meeting_id[:8]}...`")
                        st.markdown(f"**📄 Content Types:** {', '.join(info['content_types'])}")
                    with col2:
                        st.markdown(f"**🧩 Total Chunks:** {info['chunk_count']}")
                        st.markdown(f"**📎 Documents:** {info['document_count']}")
                        if st.button(f"🎯 Switch to This Meeting", key=f"analytics_switch_{meeting_id}"):
                            st.session_state.current_meeting_id = meeting_id
                            st.success(f"✅ Switched to: {info['title'][:40]}...")
                            st.rerun()
        else:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 No Analytics Data Yet</h4>
                <p>Upload and process meetings to see analytics.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; color: white; margin-top: 2rem;'>
    <h3>🧠 MeetMind - Your Intelligent Meeting Assistant</h3>
    <p><strong>✨ Features:</strong> Live Transcription | RAG Search | Code Generation | Local Privacy</p>
    <p><em>Built with Streamlit, Faster-Whisper, Ollama, ChromaDB</em></p>
</div>
""", unsafe_allow_html=True)
