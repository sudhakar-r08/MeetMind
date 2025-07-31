import streamlit as st
import whisper
import ollama
import tempfile
import os
import json
from datetime import datetime
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from chromadb.config import Settings
import uuid
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from io import BytesIO
import base64


# Set page config with better styling
st.set_page_config(
    page_title="MeetMind - AI Meeting Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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

    .chat-message {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .ai-response {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'meeting_notes' not in st.session_state:
    st.session_state.meeting_notes = ""
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_meeting_id' not in st.session_state:
    st.session_state.current_meeting_id = None
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}


class DocumentProcessor:
    """Handle different document types"""

    @staticmethod
    def extract_text_from_pdf(file_content):
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None

    @staticmethod
    def extract_text_from_docx(file_content):
        """Extract text from Word document"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return None

    @staticmethod
    def extract_text_from_txt(file_content):
        """Extract text from plain text file"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return None


class MeetMindRAG:
    def __init__(self, persist_directory="./meetmind_db"):
        """Initialize RAG system with ChromaDB and SentenceTransformer"""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        try:
            self.collection = self.client.get_collection("meetings")
        except:
            self.collection = self.client.create_collection(
                name="meetings",
                metadata={"description": "MeetMind meeting transcripts, notes, and documents"}
            )

        # Initialize embedding model
        self.embedding_model = self._load_embedding_model()

    @st.cache_resource
    def _load_embedding_model(_self):
        """Load sentence transformer model for embeddings"""
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            return None

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def add_meeting(self, meeting_id, title, transcript="", notes="", documents=None, timestamp=None):
        """Add meeting to vector database including documents"""
        if not self.embedding_model:
            return False

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        try:
            # Prepare documents and metadata
            documents_to_add = []
            metadatas = []
            ids = []

            # Add transcript chunks if available
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

            # Add meeting notes if available
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

            # Add supporting documents if available
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

            # Add to ChromaDB
            if documents_to_add:
                self.collection.add(
                    documents=documents_to_add,
                    metadatas=metadatas,
                    ids=ids
                )

            return True

        except Exception as e:
            st.error(f"Failed to add meeting to vector DB: {e}")
            return False

    def search_meetings(self, query, n_results=8, meeting_id=None, content_types=None):
        """Search meetings using semantic similarity with filtering"""
        try:
            where_clause = {}
            if meeting_id:
                where_clause["meeting_id"] = meeting_id

            if content_types:
                where_clause["type"] = {"$in": content_types}

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )

            return results

        except Exception as e:
            st.error(f"Search failed: {e}")
            return None

    def get_meeting_history(self):
        """Get list of all meetings with statistics"""
        try:
            # Get all unique meetings
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

            # Convert sets to lists for JSON serialization
            for meeting in meeting_info.values():
                meeting['content_types'] = list(meeting['content_types'])

            return meeting_info

        except Exception as e:
            st.error(f"Failed to get meeting history: {e}")
            return {}


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached)"""
    return MeetMindRAG()


@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model (cached to avoid reloading)"""
    try:
        return whisper.load_model(model_size)
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_audio(audio_file, model):
    """Transcribe audio using Whisper"""
    try:
        with st.spinner("🎯 Transcribing audio... This may take a few minutes."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name

            # Transcribe
            result = model.transcribe(tmp_file_path)

            # Clean up temp file
            os.unlink(tmp_file_path)

            return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None


def query_ollama(prompt, model="llama3"):
    """Query Ollama with a prompt"""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        st.error(f"Ollama query failed: {e}")
        return None


def generate_meeting_notes(transcript):
    """Generate structured meeting notes from transcript"""
    prompt = f"""
    Analyze the following meeting transcript and create comprehensive meeting notes. 
    Structure your response with clear sections including:

    ## 📋 Meeting Summary

    ## 🔑 Key Discussion Points

    ## ✅ Action Items

    ## 📊 Decisions Made

    ## ❓ Important Questions Raised

    ## 🚀 Next Steps

    Transcript:
    {transcript}

    Please provide clear, professional meeting notes with the above structure:
    """

    return query_ollama(prompt)


def extract_logic_and_code(transcript):
    """Extract discussed logic/algorithms and generate Python code"""
    prompt = f"""
    Analyze this meeting transcript for any algorithms, processes, workflows, or technical logic discussed.
    For each piece of logic found:
    1. Describe what was discussed
    2. Generate clean, well-commented Python code that implements the logic
    3. Explain the code and its purpose

    Format your response as:
    ### 💻 Logic Found: [Title]
    **📝 Description:** [What was discussed]

    **🐍 Python Code:**
    ```python
    [Clean, commented code]
    ```

    **💡 Explanation:** [Code explanation]

    ---

    Transcript:
    {transcript}
    """

    return query_ollama(prompt)


def parse_code_snippets(response):
    """Parse code snippets from LLM response"""
    snippets = []

    # Find all code blocks
    code_pattern = r'```python\n(.*?)\n```'
    codes = re.findall(code_pattern, response, re.DOTALL)

    # Find logic titles
    title_pattern = r'### 💻 Logic Found: (.*?)\n'
    titles = re.findall(title_pattern, response)

    # Find descriptions
    desc_pattern = r'\*\*📝 Description:\*\* (.*?)\n'
    descriptions = re.findall(desc_pattern, response)

    # Find explanations
    exp_pattern = r'\*\*💡 Explanation:\*\* (.*?)(?=---|$)'
    explanations = re.findall(exp_pattern, response, re.DOTALL)

    # Combine all parts
    for i in range(min(len(codes), len(titles))):
        snippet = {
            'title': titles[i] if i < len(titles) else f"Code Snippet {i + 1}",
            'description': descriptions[i] if i < len(descriptions) else "No description",
            'code': codes[i],
            'explanation': explanations[i].strip() if i < len(explanations) else "No explanation"
        }
        snippets.append(snippet)

    return snippets


def rag_chat_with_meetings(question, rag_system, meeting_id=None, content_types=None):
    """RAG-powered chat using vector search with content type filtering"""
    # Search for relevant context
    search_results = rag_system.search_meetings(
        question,
        n_results=8,
        meeting_id=meeting_id,
        content_types=content_types
    )

    if not search_results or not search_results['documents'][0]:
        return "I couldn't find relevant information to answer your question."

    # Build context from search results
    context = ""
    for i, doc in enumerate(search_results['documents'][0]):
        metadata = search_results['metadatas'][0][i]
        source_info = f"{metadata['title']} - {metadata['type']}"
        if metadata.get('document_name'):
            source_info += f" ({metadata['document_name']})"

        context += f"📄 From {source_info}:\n{doc}\n\n"

    prompt = f"""
    You are MeetMind, an intelligent meeting assistant. Use the relevant meeting information below to answer the user's question accurately.

    📚 Relevant Meeting Context:
    {context}

    ❓ Question: {question}

    Please provide a helpful and accurate answer based on the meeting content. If you reference specific information, mention which source it came from (transcript, notes, or document name). If the context doesn't contain enough information to answer the question, say so clearly.
    """

    return query_ollama(prompt)


def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and extract text"""
    processed_docs = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_content = uploaded_file.getvalue()

        # Determine file type and extract text
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


# Initialize RAG system
if st.session_state.vector_db is None:
    st.session_state.vector_db = initialize_rag_system()

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🧠 MeetMind - AI Meeting Assistant</h1>
    <p>Transform your meetings with AI-powered transcription, analysis & intelligent search</p>
    <p><em>✨ Now with support for meeting documents & enhanced UI</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model selection in styled containers
    with st.container():
        st.markdown("**🎙️ Audio Processing**")
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )

    with st.container():
        st.markdown("**🤖 AI Processing**")
        ollama_model = st.selectbox(
            "Ollama Model",
            ["llama3", "llama2", "codellama", "mistral"],
            index=0,
            help="Make sure the model is installed in Ollama"
        )

    st.markdown("---")

    # Enhanced meeting history
    st.markdown("### 📚 Meeting Library")

    if st.session_state.vector_db:
        meeting_history = st.session_state.vector_db.get_meeting_history()

        if meeting_history:
            # Show statistics
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

            # Meeting list
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

    # Feature highlights
    st.markdown("""
    ### ✨ Enhanced Features

    🎙️ **Audio Transcription**  
    High-quality speech-to-text

    📄 **Document Support**  
    PDF, Word, Text files

    🧠 **Smart Search**  
    Semantic search across all content

    💬 **AI Chat**  
    Ask questions about meetings

    💻 **Code Extraction**  
    Auto-generate Python code

    🔒 **Privacy First**  
    Everything runs locally
    """)

# Main content with enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload & Process",
    "📝 Meeting Notes",
    "💻 Code Extraction",
    "💬 Smart Chat",
    "🔍 Advanced Search",
    "📊 Analytics"
])
with tab1:
    st.markdown("### 🎯 Process New Meeting")

    # Meeting metadata in a nice form
    col1, col2 = st.columns(2)
    with col1:
        meeting_title = st.text_input(
            "📋 Meeting Title",
            placeholder="e.g., Q1 Planning Session - Product Team",
            help="Give your meeting a descriptive title"
        )
    with col2:
        meeting_date = st.date_input("📅 Meeting Date", value=datetime.now().date())

    # Enhanced upload section
    st.markdown("""
        <div class="upload-section">
            <h4>🎙️ Upload Meeting Audio</h4>
            <p>Supported formats: WAV, MP3, M4A, FLAC, OGG</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_audio = st.file_uploader(
        "Choose audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        label_visibility="collapsed"
    )

    # Document upload section
    st.markdown("""
        <div class="upload-section">
            <h4>📄 Upload Supporting Documents</h4>
            <p>Upload meeting agendas, presentations, reports (PDF, Word, Text)</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_docs = st.file_uploader(
        "Choose document files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Audio player and processing
    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        # Document preview
        if uploaded_docs:
            st.markdown("#### 📚 Uploaded Documents:")
            for doc in uploaded_docs:
                st.markdown(f"📄 **{doc.name}** ({doc.size:,} bytes)")

        # Processing buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Transcribe Audio", type="primary", use_container_width=True):
                model = load_whisper_model(whisper_model)
                if model:
                    transcript = transcribe_audio(uploaded_audio, model)
                    if transcript:
                        st.session_state.transcript = transcript
                        st.markdown('<div class="success-banner">✅ Audio transcribed successfully!</div>',
                                    unsafe_allow_html=True)
                        with st.expander("📄 Transcript Preview", expanded=True):
                            st.text_area("", transcript, height=200, key="transcript_preview")

        with col2:
            if st.button("📋 Generate Notes", use_container_width=True) and st.session_state.transcript:
                with st.spinner("🤖 AI is analyzing the meeting..."):
                    notes = generate_meeting_notes(st.session_state.transcript)
                    if notes:
                        st.session_state.meeting_notes = notes
                        st.markdown('<div class="success-banner">✅ Meeting notes generated!</div>', unsafe_allow_html=True)

        with col3:
            if st.button("💾 Save Everything", use_container_width=True):
                if meeting_title.strip() and (st.session_state.transcript or uploaded_docs):
                    # Process documents
                    processed_docs = {}
                    if uploaded_docs:
                        processed_docs = process_uploaded_documents(uploaded_docs)

                    # Save to vector DB
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
                        st.markdown(f'''
                            <div class="success-banner">
                                🎉 Meeting "{meeting_title}" saved successfully!<br>
                                📊 Includes: {"✅ Audio" if st.session_state.transcript else "❌ Audio"} | 
                                {"✅ Notes" if st.session_state.meeting_notes else "❌ Notes"} | 
                                📄 {len(processed_docs)} Documents
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to save meeting")
                else:
                    st.error("⚠️ Please provide a meeting title and at least audio or documents")

with tab2:
    st.markdown("### 📝 AI-Generated Meeting Notes")

    if st.session_state.meeting_notes:
        # Display notes in a styled container
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.meeting_notes)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download options
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
            # Convert to markdown for better formatting
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
            <h4>📋 No Meeting Notes Yet</h4>
            <p>Upload an audio file and generate notes to see them here.</p>
            <p><em>💡 Tip: Notes will include summary, action items, decisions, and next steps!</em></p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 💻 Code & Logic Extraction")

    if st.session_state.transcript:
        if st.button("🔍 Extract Code & Logic", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing transcript for technical content..."):
                code_response = extract_logic_and_code(st.session_state.transcript)
                if code_response:
                    st.session_state.code_snippets = parse_code_snippets(code_response)
                    st.success("✅ Code extraction completed!")

    # Display extracted code snippets
    if st.session_state.code_snippets:
        for i, snippet in enumerate(st.session_state.code_snippets):
            st.markdown(f"""
            <div class="feature-card">
                <h4>💻 {snippet['title']}</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**📝 Description:** {snippet['description']}")

            # Code display with syntax highlighting
            st.code(snippet['code'], language='python')

            st.markdown(f"**💡 Explanation:** {snippet['explanation']}")

            # Download button for each snippet
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label=f"💾 Save Code",
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
            <h4>💻 No Code Extracted Yet</h4>
            <p>Process a transcript first, then extract technical logic and code implementations.</p>
            <p><em>💡 Tip: Works best with meetings that discuss algorithms, workflows, or technical processes!</em></p>
        </div>
        """, unsafe_allow_html=True)


with tab4:
    st.markdown("### 💬 Smart Chat with AI")

    # Enhanced chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_input(
            "🤖 Ask MeetMind anything about your meetings:",
            placeholder="e.g., What were the main action items? Who is responsible for the budget review?",
            key="smart_chat_input"
        )

    with col2:
        search_scope = st.selectbox(
            "🎯 Search Scope",
            ["All Meetings", "Current Meeting Only"],
            help="Choose whether to search all meetings or just the current one"
        )

        content_filter = st.multiselect(
            "📄 Content Types",
            ["transcript", "notes", "document"],
            default=["transcript", "notes", "document"],
            help="Select which types of content to search"
        )

    if st.button("🚀 Ask MeetMind", type="primary", use_container_width=True) and question:
        with st.spinner("🧠 Searching through your meetings..."):
            meeting_filter = st.session_state.current_meeting_id if search_scope == "Current Meeting Only" else None
            answer = rag_chat_with_meetings(
                question,
                st.session_state.vector_db,
                meeting_filter,
                content_filter if content_filter else None
            )

            if answer:
                # Add to history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'scope': search_scope,
                    'content_types': content_filter
                })

    # Enhanced chat history display
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")

        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            # Question
            st.markdown(f"""
            <div class="chat-message">
                <strong>🙋 You ({entry['timestamp']}):</strong><br>
                {entry['question']}
                <br><small>🎯 Scope: {entry['scope']} | 📄 Types: {', '.join(entry.get('content_types', ['all']))}</small>
            </div>
            """, unsafe_allow_html=True)

            # Answer
            st.markdown(f"""
            <div class="ai-response">
                <strong>🧠 MeetMind:</strong><br>
                {entry['answer']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

    # Clear chat and export options
    if st.session_state.chat_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            # Export chat history
            chat_export = "\n".join([
                f"Q ({entry['timestamp']}): {entry['question']}\nA: {entry['answer']}\n---\n"
                for entry in st.session_state.chat_history
            ])
            st.download_button(
                label="💾 Export Chat",
                data=chat_export,
                file_name=f"meetmind_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

with tab5:
    st.markdown("### 🔍 Advanced Search & Analytics")

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "🔍 Search across all meetings and documents:",
            placeholder="e.g., budget constraints, project timeline, technical requirements"
        )

    with col2:
        search_content_types = st.multiselect(
            "Content Filter",
            ["transcript", "notes", "document"],
            default=["transcript", "notes", "document"]
        )

    # Advanced search options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results", 5, 20, 10)
        with col2:
            min_relevance = st.slider("Minimum relevance", 0.0, 1.0, 0.3, 0.1)

    if st.button("🔍 Search", type="primary", use_container_width=True) and search_query:
        with st.spinner("🔍 Searching through your meeting database..."):
            results = st.session_state.vector_db.search_meetings(
                search_query,
                n_results=num_results,
                content_types=search_content_types if search_content_types else None
            )

            if results and results['documents'][0]:
                # Filter by relevance
                filtered_results = []
                for i, doc in enumerate(results['documents'][0]):
                    relevance = 1 - results['distances'][0][i] if 'distances' in results else 1.0
                    if relevance >= min_relevance:
                        filtered_results.append((doc, results['metadatas'][0][i], relevance))

                if filtered_results:
                    st.markdown(f"### 📊 Found {len(filtered_results)} relevant results")

                    # Results summary
                    content_type_counts = {}
                    meeting_counts = {}

                    for _, metadata, _ in filtered_results:
                        content_type = metadata['type']
                        meeting_title = metadata['title']

                        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
                        meeting_counts[meeting_title] = meeting_counts.get(meeting_title, 0) + 1

                    # Display summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📄 Content Types</h4>
                            {'<br>'.join([f"{k}: {v}" for k, v in content_type_counts.items()])}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎯 Meetings Found</h4>
                            <p>{len(meeting_counts)} unique meetings</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        avg_relevance = sum(r[2] for r in filtered_results) / len(filtered_results)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📈 Avg Relevance</h4>
                            <p>{avg_relevance:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display results
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
                    st.info(
                        f"No results found with relevance >= {min_relevance:.1%}. Try lowering the minimum relevance threshold.")
            else:
                st.info("No results found. Try different keywords or check your content type filters.")

with tab6:
    st.markdown("### 📊 Meeting Analytics & Insights")

    if st.session_state.vector_db:
        meeting_history = st.session_state.vector_db.get_meeting_history()

        if meeting_history:
            # Overall statistics
            total_meetings = len(meeting_history)
            total_chunks = sum(m['chunk_count'] for m in meeting_history.values())
            total_docs = sum(m['document_count'] for m in meeting_history.values())

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

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

            with col4:
                avg_chunks = total_chunks / total_meetings if total_meetings > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈</h3>
                    <h2>{avg_chunks:.1f}</h2>
                    <p>Avg Chunks/Meeting</p>
                </div>
                """, unsafe_allow_html=True)

            # Content type breakdown
            st.markdown("### 📊 Content Analysis")

            content_type_stats = {'transcript': 0, 'notes': 0, 'document': 0}
            meeting_sizes = []
            recent_meetings = []

            for meeting_id, info in meeting_history.items():
                meeting_sizes.append(info['chunk_count'])

                # Parse timestamp
                try:
                    meeting_dt = datetime.fromisoformat(info['timestamp'])
                    recent_meetings.append((meeting_dt, info['title']))
                except:
                    pass

                for content_type in info['content_types']:
                    if content_type in content_type_stats:
                        content_type_stats[content_type] += 1

            # Content distribution
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📄 Content Types Distribution")
                for content_type, count in content_type_stats.items():
                    percentage = (count / total_meetings * 100) if total_meetings > 0 else 0
                    st.markdown(f"**{content_type.title()}:** {count} meetings ({percentage:.1f}%)")
                    st.progress(percentage / 100)

            with col2:
                st.markdown("#### 📈 Meeting Sizes")
                if meeting_sizes:
                    avg_size = sum(meeting_sizes) / len(meeting_sizes)
                    max_size = max(meeting_sizes)
                    min_size = min(meeting_sizes)

                    st.markdown(f"**Average:** {avg_size:.1f} chunks")
                    st.markdown(f"**Largest:** {max_size} chunks")
                    st.markdown(f"**Smallest:** {min_size} chunks")

            # Recent activity
            if recent_meetings:
                st.markdown("### 📅 Recent Activity")
                recent_meetings.sort(reverse=True)  # Most recent first

                for i, (meeting_dt, title) in enumerate(recent_meetings[:5]):
                    days_ago = (datetime.now() - meeting_dt).days
                    time_str = f"{days_ago} days ago" if days_ago > 0 else "Today"

                    st.markdown(f"**{meeting_dt.strftime('%Y-%m-%d')}** ({time_str}): {title}")

            # Detailed meeting breakdown
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
                <p>Upload and process some meetings to see analytics here.</p>
                <p><em>💡 Tip: Analytics will show meeting trends, content distribution, and usage patterns!</em></p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; color: white; margin-top: 2rem;'>
        <h3>🧠 MeetMind - Your Intelligent Meeting Assistant</h3>
        <p><strong>✨ Enhanced Features:</strong> Document Support | Advanced Search | Better UI | Analytics Dashboard</p>
        <p><strong>🔒 Privacy:</strong> All processing happens locally - your data never leaves your machine</p>
        <p><strong>💾 Persistent:</strong> Your meetings and documents are stored locally and searchable forever</p>
        <p><em>Built with ❤️ using Streamlit, Whisper, Ollama, and ChromaDB</em></p>
    </div>
    """, unsafe_allow_html=True)