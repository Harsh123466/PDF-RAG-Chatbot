import streamlit as st
import os
import re
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_style = """
<style>
    [data-testid='stAppViewContainer'] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    section.main > div.block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
        margin: 0 auto;
    }
    
    h1 {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    
    [data-testid='stMarkdownContainer'] {
        background: transparent !important;
    }
    
    .stTabs [data-baseweb='tab-list'] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb='tab'] {
        border-radius: 14px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected='true'] {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
        color: white !important;
    }
    
    [data-testid='stMetricValue'] {
        color: #7c3aed !important;
        font-size: 2.2rem !important;
    }
    
    [data-testid='stMetric'] {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    .stButton>button {
        border-radius: 14px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.4);
        transform: translateY(-2px);
    }
    
    .stTextArea textarea {
        border-radius: 14px;
        border: 1px solid rgba(124, 58, 237, 0.3);
        background: rgba(15, 23, 42, 0.8);
        color: #e2e8f0;
        padding: 1rem !important;
        font-size: 1rem;
    }
    
    .stTextInput input {
        border-radius: 14px;
        border: 1px solid rgba(124, 58, 237, 0.3);
        background: rgba(15, 23, 42, 0.8);
        color: #e2e8f0;
        padding: 0.8rem !important;
    }
    
    .stFileUploader [data-testid='stFileUploader'] {
        border-radius: 14px;
        border: 2px dashed rgba(124, 58, 237, 0.4);
        background: rgba(124, 58, 237, 0.08);
        padding: 2rem !important;
    }
    
    .stSuccess, .stInfo, .stWarning {
        border-radius: 14px !important;
    }
    
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
    }
    
    [data-testid='stDivider'] {
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.3), transparent);
        margin: 2rem 0;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .answer-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        line-height: 1.8;
        font-size: 1rem;
        color: #e2e8f0;
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# INGESTION PIPELINE

# Data => Documents


def load_allpdfs():
    folder_path = "data"
    num_docs = 0
    all_docs = []

    # Check if data folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: '{folder_path}' folder not found. Please create it and add PDF files.")
        return all_docs

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            # complete file path
            pdf_path = os.path.join(folder_path,filename)
    
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
    
            all_docs.extend(docs)
            num_docs += 1

    print("total pdfs : ", num_docs)
    print("total pages : ", len(all_docs))

    return all_docs

all_pdf_document = load_allpdfs()

# CHUNKS
def splits_docs(document, chunk_size=500, chunk_overlap=50):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunked_docs = text_splitter.split_documents(document)
    return chunked_docs

chunks = splits_docs(all_pdf_document)

# EMBEDDINGS
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("Loading model.....", model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embedding dimension", self.model.get_embedding_dimension())

    def generate_embedding(self,text):
        embedding = self.model.encode(text,show_progress_bar=True)
        print("embedding shape", embedding.shape)
        return embedding
    
embedding_manager = EmbeddingManager()

# VECTOR DATABASE
class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documnets"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)

        # create a client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # create a collection
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata = {"description" : "vectore store collection for pdf embedding in RAG"}
        )

        print("initialized the vectore store with collection", self.collection_name)
        print("docs in collection", self.collection.count())

    # function for store embedding and documents in vectore store

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("length of documents and embedding is not match")

        # store => idx, doucments, embedding, metadata
        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (docs, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(docs.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(docs.page_content)

            documents_content.append(docs.page_content)
            all_metadata.append(metadata)

            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )
            
        
vector_store = VectorStoreManager()

# Only process embeddings if documents are loaded
if chunks:
    texts = [docs.page_content for docs in chunks]
    embedding = embedding_manager.generate_embedding(texts)
    vector_store.add_documents(chunks, embedding)
else:
    print("No PDF documents loaded. Vector store is empty.")


# RETRIEVAL PIPELINE
class RAGRetriever:
    def __init__(self,embedding_manager,vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        # query => embedding
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        # semantic search
        result = self.vector_store.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = top_k
        )

        # cosine similarity
        retrieved_docs = []
        if result["documents"] and result["documents"][0]:
            ids = result["ids"][0]
            metadatas = result["metadatas"][0]
            documents = result["documents"][0]
            distances = result["distances"][0]

            for i , (docs_id, metadata, document, distance) in enumerate(zip(ids,metadatas,documents,distances)):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "ids" : docs_id,
                        "metadata" : metadata,
                        "document" : document,
                        "similarity_score" : similarity_score,
                        "distance" : distance,
                        "rank" : i+1
                    })
                    
            print(f"retrieved {len(retrieved_docs)} documents")
            
        else:
            print("no documents found")

        return retrieved_docs
    
rag_retriever = RAGRetriever(embedding_manager,vector_store)


# INTEGRATE WITH LLM

import json
from langchain_groq import ChatGroq

# Load API key from environment variable
api_key_groq = os.getenv("GROQ_API_KEY")
if not api_key_groq:
    st.error("❌ GROQ_API_KEY not found. Please add it to .env file.")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key_groq,
    model="qwen/qwen3-32b",
    temperature=0.1,
    max_tokens=1024
)

def clean_response(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

# generate our retrieval-augmented output
def generate_output(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k)

    context = "\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        print("we did not find relevant context for this query")

    prompt = f"""  use given context to generate the answer for the query
                   Context: {context}
                   Query: {query} """
    response = llm.invoke([prompt.format(context=context, query=query)])   # expecting a list as prompt
    return clean_response(response.content)







# STREAMLIT UI

st.title("📚 PDF RAG Chatbot")
st.markdown("Your intelligent document analyzer powered by retrieval-augmented generation")

with st.sidebar:
    st.markdown("### 📖 Getting Started")
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Step 1:** Upload your PDF files using the upload form
        
        **Step 2:** Click "Reload Index" to process and embed your documents
        
        **Step 3:** Navigate to "Ask a Question" and start querying your documents
        
        The app extracts content, builds embeddings, and retrieves relevant sections to answer your questions.
        """)
    
    with st.expander("Pro Tips", expanded=False):
        st.markdown("""
        💡 **Ask specific questions** for better answers
        
        📄 **Upload multiple PDFs** to build a richer knowledge base
        
        🔄 **Reload after uploads** to update the index
        
        🎯 **Be concise** - shorter questions often yield clearer answers
        """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    top_k = st.slider("Retrieval Results", min_value=1, max_value=10, value=5, 
                      help="Number of document chunks to retrieve for each query")
    
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.number_input("Chunk Size", min_value=200, max_value=1000, value=500)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=200, value=50)

tab_upload, tab_query, tab_stats, tab_about = st.tabs(["📤 Upload & Manage", "❓ Ask Question", "📊 Statistics", "ℹ️ About"])

with tab_upload:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload PDF Files")
        st.markdown("Drag and drop your PDFs or click to select them")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Uploaded {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            st.success(f"✅ Successfully uploaded {len(uploaded_files)} file(s)!")
            st.info("👉 Click 'Reload Index' below to process the documents")
    
    with col2:
        st.markdown("### Current Status")
        pdf_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")] if os.path.exists("data") else []
        st.metric("PDFs Loaded", len(pdf_files))
        st.metric("Indexed Documents", len(chunks) if chunks else 0)
        st.metric("Embeddings Ready", vector_store.collection.count())
    
    st.divider()
    
    st.markdown("### Manage Index")
    col_reload, col_clear = st.columns(2)
    
    with col_reload:
        if st.button("🔄 Reload Index", use_container_width=True, key="reload_btn"):
            with st.spinner("🔄 Rebuilding vector index..."):
                all_pdf_document = load_allpdfs()
                chunks = splits_docs(all_pdf_document)
                
                if chunks:
                    vector_store.collection.delete()
                    vector_store.collection = vector_store.client.get_or_create_collection(
                        name=vector_store.collection_name,
                        metadata={"description": "vectore store collection for pdf embedding in RAG"}
                    )
                    texts = [doc.page_content for doc in chunks]
                    embeddings = embedding_manager.generate_embedding(texts)
                    vector_store.add_documents(chunks, embeddings)
                    st.success("✅ Index rebuilt successfully!")
                else:
                    st.warning("⚠️ No PDFs found in data folder")
    
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True, key="clear_btn"):
            if pdf_files:
                for file in pdf_files:
                    os.remove(os.path.join("data", file))
                st.success("✅ All PDFs cleared!")
                st.rerun()
    
    st.divider()
    
    if pdf_files:
        with st.expander("📋 View Uploaded Files", expanded=True):
            for i, file in enumerate(pdf_files, 1):
                col_name, col_delete = st.columns([4, 1])
                with col_name:
                    st.markdown(f"{i}. **{file}**")
                with col_delete:
                    if st.button("✕", key=f"del_{file}", use_container_width=True):
                        os.remove(os.path.join("data", file))
                        st.rerun()

with tab_query:
    st.markdown("### Ask Your Questions")
    st.markdown("The system will search your documents and provide answers based on the retrieved content")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    col_input, col_retrieve = st.columns([3, 1])
    
    with col_input:
        question = st.text_area(
            "Your question:",
            height=120,
            placeholder="Ask anything about your PDFs...",
            label_visibility="collapsed"
        )
    
    with col_retrieve:
        st.markdown("### Settings")
        num_results = st.slider("Results to show", 1, 10, value=3, key="query_slider")
    
    if st.button("🚀 Get Answer", use_container_width=True, key="ask_btn"):
        if not question.strip():
            st.error("❌ Please enter a question")
        elif vector_store.collection.count() == 0:
            st.error("❌ No documents in index. Please upload and reload PDFs first.")
        else:
            with st.spinner("⏳ Searching documents and generating answer..."):
                answer = generate_output(question, rag_retriever, llm, top_k=num_results)
                results = rag_retriever.retrieve(question, top_k=num_results)
                
                # Store in history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": len(results)
                })
            
            st.markdown("---")
            st.markdown("### ✅ Answer")
            st.markdown(f"""
            <div style="background: rgba(124, 58, 237, 0.1); border-left: 4px solid #7c3aed; padding: 1.5rem; border-radius: 8px; line-height: 1.8;">
            {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            with st.expander(f"📚 Retrieved Sources ({len(results)} chunks)", expanded=True):
                for idx, result in enumerate(results, 1):
                    with st.container():
                        col_rank, col_score = st.columns([1, 3])
                        with col_rank:
                            st.metric("Rank", result["rank"])
                        with col_score:
                            st.metric("Similarity", f"{result['similarity_score']:.2%}")
                        
                        st.markdown(f"**Source:** Page {result['metadata'].get('page', 'N/A')}")
                        st.text_area(
                            f"Content {idx}",
                            value=result["document"][:300] + "..." if len(result["document"]) > 300 else result["document"],
                            height=100,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.divider()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Recent Questions")
        for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{i}: {item['question'][:50]}... ({item['sources']} sources)"):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer'][:500]}...")

with tab_stats:
    st.markdown("### 📊 Index Statistics")
    
    col1, col2, col3 = st.columns(3)
    pdf_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")] if os.path.exists("data") else []
    
    with col1:
        st.metric("📄 Total PDFs", len(pdf_files))
    with col2:
        st.metric("📑 Total Chunks", len(chunks) if chunks else 0)
    with col3:
        st.metric("🔗 Total Embeddings", vector_store.collection.count())
    
    st.divider()
    
    st.markdown("### 📈 Vector Store Info")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info(f"""
        **Collection Name:** {vector_store.collection_name}
        
        **Persist Path:** Data/vector_store
        
        **Embedding Model:** all-MiniLM-L6-v2
        """)
    
    with col_info2:
        st.info(f"""
        **LLM Model:** Groq (qwen/qwen3-32b)
        
        **Temperature:** 0.1
        
        **Max Tokens:** 1024
        """)
    
    if pdf_files:
        st.markdown("### 📁 PDF Files")
        for file in sorted(pdf_files):
            st.markdown(f"- **{file}**")

with tab_about:
    st.markdown("""
    ## 🎯 About PDF RAG Chatbot
    
    This application combines **Retrieval-Augmented Generation (RAG)** with modern language models to provide intelligent answers about your PDF documents.
    
    ### 🔧 Technology Stack
    
    - **LLM Backend:** Groq API with Qwen model
    - **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
    - **Vector Database:** ChromaDB
    - **UI Framework:** Streamlit
    - **Document Processing:** LangChain Community
    
    ### 📋 How It Works
    
    1. **Document Loading:** PDFs are loaded and parsed into text
    2. **Chunking:** Content is split into overlapping chunks for better context
    3. **Embeddings:** Each chunk is converted to a semantic embedding
    4. **Vector Storage:** Embeddings are stored in ChromaDB for fast retrieval
    5. **Query Processing:** Your question is converted to an embedding
    6. **Semantic Search:** Similar document chunks are retrieved
    7. **Generation:** An LLM generates an answer based on retrieved context
    
    ### 🎓 Features
    
    ✅ Upload multiple PDFs at once
    
    ✅ Automatic document indexing and embedding
    
    ✅ Semantic search with similarity scoring
    
    ✅ Source attribution for answers
    
    ✅ Chat history for reference
    
    ✅ Configurable retrieval settings
    
    ### 🚀 Tips for Best Results
    
    - Upload PDFs with clear, readable text
    - Ask specific, well-formed questions
    - Use simpler language for technical documents
    - Reload the index after adding new PDFs
    - Check retrieved sources to verify answer accuracy
    """)

