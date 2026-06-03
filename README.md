# PDF RAG Chatbot

## Overview
PDF RAG Chatbot is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system retrieves the most relevant document chunks using vector search and generates context-aware answers using Google's Gemini model.

## Features

- Upload PDF documents
- Extract and process PDF content
- Split documents into chunks
- Generate embeddings for semantic search
- Store embeddings using ChromaDB
- Retrieve relevant document sections
- Generate answers using Gemini LLM
- Source-aware question answering

## Tech Stack

### Frontend

- Streamlit

### Backend

- Python
- LangChain

### Vector Database

- ChromaDB

### Embedding Model

- SentenceTransformer Embeddings

### LLM

- Groq API / Gemini API

## Project Workflow
PDF Upload
→ Text Extraction
→ Text Chunking
→ Embedding Generation
→ ChromaDB Storage
→ Similarity Search
→ Context Retrieval
→ Gemini Response Generation

## Installation

### Clone Repository

```
git clone 
cd PDF_RAG_CHATBOT
```

### Create Virtual Environment

```
python -m venv venv
```

### Activate Virtual Environment
Windows:

```
venv\Scripts\activate
```
Linux/Mac:

```
source venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file:

```
GEMINI_API_KEY=YOUR_API_KEY
```

### Run Application

```
streamlit run app.py
```

## Project Structure

```
PDF_RAG_CHATBOT/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   ├── pdfs/
│   └── vector_store/
│
├── templates/
│   └── index.html
│
└── rag/
    ├── pdf_loader.py
    ├── embeddings.py
    ├── vector_store.py
    └── retriever.py
```

## Future Improvements

- Multi-PDF support
- DOCX and TXT document support
- Conversation memory
- Citation-based answers
- Hybrid search (keyword + semantic)
- User authentication

## Learning Outcomes
This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Embeddings
- Semantic Search
- LangChain Integration
- LLM-based Question Answering
- Streamlit Deployment

## Author
Harsh Adhana