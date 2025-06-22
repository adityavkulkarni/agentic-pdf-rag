# 🚀 Agentic PDF RAG Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078d4.svg)](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)

> **A next-generation RAG pipeline that transforms any document into intelligent, queryable knowledge using AI-powered parsing, dual chunking strategies, and vector search.**

## 🌟 Why Agentic PDF RAG?

Traditional document processing pipelines struggle with complex layouts, visual elements, and semantic understanding. The library combines **intelligent PDF parsing**, **AI-driven chunking**, and **advanced retrieval** to create a truly intelligent document processing system.

### 🎯 Key Innovations

- **🧠 Agentic Chunking**: AI agents dynamically group related content with semantic understanding
- **👁️ Visual Intelligence**: Detect signatures, diagrams, checkboxes, and other visual elements
- **🔄 Dual Strategy**: Combine semantic and agentic chunking for optimal content organization
- **⚡ Flexible Deployment**: Library, API server, or individual components
- **🎯 Context-Aware**: Retrieve the most relevant information for any query

---

## 🏗️ Architecture Overview

```
📄 Document Input → 🔍 AI Parser → 🧠 Dual Chunking → 🗄️ Vector DB → 🔍 Retrieval → 💬 Generation
    (PDF)               (OCR+LLM)     (Semantic+AI)      (PostgreSQL)    (Hybrid)      (Azure OpenAI)
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **AgenticPDFParser** | Intelligent document parsing | OCR, visual element detection, LLM understanding |
| **AgenticChunker** | AI-powered content grouping | Semantic analysis, dynamic titles, context-aware |
| **PDFChunker** | Dual chunking orchestration | Semantic + agentic strategies, embedding generation |
| **DBHandler** | Vector database operations | PostgreSQL + pgvector, similarity search |
| **RetrievalEngine** | Context retrieval | Multi-strategy search, ranking, aggregation |
| **GenerationEngine** | Response synthesis | Context-aware generation, custom instructions |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Azure OpenAI API keys
- Tesseract OCR

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/adityavkulkarni/agentic-pdf-rag
cd agentic-pdf-rag

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Or macOS
brew install tesseract
```

### 2. Configuration

Create `config/config.ini`:

```ini
[models]
agentic_pdf_parser_model = gpt-4o
agentic_chunker_model = gpt-4o
openai_embedding_model = text-embedding-3-large

[azure_openai]
openai_endpoint = https://your-resource.openai.azure.com/
openai_embeddings_endpoint = https://your-embeddings-resource.openai.azure.com/
openai_api_version = 2024-08-01
openai_embeddings_api_version = 2024-08-01

[directories]
output_directory = pdf_images

[database]
dbname = rag_database
user = postgres
password = your_password
host = localhost
port = 5432
```

### 3. Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
```

---

## 💡 Usage Modes

### Mode 1: 📚 Library Integration (Recommended)

Perfect for programmatic control and custom workflows.

```python
import os
from agentic_pdf_rag import RAGPipeline

# Initialize pipeline
rag_pipeline = RAGPipeline(
    config_file="config/config.ini",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_embeddings_api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Option A: Step-by-step processing
parsed_pdf = rag_pipeline.parse_pdf(pdf_path="document.pdf")
chunks = rag_pipeline.create_chunks(parsed_pdf)
rag_pipeline.add_document_to_db(parsed_pdf, chunks)

# Option B: One-shot processing
rag_pipeline.add_document_to_knowledge(pdf_path="document.pdf")

# Query your documents
context = rag_pipeline.retrieve_context(query="What are the key terms?")
response = rag_pipeline.generate_response(
    query="What are the key terms?", 
    context=context,
    additional_instructions="Focus on legal terminology."
)

# Or get direct response
response = rag_pipeline.get_final_response(
    query="Summarize the main points",
    additional_instructions="Provide a bullet-point summary."
)
```

### Mode 2: 🌐 Client-Server API

Ideal for distributed systems and web applications.

**Start the server:**
```bash
python server.py
```

**Use the client:**
```python
from client import RAGClient

client = RAGClient("http://localhost:5000")

# Add document to knowledge base
client.add_document_to_context(
    pdf_path="contract.pdf", 
    filename="contract.pdf",
    agentic_chunker_context="Legal document processing"
)

# Query the system
context = client.get_context(query="What are the payment terms?")
response = client.get_final_response(
    query="What are the payment terms?",
    context=context,
    additional_instructions="Focus on dates and amounts."
)

print(response)
```

**Available API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/get_config` | Retrieve current configuration |
| POST | `/set_config` | Update configuration |
| POST | `/parse_pdf` | Process PDF file |
| POST | `/add_document_to_context` | Add document to knowledge base |
| GET | `/get_documents` | Get all stored documents |
| POST | `/get_context` | Retrieve context for query |
| POST | `/get_final_response` | Generate response with context |

### Mode 3: 🔧 Component-Level Control

Maximum flexibility for custom implementations.

```python
from agentic_pdf_rag.agentic_pdf_parser import AgenticPDFParser
from agentic_pdf_rag.agentic_chunker import AgenticChunker
from agentic_pdf_rag.db_handler import DBHandler

# Custom PDF parser
parser = AgenticPDFParser(
    model="gpt-4o-2024-08-06",
    openai_endpoint="your-endpoint",
    openai_api_key="your-key"
)

# Custom chunker with specific context
chunker = AgenticChunker(
    context="Scientific research papers",
    generate_new_metadata_ind=True
)

# Custom database handler
db = DBHandler(
    dbname="custom_db",
    user="user",
    password="pass",
    host="localhost",
    port=5432
)
```

---

## 🎯 Advanced Features

### 📄 Intelligent PDF Processing

This AI-powered parser goes beyond simple OCR:

```python
# Extract with custom prompts
parser = AgenticPDFParser()
result = parser.run_pipeline(
    pdf_file="complex_document.pdf",
    custom_extraction_prompt="Extract all financial figures and dates",
    custom_feature_model=CustomFinancialModel
)
```

**Capabilities:**
- **OCR Excellence**: Confidence-based text filtering (>30% confidence)
- **Visual Intelligence**: Signatures, diagrams, checkboxes, stamps
- **Structured Extraction**: Tables, forms, key-value pairs
- **Multi-language Support**: Tesseract-powered OCR
- **Metadata Generation**: Automated summaries and entity recognition

### 🧠 Dual Chunking Strategy

**Semantic Chunking:**
```python
chunker = PDFChunker(
    semantic_chunker_buffer_size=2,
    semantic_chunker_breakpoint_threshold_type="percentile",
    semantic_chunker_min_chunk_size=256
)
```

**Agentic Chunking:**
```python
chunker = AgenticChunker(
    context="Legal contracts focus on clauses and obligations",
    generate_new_metadata_ind=True
)
```

### 🔍 Advanced Retrieval

```python
# Multi-strategy retrieval
retriever = RetrievalEngine(db_handler=db)

# Content-based search
chunks = retriever.get_similar_chunks(query_embeddings)

# Summary-based search  
chunks = retriever.get_similar_chunk_by_summary(query_embeddings, top_k=10)

# Document-level search
docs = retriever.get_similar_document(query_embeddings)
```

---

## 🛠️ Configuration Options

### Chunking Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `buffer_size` | Semantic chunker buffer | 1 | 1-5 |
| `breakpoint_threshold_type` | Threshold calculation method | "percentile" | "percentile", "standard_deviation" |
| `min_chunk_size` | Minimum chunk size | 128 | 64-512 |
| `sentence_split_regex` | Sentence splitting pattern | `r"(?<=[.?!])\s+"` | Custom regex |

### Model Configuration

```python
# Custom model settings
pipeline = RAGPipeline(
    agentic_pdf_parser_model="gpt-4o-2024-08-06",
    agentic_chunker_model="gpt-4o-mini",  # Use smaller model for chunking
    openai_embedding_model="text-embedding-3-large"
)
```

---

## 📊 Performance & Monitoring

### Workflow Stages

1. **📥 Document Ingestion** - PDF processing and conversion
2. **🔍 Content Extraction** - OCR + LLM-based analysis  
3. **🧠 Intelligent Chunking** - Dual strategy implementation
4. **🔢 Embedding Generation** - Vector representations
5. **💾 Database Storage** - Structured storage with indexing
6. **❓ Query Processing** - User query analysis
7. **🔍 Context Retrieval** - Multi-strategy similarity search
8. **💬 Response Generation** - LLM-based synthesis

### Monitoring Tips

Coming soon

---

## 🔧 Component Installation

**1. PostgreSQL Connection Issues**
```bash
# Ensure pgvector is installed
sudo apt-get install postgresql-14-pgvector

# Enable extension in PostgreSQL
CREATE EXTENSION vector;
```

**2. OCR Not Working**
```bash
# Install Tesseract language packs
sudo apt-get install tesseract-ocr-eng tesseract-ocr-fra
```

### Debug Mode

The library has logging integrated, set the logging level to debug.

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Verbose chunker output
chunker = AgenticChunker(print_logging=True)
```

---

## 📚 Examples

### Legal Document Processing

```python
# Specialized legal document processing
rag_pipeline = RAGPipeline()

# Add context for legal documents
rag_pipeline.add_document_to_knowledge(
    pdf_path="contract.pdf",
    agentic_chunker_context="Legal contract with focus on terms, obligations, and clauses"
)

# Query with legal focus
response = rag_pipeline.get_final_response(
    query="What are the termination conditions?",
    additional_instructions="You are a legal AI assistant. Provide precise legal analysis."
)
```

### Research Paper Analysis

```python
# Scientific paper processing
rag_pipeline.add_document_to_knowledge(
    pdf_path="research_paper.pdf",
    agentic_chunker_context="Academic research paper with methodology, results, and conclusions"
)

response = rag_pipeline.get_final_response(
    query="What methodology was used?",
    additional_instructions="Focus on experimental design and statistical methods."
)
```

### Multi-Document Knowledge Base

```python
# Build comprehensive knowledge base
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for doc in documents:
    rag_pipeline.add_document_to_knowledge(
        pdf_path=doc,
        agentic_chunker_context=f"Technical documentation for {doc}"
    )

# Cross-document queries
response = rag_pipeline.get_final_response(
    query="Compare the approaches across all documents",
    additional_instructions="Synthesize information from multiple sources."
)
```

---

## 📋 Dependencies

### Core Dependencies

```txt
flask>=2.0.0                    # Web framework for API server
psycopg2-binary>=2.9.0         # PostgreSQL adapter
openai>=1.0.0                  # Azure OpenAI client
pdf2image>=1.16.0              # PDF to image conversion
pytesseract>=0.3.10            # OCR functionality
opencv-python>=4.5.0           # Image processing
Pillow>=8.0.0                  # Image handling
pydantic>=2.0.0                # Data validation
langchain-experimental>=0.0.40  # Semantic chunking
numpy>=1.21.0                  # Numerical operations
python-dotenv>=0.19.0          # Environment variables
requests>=2.28.0               # HTTP client
```

### System Dependencies

- **PostgreSQL 12+** with pgvector extension
- **Tesseract OCR** for text extraction
- **Python 3.8+** for compatibility

### Acknowledgements
- **Agentic Chunking:**  
  Uses agentic chunker inspired by [Agentic-Chunker](https://github.com/Ranjith-JS2803/Agentic-Chunker).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for intelligent document processing**

</div>
