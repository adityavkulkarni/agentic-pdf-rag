# RAG Pipeline Library

A comprehensive Retrieval-Augmented Generation (RAG) pipeline for intelligent document processing, featuring advanced PDF parsing, dual chunking strategies, vector storage, and AI-powered question answering.

## 🚀 Features

### 📄 Advanced PDF Processing
- **AI-Powered Parsing**: Intelligent text extraction using Azure OpenAI with OCR fallback
- **Layout Preservation**: Maintains document structure including headers, tables, and formatting
- **Visual Element Detection**: Identifies and extracts signatures, diagrams, forms, and visual components
- **Multi-Page Support**: Processes complex documents with multiple pages and sections
- **Image Analysis**: Automatic grouping and analysis of text regions using computer vision

### 🧩 Dual Chunking Strategies
- **Semantic Chunking**: Uses embedding similarity to create contextually coherent chunks
- **Agentic Chunking**: AI-driven intelligent grouping based on content themes and relationships
- **Configurable Parameters**: Customizable chunk sizes, overlap, and splitting strategies
- **Context-Aware Optimization**: Adapts chunking based on document type and content

### 🗄️ Vector Database Integration
- **PostgreSQL + pgvector**: High-performance vector similarity search
- **Multi-Level Embeddings**: Stores both document-level and chunk-level embeddings
- **Metadata Storage**: Rich metadata using JSONB for flexible querying
- **Batch Operations**: Efficient bulk insertion and retrieval operations
- **Similarity Search**: Cosine similarity-based document and chunk retrieval

### 🤖 AI Integration
- **Azure OpenAI Support**: Native integration with Azure OpenAI services
- **Multiple Models**: Support for different models for parsing, chunking, and generation
- **Structured Outputs**: Pydantic-based structured data extraction
- **Custom Extraction**: Configurable field extraction for specific document types
- **Embedding Generation**: Automatic embedding creation for all text content

### 🔄 Complete RAG Pipeline
- **End-to-End Processing**: From PDF input to AI-generated responses
- **Context Retrieval**: Intelligent context selection based on query similarity
- **Response Generation**: AI-powered answer generation with source attribution
- **Multi-Document Support**: Query across multiple documents simultaneously
- **Extensible Architecture**: Modular design for easy customization and extension

## 📦 Installation

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Tesseract OCR engine
- Azure OpenAI API access

### Install Dependencies

```bash
pip install openai pydantic langchain-experimental langchain-core
pip install psycopg2-binary python-dotenv configparser
pip install pdf2image pytesseract Pillow opencv-python numpy
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler from: https://blog.alivate.com.au/poppler-windows/

### PostgreSQL Setup

```sql
-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- The library will automatically create required tables
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in your project root:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```

### 2. Configuration File

Create `config/config.ini`:

```ini
[models]
agentic_pdf_parser_model = gpt-4o-2024-08-06
agentic_chunker_model = gpt-4o
openai_embedding_model = text-embedding-3-large

[azure_openai]
openai_endpoint = https://your-resource-name.openai.azure.com/
openai_embeddings_endpoint = https://your-resource-name.openai.azure.com/
openai_api_version = 2024-08-01-preview
openai_embeddings_api_version = 2024-08-01-preview

[directories]
output_directory = pdf_images

[database]
dbname = rag_database
user = your_db_user
password = your_db_password
host = localhost
port = 5432
```

## 🔧 Quick Start

### Basic Usage

```python
from your_library import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    config_file="config/config.ini",
    openai_api_key="your-api-key",  # Optional: overrides config
    openai_embeddings_api_key="your-embeddings-key"  # Optional
)

# Parse a PDF document
pdf_parser = pipeline.get_agentic_pdf_parser()
parsed_result = pdf_parser.run_pipline(
    pdf_file="path/to/your/document.pdf",
    file_name="document.pdf"
)

# Create chunks
chunker = pipeline.get_pdf_chunker(
    agentic_pdf_parser=pdf_parser,
    agentic_chunker_context="Legal contract analysis"
)
chunks = chunker.run_pipline()

# Store in database
db_handler = pipeline.get_db_handler()
db_handler.insert_document(parsed_result)
db_handler.batch_insert_embeddings(
    agentic_chunks=chunks["agentic_chunks"],
    semantic_chunks=chunks["semantic_chunks"]
)

# Query the system
retrieval_engine = pipeline.get_retrieval_engine()
generation_engine = pipeline.get_generation_engine()

query = "What are the key terms of this contract?"
context = retrieval_engine.get_context(query)
response = generation_engine.generate_response(query, context)

print(response)
```

### Advanced Usage

```python
# Custom chunking parameters
chunker = pipeline.get_pdf_chunker(
    agentic_chunker_context="Financial document analysis",
    buffer_size=2,
    breakpoint_threshold_type="percentile",
    min_chunk_size=200,
    number_of_chunks=15
)

# Custom extraction with structured output
from pydantic import BaseModel, Field

class ContractTerms(BaseModel):
    parties: list[str] = Field(description="Contract parties")
    effective_date: str = Field(description="Contract effective date")
    termination_date: str = Field(description="Contract termination date")
    key_obligations: list[str] = Field(description="Key obligations")

custom_prompt = "Extract contract terms from: <>"  # <> will be replaced with document text
extracted_data = pdf_parser.get_custom_extraction(
    custom_extraction_prompt=custom_prompt,
    custom_feature_model=ContractTerms
)

# Advanced querying
similar_docs = retrieval_engine.get_similar_document(
    query_embeddings=db_handler.embedding_client.create_embedding_dict([query])[query],
    top_k=10
)

# Context-aware generation
response = generation_engine.generate_response(
    query=query,
    context=context,
    additional_instructions="Focus on legal implications and provide specific citations."
)
```

## 📚 API Reference

### RAGPipeline

Main orchestrator class that provides factory methods for all components.

#### Methods:
- `get_agentic_pdf_parser()` → `AgenticPDFParser`
- `get_pdf_chunker(**kwargs)` → `PDFChunker`  
- `get_db_handler()` → `DBHandler`
- `get_retrieval_engine(db_handler=None)` → `RetrievalEngine`
- `get_generation_engine(llm_client=None)` → `GenerationEngine`
- `get_openai_client()` → `AzureOpenAIChatClient`

### AgenticPDFParser

Intelligent PDF parsing with AI-powered text extraction and analysis.

#### Key Methods:
- `parse_file(pdf_file, file_name=None)` - Convert PDF to processable format
- `get_image_data()` - Extract and process page images
- `process_text()` - Perform OCR and AI-based text extraction
- `get_summary_and_ner()` - Generate document summary and extract named entities
- `run_pipline(pdf_file, file_name=None, custom_extraction_prompt=None, custom_feature_model=None)` - Complete processing pipeline

### PDFChunker

Dual-strategy document chunking system.

#### Key Methods:
- `get_semantic_chunks(sentences=None, embeddings=True, metadata={})` - Create semantic chunks
- `get_agentic_chunks(embeddings=True)` - Create AI-driven chunks
- `run_pipline(agentic_pdf_parser=None, agentic_chunker_context="")` - Complete chunking pipeline

#### Parameters:
- `buffer_size` - Size of context buffer for semantic chunking
- `breakpoint_threshold_type` - Method for determining chunk boundaries
- `min_chunk_size` - Minimum characters per chunk
- `number_of_chunks` - Target number of chunks

### DBHandler

PostgreSQL vector database management.

#### Key Methods:
- `insert_document(document)` - Store complete document with metadata
- `batch_insert_embeddings(agentic_chunks, semantic_chunks)` - Bulk insert chunks
- `similarity_search_chunks(table_name, query_embedding, embedding_column, top_k=5)` - Find similar chunks
- `similarity_search_document(table_name, query_embedding, top_k=5)` - Find similar documents

### RetrievalEngine

Context retrieval and query analysis.

#### Key Methods:
- `get_similar_chunks(query_embeddings)` - Retrieve similar text chunks
- `get_similar_chunk_by_summary(query_embeddings, top_k=5)` - Retrieve by document summary
- `get_similar_document(query_embeddings, top_k=5)` - Retrieve similar documents
- `get_context(query)` - Get comprehensive context for query

### GenerationEngine

AI-powered response generation.

#### Key Methods:
- `generate_response(query, context, additional_instructions=None)` - Generate contextual responses

## 🛠️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  AgenticPDFParser │───▶│   Parsed Data   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Query     │    │   PDFChunker     │◀───│   Text Content  │  
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                      
         │                        ▼                      
         │              ┌─────────────────┐              
         │              │   Dual Chunks   │              
         │              │ Semantic+Agentic│              
         │              └─────────────────┘              
         │                        │                      
         │                        ▼                      
         │              ┌─────────────────┐              
         │              │   PostgreSQL    │              
         │              │   + pgvector    │              
         │              └─────────────────┘              
         │                        │                      
         ▼                        ▼                      
┌─────────────────┐    ┌──────────────────┐              
│ RetrievalEngine │◀───│   Vector Store   │              
└─────────────────┘    └──────────────────┘              
         │                                                
         ▼                                                
┌─────────────────┐    ┌──────────────────┐              
│   Context       │───▶│ GenerationEngine │              
└─────────────────┘    └──────────────────┘              
                                │                        
                                ▼                        
                       ┌─────────────────┐              
                       │ AI Response     │              
                       └─────────────────┘              
```

## 🔍 Use Cases

- **Legal Document Analysis**: Contract review, clause extraction, compliance checking
- **Financial Document Processing**: Report analysis, data extraction, regulatory compliance
- **Research Paper Analysis**: Academic document processing, citation extraction, summarization
- **Technical Documentation**: Manual processing, FAQ generation, knowledge base creation
- **Medical Records**: Clinical document analysis, patient data extraction (with appropriate privacy controls)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the documentation for common solutions
- Review the configuration guide for setup problems

## 🏗️ Roadmap

- [ ] Multi-format document support (Word, Excel, etc.)
- [ ] Additional vector database backends
- [ ] Web interface for document management
- [ ] API server implementation
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Performance optimization tools
- [ ] Advanced query analysis features

---

**Note**: This library requires Azure OpenAI API access and a PostgreSQL database with pgvector extension. Ensure you have the necessary credentials and infrastructure before getting started.