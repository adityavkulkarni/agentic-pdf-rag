# 🚀 Agentic PDF Parser Suite

**Unlock the Power of AI-Driven Document Understanding!**

Welcome to the Agentic PDF Parser Suite—a cutting-edge Python package designed to revolutionize how you process, analyze, and extract insights from PDF documents. Whether you’re dealing with legal contracts, academic papers, or business reports, this suite combines OCR, AI-powered chunking, and semantic analysis for unparalleled document intelligence.

---

## 🔥 Features at a Glance
- **AI-Powered PDF Parsing**  
  - Extract text, images, and structure from any PDF using OCR and advanced vision models.
- **Smart Chunking**  
  - Group document content by semantic meaning or agent-driven logic for better analysis and retrieval.
- **Agentic Chunking**  
  - Use AI to group sentences by topic, update summaries, and maintain context-aware metadata.
- **Semantic Chunking**  
  - Leverage embedding models to split documents into meaningful sections based on content similarity.
- **Named Entity Recognition & Summarization**  
  - Automatically identify key entities and generate concise summaries for each document or section.
- **Custom Extraction**  
  - Define your own prompts and output schemas for specialized data extraction.
- **Visual Element Processing**  
  - Identify and process diagrams, signatures, checkboxes, and more.
- **Batch Processing**  
  - Handle multiple documents with ease, with support for batch embeddings and chunking.

---

## 🚀 Get Started
```shell
pip install agentic-pdf-parser # (Coming soon to PyPI!)
```

Or clone this repo and explore the code yourself!

---

## 🛠️ How It Works

### 1. **Parse Your PDF**

```python
from agentic_pdf_rag import AgenticPDFParser

# Initialize with your PDF path
parser = AgenticPDFParser("your_document.pdf")
results = parser.extract_and_retrieve_results()
```

Extracts text, images, and structure from your PDF using OCR and AI vision.

---
### 2. **Extract Insights**
```python
# Get summary and named entities
parser.get_summary_and_ner()

# Custom extraction with your own prompt
parser.get_custom_extraction("Your custom prompt", YourPydanticModel)
```

Summarizes content, identifies entities, and supports custom data extraction.

---
### 3. **Process Visual Elements**
```python
parser.results.pages_descriptions
```

Automatically detects and processes diagrams, signatures, checkboxes, and more—right out of the box!

---
### 4. **Chunk Your Document**

```python
from agentic_pdf_rag import PDFChunker

# Initialize chunker with the parser
chunker = PDFChunker(parser)
chunker.get_semantic_chunks()  # Semantic chunking
chunker.get_agentic_chunks()  # Agentic chunking
```

Groups content by meaning or agent logic for easier analysis and retrieval.

---

## 🎯 Key Use Cases
- **Legal Document Analysis:** Extract clauses, obligations, and entities from contracts.
- **Academic Research:** Summarize papers, extract key findings, and group related content.
- **Business Intelligence:** Analyze reports, extract insights, and automate data entry.
- **Document Search:** Build semantic search indices for large document collections.

---
## 🌟 Why Choose Agentic PDF Parser Suite?
- **AI-Powered:** Leverages the latest in GPT-4o and embedding models for superior understanding.
- **Extensible:** Easily customize extraction, chunking, and analysis for your needs.
- **Modern & Fun:** Designed with developers in mind—clean code, clear docs, and a touch of personality!
---

## 📦 Package Structure
The Agentic PDF Parser Suite is modular and extensible. Here’s what’s inside:
| File/Module           | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `init.py`             | Exposes core classes for import                                         |
| `agentic_pdf_parser.py` | Main PDF parser with OCR, AI vision, and extraction logic              |
| `agentic_chunker.py`  | Agent-driven chunking and metadata management                           |
| `pdf_chunker.py`      | Semantic and agentic chunking orchestration                             |
| `embeddings.py`       | Embedding model adapters for semantic analysis                          |
| `image_parser.py`     | Visual processing, text grouping, and image analysis                    |
| `openai_client.py`    | Azure OpenAI integration for chat and embeddings                        |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 💬 Feedback & Contributions

We love feedback! Open an issue or submit a PR to help make this package even better.
  
---

**Happy Document Parsing! 🚀📄✨**



