## 🎯 Intelligent Retrieval Strategy & Agentic Workflow

### 🧠 How the System Decides What to Retrieve

The **RetrievalEngine** doesn't just blindly search for information—it intelligently analyzes your query to determine the most effective retrieval strategy. This agentic approach ensures you get precisely the right information, whether you need high-level insights or specific details.

#### 🔍 The Decision Framework

The system uses a sophisticated **two-phase decision process**:

```
📝 User Query → 🤖 Query Analysis → 🎯 Strategy Selection → 🔍 Targeted Retrieval → 📊 Ranked Results
```

##### Phase 1: Query Intent Analysis

The system first analyzes your query to understand what you're really asking for:

```python
def analyze_query(self, query):
    """
    Determine the most efficient retrieval type:
    - 'summary': Document-level understanding 
    - 'chunks': Specific content extraction
    """
    # AI-powered analysis determines strategy...
```

##### Phase 2: Strategy Selection

Based on the analysis, the system chooses between two retrieval strategies:

| 🎯 **Strategy** | 📋 **Best For** | 🔍 **Trigger Phrases** |
|----------------|------------------|-------------------------|
| **`summary`** | High-level understanding, comparisons, trends | "Compare", "Summarize", "Overview", "What are the key..." |
| **`chunks`** | Specific facts, exact quotes, detailed extraction | "Extract", "Show specific", "Exact value", "In [file] on page..." |

---

### 🚀 Agentic Workflow Deep Dive

#### 🎯 Summary Strategy Workflow

When your query requires **high-level understanding**:

```python
# Example: "Compare the risk factors across all documents"
query_analysis = {
    "type": "summary",
    "files": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    "reasoning": "Requires synthesis across multiple documents"
}

# Returns document summaries with key insights
context = self.get_document_outlines(files=analysis["files"])
```

**Perfect for:**
- 📊 **Cross-document analysis**: "Compare strategies across all files"
- 🔍 **Trend identification**: "What are the emerging themes?"
- 📈 **Performance overviews**: "Summarize quarterly results"
- 🎯 **Strategic insights**: "What are the key recommendations?"

#### 🔎 Chunks Strategy Workflow

When your query needs **specific, detailed information**:

```python
# Example: "What was the exact Q3 revenue figure?"
query_analysis = {
    "type": "chunks",
    "files": ["financial_report.pdf"],
    "augmented_query": "Q3 revenue figures financial results quarterly earnings"
}

# Multi-layered retrieval process
context = self._query_context(
    query_embeddings=original_query_embedding,
    a_query_embeddings=augmented_query_embedding,
    files=relevant_files
)
```

**Perfect for:**
- 💰 **Specific numbers**: "What was the Q3 sales figure?"
- 📜 **Exact quotes**: "Copy the paragraph about compliance"
- 📍 **Location-specific data**: "From section 3.2 on page 14"
- ⚖️ **Precise terms**: "Show the exact warranty conditions"

---

### 🎭 The Agentic Advantage

#### 🤖 Smart Query Augmentation

The system doesn't just use your exact query—it **intelligently expands** it for better results:

```python
# Your query: "Sales data"
# System thinking: "This is too vague, let me enhance it..."

augmented_query = "Q4 2023 sales figures revenue performance quarterly earnings EMEA region"
# Now retrieves much more relevant content!
```

#### 🧩 Multi-Strategy Hybrid Search

For chunk-based queries, the system uses **4 different search methods** simultaneously:

```python
def _query_context(self, query_embeddings, a_query_embeddings, top_k=5, files=None):
    return sorted(
        self.get_similar_chunks(query_embeddings, files=files) +           # Direct semantic match
        self.get_similar_chunk_by_summary(query_embeddings, files=files) + # Summary-based match  
        self.get_similar_chunks(a_query_embeddings, files=files) +         # Augmented query match
        self.get_similar_chunk_by_summary(a_query_embeddings, files=files), # Augmented summary match
        key=lambda x: x["similarity"], reverse=True
    )[:top_k]
```

This **hybrid approach** ensures you never miss relevant information, whether it's:
- 🎯 **Directly mentioned** in your query
- 🔍 **Semantically related** to your intent  
- 📝 **Summarized** in document overviews
- 🚀 **Enhanced** through query augmentation

#### 📊 File-Specific Intelligence

The system analyzes document outlines to **automatically identify** which files are relevant:

```python
# Document Analysis Pipeline
outlines = self.get_document_outlines()
# "filename: contract.pdf | Summary: Service agreement with payment terms | 
#  Entities: Company A, Company B, payment, obligations | Title: Service Contract"

relevant_files = ai_analysis(query, outlines)
# Automatically selects only the files that contain relevant information
```

---

### 🛠️ Configuration & Customization

#### 🎛️ Fine-Tune Retrieval Behavior

```python
# Customize chunk retrieval
retriever = RetrievalEngine()

# Adjust similarity thresholds
chunks = retriever.get_similar_chunks(
    query_embeddings, 
    files=["specific_doc.pdf"],
    top_k=10  # Get more results
)

# Control summary vs chunks behavior
detailed_context = retriever.get_context(
    query="Your query here",
    detailed=True  # Returns full chunk objects with metadata
)
```

---

### 🎯 Best Practices

#### ✅ **For Summary Queries:**
- Use comparison keywords: "compare", "analyze", "overview"
- Ask about trends, patterns, or high-level insights  
- Don't specify exact page numbers or specific details

#### ✅ **For Chunk Queries:**
- Be specific about what you want: "exact figure", "specific clause"
- Use location indicators: "in section X", "on page Y"
- Ask for verbatim content or precise data points
