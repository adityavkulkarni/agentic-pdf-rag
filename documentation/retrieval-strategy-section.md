## 🎯 Intelligent Retrieval Strategy

The **RetrievalEngine** uses an advanced, agentic approach to analyze your query and retrieve the most relevant information—whether you need high-level insights or specific details. The system is built around two main retrieval branches: **page-level context** and **document-level context** (chunks or summary).

### How the System Decides What to Retrieve

The system follows a **multi-phase decision process** to ensure you get the right information, tailored to your needs.
```
📝 User Query → 🤖 Query Analysis → 🎯 Strategy Selection → 🔍 Targeted Retrieval → 📊 Ranked Results
```

Depending on your query, the system activates one or both of the following workflows:

### 1. Page-Level Context Workflow

**When you need information from specific pages across documents:**

1. **Extract Pages & Outlines:**  
   - The system processes PDF inputs and extracts pages, along with their summaries and metadata.
2. **User Query Analysis:**  
   - Your query is analyzed to understand which pages are relevant.
3. **LLM Query Optimization:**  
   - The system uses a language model to optimize your query, using document outlines for context.
4. **Filter Relevant Pages:**  
   - Only the pages that match the optimized query are selected.
5. **Return Page Context:**  
   - The system returns the text and visual elements from those pages.

**Example Use Case:**  
> "What does the contract say about termination on page 7 of contract.pdf?"

### 2. Document-Level Context Workflow

**When you need either high-level insights or detailed extraction:**

The system chooses between two strategies based on your query:

| 🎯 **Strategy** | 📋 **Best For** | 🔍 **Trigger Phrases** |
|-----------------|------------------|-------------------------|
| **`summary`**   | High-level understanding, comparisons, trends | "Compare", "Summarize", "Overview", "What are the key..." |
| **`chunks`**    | Specific facts, exact quotes, detailed extraction | "Extract", "Show specific", "Exact value", "In [file] on page..." |

#### Summary Strategy Workflow

When your query requires **high-level understanding**:

Example: "Compare the risk factors across all documents"
```
query_analysis = {
"type": "summary",
"files": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
"reasoning": "Requires synthesis across multiple documents"
}

Returns document summaries with key insights
context = self.get_document_outlines(files=analysis["files"])
```

**Perfect for:**
- **Cross-document analysis:** "Compare strategies across all files"
- **Trend identification:** "What are the emerging themes?"
- **Performance overviews:** "Summarize quarterly results"
- **Strategic insights:** "What are the key recommendations?"

#### Chunks Strategy Workflow

When your query needs **specific, detailed information**:

Example: "What was the exact Q3 revenue figure?"
```query_analysis = {
"type": "chunks",
"files": ["financial_report.pdf"],
"augmented_query": "Q3 revenue figures financial results quarterly earnings"
}

Multi-layered retrieval process
context = self._query_context(
query_embeddings=original_query_embedding,
a_query_embeddings=augmented_query_embedding,
files=relevant_files
)
```

**Perfect for:**
- **Specific numbers:** "What was the Q3 sales figure?"
- **Exact quotes:** "Copy the paragraph about compliance"
- **Location-specific data:** "From section 3.2 on page 14"
- **Precise terms:** "Show the exact warranty conditions"

### Combining Context for Comprehensive Results

The **RetrievalEngine** can combine both **page-level** and **document-level** context to provide you with the most comprehensive answer possible.  
For example, if your query requires both specific page details and a summary of trends, the system will automatically merge the relevant information.


```text
graph TD
    A[User Query] --> B[Get Relevant Files from DB]
    B --> C1[Page-Level Branch]
    B --> C2[Doc-Level Branch]
    
    subgraph Page-Level Branch
        C1 --> D1[Get Pages from DB]
        D1 --> E1[Select Relevant Pages]
        E1 --> F1[Return page_context]
    end
    
    subgraph Doc-Level Branch
        C2 --> D2{Determine Type}
        D2 -->|Chunks| E2[Get Augmented Query]
        D2 -->|Summary| F2[Get File Outlines]
        E2 --> G2[Fetch Chunks via Multi-Strategy Search]
        G2 --> H2[Generate doc_context]
        F2 --> I2[Generate doc_context]
    end
    
    F1 --> J[Compare & Evaluate]
    H2 --> J
    I2 --> J
    J --> K[Return Best Matching Context]


```