

Chunking is the process of splitting large documents into smaller, semantically coherent units (chunks) that can be effectively embedded and retrieved. It is the most critical preprocessing step in RAG, as poor chunking leads to poor retrieval, regardless of the quality of the embedding model or LLM.

## Chunking Basics

### Importance of Chunking
*   **Embedding Limits:** Embedding models have maximum token limits (e.g., 512, 8192 tokens). Large documents must be split to fit.
*   **Retrieval Precision:** Retrieving a massive document for a specific question introduces noise. Smaller, focused chunks improve the signal-to-noise ratio.
*   **Context Window Efficiency:** LLMs have limited context windows. Sending only relevant chunks saves tokens and reduces cost/latency.
*   **Semantic Integrity:** Proper chunking ensures that each unit contains a complete thought or fact, preventing fragmented meanings.

### Chunk Size Selection
There is no "one-size-fits-all" chunk size; it depends on the data type and embedding model.
*   **General Rule of Thumb:** 200–500 tokens per chunk is a common starting point.
*   **Embedding Model Constraint:** Must be within the model’s max input length.
*   **Query Complexity:**
    *   *Simple Factoid Queries:* Smaller chunks (100–200 tokens) are sufficient.
    *   *Complex/Analytical Queries:* Larger chunks (500–1000+ tokens) provide necessary context.
*   **Empirical Tuning:** Evaluate retrieval metrics (Precision@k, Recall@k) across different chunk sizes to find the optimal balance.

### Character Overlap
*   **Definition:** The number of characters or tokens shared between adjacent chunks.
*   **Purpose:** Prevents semantic fragmentation at chunk boundaries. If a sentence or concept is split exactly in the middle, neither chunk may contain enough context to be meaningful.
*   **Typical Value:** 10–20% of the chunk size (e.g., if chunk size is 500 tokens, overlap might be 50–100 tokens).
*   **Trade-off:** Too much overlap increases storage costs and redundancy; too little risks losing context.

### Consequences of Chunk Size Extremes

| Scenario | Consequences |
| :--- | :--- |
| **Chunks Too Small** | • **Loss of Context:** Individual chunks may lack sufficient information to answer queries.<br>• **Fragmented Meaning:** Sentences or concepts are cut off.<br>• **High Noise:** More chunks retrieved to cover the same info, increasing irrelevant hits.<br>• **Increased Latency:** More embeddings to compute and store. |
| **Chunks Too Large** | • **Diluted Semantics:** Embeddings become averages of multiple topics, reducing specificity.<br>• **Lower Precision:** Retrieved chunks may contain the answer but also lots of irrelevant text.<br>• **Context Window Waste:** Consumes more LLM context tokens, limiting the number of chunks you can retrieve.<br>• **"Lost in the Middle":** Key info may be buried in a large block, ignored by the LLM. |

---

## Chunking Methods

### 1. Fixed-Size (Character/Token) Chunking
*   **Method:** Split text every $N$ characters/tokens with $M$ overlap.
*   **Pros:** Simple, fast, computationally cheap.
*   **Cons:** Ignores semantic boundaries; often splits sentences or paragraphs arbitrarily.
*   **Best For:** Uniform text where semantic boundaries are less critical, or as a baseline.

### 2. Recursive Character Chunking
*   **Method:** Splits text by a hierarchy of separators (e.g., `\n\n`, `\n`, ` `, ``) until the chunk fits the size limit.
*   **Pros:** Respects natural boundaries (paragraphs, sentences) better than fixed-size.
*   **Cons:** Still may split complex structures; requires tuning separator order.
*   **Best For:** General-purpose text documents (articles, blogs).

### 3. Semantic Chunking
*   **Method:** Uses an embedding model to calculate similarity between consecutive sentences. A split occurs when the similarity drops below a threshold (indicating a topic change).
*   **Pros:** Ensures each chunk is semantically coherent; highly effective for complex topics.
*   **Cons:** Computationally expensive (requires embedding every sentence); slower indexing.
*   **Best For:** Dense technical documents, research papers, or when high retrieval precision is critical.

### 4. Structure-Aware Chunking (Structured vs. Plain Text)

#### Plain Text Documents
*   **Challenge:** No explicit markers for sections.
*   **Strategy:** Use recursive splitting based on punctuation and paragraph breaks. Semantic chunking is highly beneficial here.

#### Structured Documents (PDFs, HTML, Markdown)
*   **Challenge:** Contains tables, figures, headers, footers, and nested lists.
*   **Strategy:**
    *   **Header-Based Chunking:** Split text based on header hierarchy (H1, H2, H3). Each chunk inherits the path of headers (e.g., "Chapter 1 > Section 2 > Subsection A").
    *   **Table Handling:** Do not split tables row-by-row. Keep entire tables as single chunks or convert them to markdown/JSON format. If too large, summarize the table or split by logical groups of rows with header repetition.
    *   **Figure/Image Handling:** Extract alt-text or captions. Use multimodal embeddings if available. Otherwise, treat as separate metadata-linked chunks.
    *   **Code Blocks:** Keep code blocks intact. Split by function or class definitions rather than character count.

### Criteria for Choosing a Chunking Method
1.  **Document Structure:** Is it plain text, HTML, PDF, or code?
2.  **Query Type:** Are users asking for specific facts (smaller chunks) or summaries (larger chunks)?
3.  **Computational Budget:** Can you afford the latency of semantic chunking?
4.  **Embedding Model:** Some models perform better with sentence-level inputs; others handle paragraph-level well.
5.  **Evaluation Metrics:** Test different methods against a golden dataset using Recall@k and Precision@k.

---

## Chunk Enhancements

Enhancements aim to enrich chunks with additional context to improve retrieval accuracy, especially when chunks are small or lack standalone meaning.

### 1. Contextual Chunk Headers (CCH)
*   **Technique:** Prepend hierarchical metadata (e.g., document title, section headers) to each chunk before embedding.
*   **Example:**
    *   *Original Chunk:* "It was released in 2023."
    *   *Enhanced Chunk:* "Document: iPhone Review > Section: Release Date > Content: It was released in 2023."
*   **Pros:**
    *   Provides crucial context for ambiguous chunks.
    *   Improves semantic matching by including broader topic keywords.
    *   Helps distinguish between similar sections in different parts of a document.
*   **Cons:**
    *   Increases token usage for embeddings.
    *   Requires accurate parsing of document structure (headers).

### 2. Summary Augmentation
*   **Technique:** Generate a concise summary of the chunk (or the parent document) and prepend/append it to the chunk.
*   **Pros:**
    *   Captures high-level themes that might be missing in a small chunk.
    *   Improves retrieval for abstract or thematic queries.
*   **Cons:**
    *   High computational cost (requires LLM call for every chunk).
    *   Risk of summary hallucination or inaccuracies.

### 3. Hypothetical Questions (HyDE-style Indexing)
*   **Technique:** Use an LLM to generate hypothetical questions that the chunk could answer. Store these questions alongside the chunk.
*   **Pros:**
    *   Aligns the index with user query language (questions vs. statements).
    *   Greatly improves recall for natural language queries.
*   **Cons:**
    *   Very high indexing cost and time.
    *   Storage overhead (storing multiple questions per chunk).

### 4. Metadata Enrichment
*   **Technique:** Extract entities (dates, names, locations), topics, or sentiment from the chunk and store as filterable metadata.
*   **Pros:**
    *   Enables hybrid search (semantic + metadata filtering).
    *   Allows pre-filtering (e.g., "only show me documents from 2024").
*   **Cons:**
    *   Requires robust NER (Named Entity Recognition) or classification models.
    *   Complex query logic needed to leverage filters effectively.

### Pros and Cons of Chunk Enhancement Techniques

| Technique | Pros | Cons |
| :--- | :--- | :--- |
| **Contextual Headers** | Low cost, high impact on context, easy to implement. | Dependent on document structure quality. |
| **Summary Augmentation** | Adds high-level context, improves thematic retrieval. | Expensive (LLM calls), slow indexing, potential hallucination. |
| **Hypothetical Questions** | Best for aligning with user intent, boosts recall. | Very expensive, high storage overhead, complex to maintain. |
| **Metadata Enrichment** | Enables precise filtering, improves hybrid search. | Requires extra processing pipeline, complex query construction. |

**Recommendation:** Start with **Contextual Chunk Headers** as they offer the best balance of effort and performance. Use **Summary Augmentation** or **Hypothetical Questions** only for high-value, dense datasets where retrieval accuracy is paramount and budget allows.