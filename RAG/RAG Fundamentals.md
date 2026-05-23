
## Core Concepts


Retrieval-Augmented Generation (RAG) is a hybrid architecture that combines the parametric knowledge of Large Language Models (LLMs) with non-parametric knowledge from external data sources.

*   **R (Retrieval):**
    *   The process of searching an external knowledge base (e.g., vector database, knowledge graph, or document store) to find information relevant to the user’s query.
    *   Converts the user query into a representation (usually embeddings) to fetch top-$k$ relevant documents or chunks.
    *   *Goal:* Provide ground-truth context that the LLM may not have been trained on or has forgotten.
*   **A (Augmentation):**
    *   The process of synthesizing the retrieved context with the original user query to construct an enhanced prompt.
    *   Involves formatting the retrieved chunks (e.g., adding headers, citations, or delimiters) and instructing the LLM on how to use this information.
    *   *Goal:* Bridge the gap between raw data and the LLM’s input format, ensuring the model knows which information is authoritative.
*   **G (Generation):**
    *   The LLM processes the augmented prompt and generates a natural language response.
    *   The model conditions its output on both its internal weights (pre-trained knowledge) and the provided external context.
    *   *Goal:* Produce a coherent, accurate, and context-aware answer.

### Relevance in the Era of Long-Context LLMs
Despite LLMs supporting context windows of 100k–1M+ tokens, RAG remains critical for several reasons:

1.  **"Lost in the Middle" Phenomenon:** LLMs often struggle to attend to information buried in the middle of massive contexts. RAG retrieves only the most relevant snippets, keeping the context concise and focused.
2.  **Cost and Latency:** Processing hundreds of thousands of tokens per query is computationally expensive and slow. RAG reduces the input size to only what is necessary, lowering inference costs and latency.
3.  **Dynamic Knowledge:** Long-context LLMs still have static training data. RAG allows access to real-time, proprietary, or frequently updated data without retraining or fine-tuning the model.
4.  **Noise Reduction:** Dumping entire documents into a long context introduces noise. RAG acts as a filter, improving the signal-to-noise ratio for the generator.

### Reducing Hallucinations
Hallucinations occur when an LLM generates plausible but factually incorrect information. RAG mitigates this by:

*   **Grounding:** Forcing the LLM to base its answer on provided evidence rather than relying solely on parametric memory.
*   **Citation/Attribution:** Encouraging the model to cite specific retrieved chunks, making it easier to verify facts.
*   **Constraint Prompting:** Instructions such as *"Answer only using the provided context. If the answer is not present, state that you do not know"* reduce the likelihood of the model inventing facts.
*   **Separation of Concerns:** The retriever handles factual lookup, while the generator handles language formulation, reducing the cognitive load on the LLM to "remember" facts.

---

## Pipeline Steps

### 1. Indexing Process
The offline phase where data is prepared for retrieval.

1.  **Ingestion:** Loading raw data (PDFs, HTML, DB records) into the pipeline.
2.  **Cleaning:** Removing noise (headers, footers, ads), normalizing text, and handling encoding issues.
3.  **Chunking:** Splitting documents into smaller, manageable segments.
    *   *Strategies:* Fixed-size, semantic, recursive, or structure-aware chunking.
    *   *Overlap:* Adding character overlap between chunks to preserve context continuity.
4.  **Embedding:** Passing each chunk through an embedding model to generate high-dimensional vector representations.
5.  **Storage:** Storing vectors and metadata (source, page number, etc.) in a Vector Database (VectorDB) or index.

### 2. Retrieval Process
The online phase triggered by a user query.

1.  **Query Transformation:** Enhancing the raw user query (e.g., HyDE, step-back prompting, or query expansion) to improve retrieval accuracy.
2.  **Embedding Query:** Converting the (transformed) query into a vector using the same embedding model used during indexing.
3.  **Similarity Search:** Performing Approximate Nearest Neighbor (ANN) search in the VectorDB to find top-$k$ chunks with highest similarity (e.g., cosine similarity).
4.  **Re-Ranking (Optional but Recommended):** Using a cross-encoder re-ranker to re-evaluate the relevance of the top-$k$ chunks against the query, providing a more precise ranking.
5.  **Context Selection:** Selecting the top-$n$ chunks (where $n \le k$) to fit within the LLM’s context window.

### 3. Prompt Engineering Differences: RAG vs. Standard

| Feature | Standard LLM Prompt | RAG Augmented Prompt |
| :--- | :--- | :--- |
| **Input Content** | User Query + System Instruction | User Query + System Instruction + **Retrieved Context** |
| **Knowledge Source** | Parametric (Model Weights) | Non-Parametric (Retrieved Documents) |
| **Instruction Style** | Open-ended generation | **Constrained generation** (e.g., "Use only provided context") |
| **Structure** | Simple Q&A or Chat | Structured template with delimiters (e.g., `<context>...</context>`) |
| **Handling Unknowns** | Model may guess/hallucinate | Explicit instruction to say "I don't know" if context is missing info |

**Example RAG Prompt Structure:**
```text
System: You are a helpful assistant. Answer the question based ONLY on the following context. If the answer is not in the context, say "I cannot answer this based on the provided information."

Context:
[Chunk 1]: ...
[Chunk 2]: ...

Question: {user_query}
Answer:
```

---

## Components

### Choice of LLM

#### Reasoning vs. Non-Reasoning LLMs
*   **Non-Reasoning (Standard) LLMs:**
    *   *Pros:* Faster, cheaper, sufficient for straightforward extraction and summarization tasks.
    *   *Cons:* May struggle with complex logical deduction across multiple retrieved chunks.
*   **Reasoning LLMs (e.g., o1, R1):**
    *   *Pros:* Better at synthesizing conflicting information, performing multi-step logic, and handling complex analytical queries.
    *   *Cons:* Higher latency and cost; may over-think simple factual queries.
    *   *Usage:* Use reasoning models for complex RAG tasks (e.g., legal analysis, medical diagnosis) and standard models for simple QA.

#### Consequences of a Weak Generator LLM
*   **Poor Synthesis:** Fails to combine information from multiple chunks coherently.
*   **Ignored Context:** May ignore retrieved evidence and rely on pre-trained biases (leading to hallucinations).
*   **Verbose/Irrelevant Output:** Generates fluff instead of direct answers.
*   **Format Errors:** Fails to follow strict output formats (e.g., JSON, citations).

### Frameworks
Popular frameworks for building RAG systems:
1.  **LangChain:** Highly modular, extensive ecosystem, good for prototyping. *Justification:* Best for complex chains and agent-based RAG.
2.  **LlamaIndex:** Data-centric, optimized for indexing and retrieval strategies. *Justification:* Superior for complex data structures and advanced retrieval techniques.
3.  **Haystack:** Production-ready, strong support for industrial pipelines. *Justification:* Best for scalable, enterprise-grade deployments.
4.  **DSPy:** Programming framework for optimizing LM pipelines. *Justification:* Best for automatically tuning prompts and retrieval parameters.

### Hyperparameters in RAG
Key hyperparameters affecting performance:
*   **Retrieval:**
    *   `top_k`: Number of chunks retrieved initially.
    *   `similarity_threshold`: Minimum similarity score to consider a chunk relevant.
    *   `rerank_top_n`: Number of chunks passed to the re-ranker.
*   **Generation:**
    *   `temperature`: Controls randomness (lower for factual accuracy).
    *   `max_tokens`: Limits response length.
    *   `presence/frequency_penalty`: Reduces repetition.
*   **Chunking:**
    *   `chunk_size`: Size of each text segment.
    *   `chunk_overlap`: Characters shared between adjacent chunks.

### Influence of LLM Context Window Size
*   **Small Context Window (<8k tokens):**
    *   Requires aggressive retrieval (low `top_k`).
    *   Necessitates heavy compression or summarization of retrieved chunks.
    *   Increases reliance on high-precision re-ranking to avoid wasting space on irrelevant docs.
*   **Large Context Window (>100k tokens):**
    *   Allows retrieving more chunks (`higher top_k`), improving recall.
    *   Enables "Retrieve-and-Read" approaches where entire documents are fed.
    *   *Risk:* Increased latency and cost; potential for "lost in the middle" degradation if not carefully managed.
    *   *Strategy:* Can afford to be less aggressive with pre-filtering, but still benefits from re-ranking to prioritize the most relevant info at the beginning/end of the context.