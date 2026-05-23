
The quality of retrieval is heavily dependent on the quality of the input query. Users often ask vague, short, or context-dependent questions that do not match the semantic structure of the stored documents. **Query Optimization** transforms the raw user input into a format that maximizes retrieval accuracy before the search even begins.

## Query Transformation Techniques

### 1. Handling Ambiguous/Vague Queries
Raw user queries often lack context or specificity.
*   **Query Expansion:** Adding synonyms or related terms to the original query to broaden the search net.
    *   *Example:* "AI risks" $\rightarrow$ "AI risks, artificial intelligence safety, algorithmic bias, job displacement."
*   **Step-Back Prompting:** Asking the LLM to generate a higher-level, abstract question from the specific user query. The abstract question retrieves broader context, which helps answer the specific question.
    *   *Example:* "Why did the stock drop 5% today?" $\rightarrow$ Step-back: "What are the general factors influencing stock prices?" + Original Query.
*   **Sub-Query Decomposition:** Breaking complex, multi-part questions into simpler, independent sub-queries. Each sub-query is retrieved separately, and results are synthesized.
    *   *Example:* "Compare the battery life of iPhone 15 and Samsung S24." $\rightarrow$ Q1: "iPhone 15 battery life specs"; Q2: "Samsung S24 battery life specs."

### 2. HyDE (Hypothetical Document Embeddings)
*   **Concept:** Instead of embedding the user’s short query, use an LLM to generate a *hypothetical answer* (a fake document) to that query. Then, embed this hypothetical document and search for similar real documents.
*   **Logic:** "Answer-like" documents are semantically closer to other "answer-like" documents in the vector space than a short question is to a long paragraph.
*   **Workflow:**
    1.  User Query: "How does photosynthesis work?"
    2.  LLM generates fake answer: "Photosynthesis is the process by which plants convert sunlight into energy..."
    3.  Embed the fake answer.
    4.  Retrieve real documents similar to the fake answer.
*   **Pros:**
    *   Bridges the "query-document mismatch" gap.
    *   Excellent for complex, explanatory queries.
    *   Improves recall significantly in many benchmarks.
*   **Cons:**
    *   **Latency:** Requires an LLM generation step before retrieval.
    *   **Hallucination Risk:** If the hypothetical answer is factually wrong but semantically plausible, it might retrieve irrelevant but similarly wrong documents.
    *   **Cost:** Extra LLM API call per query.

### 3. HyPE (Hypothetical Pointers/Entities)
*   **Concept:** A variation where the LLM extracts or generates key entities, pointers, or structured metadata from the query rather than a full narrative answer. It focuses on *identifying* the crucial elements needed for retrieval.
*   **Logic:** Some queries are best answered by finding documents containing specific entities or codes rather than semantic prose.
*   **Workflow:**
    1.  User Query: "Show me the Q3 revenue for Tesla."
    2.  HyPE Extraction: `{"entity": "Tesla", "metric": "revenue", "period": "Q3 2023"}`.
    3.  Use these extracted terms to filter or boost retrieval (often combined with keyword search).
*   **Pros:**
    *   More precise for factual/structured queries.
    *   Less prone to narrative hallucination than HyDE.
    *   Can leverage metadata filtering effectively.
*   **Cons:**
    *   Requires robust entity extraction prompts.
    *   Less effective for open-ended, conceptual questions.

### Comparison: HyDE vs. HyPE

| Feature | **HyDE** | **HyPE** |
| :--- | :--- | :--- |
| **Output** | Full hypothetical paragraph/answer. | Key entities, keywords, or structured pointers. |
| **Best For** | Conceptual, explanatory, "how-to" queries. | Factual, specific, entity-driven queries. |
| **Mechanism** | Semantic similarity to a fake answer. | Keyword/Entity matching or filtering. |
| **Latency** | High (generates text). | Medium (extracts entities). |
| **Risk** | Semantic drift if hallucinated. | Missing context if entity extraction fails. |

### Pros and Cons of Query Transformation Techniques

| Technique           | Pros                                                                | Cons                                                        |
| :------------------ | :------------------------------------------------------------------ | :---------------------------------------------------------- |
| **Query Expansion** | Simple, improves recall for synonym-heavy domains.                  | Can introduce noise; dilutes query focus.                   |
| **Step-Back**       | Provides broader context; good for reasoning.                       | Two-step process increases latency.                         |
| **HyDE**            | Significant recall boost for complex queries; bridges semantic gap. | High latency; cost; potential for hallucinated guidance.    |
| **HyPE**            | Precise for factual data; leverages metadata.                       | Complex to implement; requires good NER/extraction.         |
| **Decomposition**   | Handles complex, multi-faceted questions well.                      | Very high latency (multiple retrievals); complex synthesis. |

---

## Latency Reduction Strategies

Pre-retrieval enhancements often add latency (e.g., LLM calls for HyDE). To maintain a responsive user experience, optimization is critical.

### 1. Caching Strategies
*   **Semantic Cache:** Store embeddings of previous queries and their results. If a new query is semantically similar (cosine similarity > threshold) to a cached query, return the cached result immediately.
    *   *Impact:* Near-zero latency for repeated or similar questions.
*   **Exact Match Cache:** Standard key-value cache for identical queries.

### 2. Lightweight Transformation Models
*   **Small Language Models (SLMs):** Use small, fast models (e.g., Llama-3-8B, Mistral-7B, or even smaller T5 models) for query transformation tasks like HyDE or expansion instead of large, slow models (GPT-4).
*   **Distilled Models:** Use specialized, distilled models trained specifically for query rewriting (e.g., BGE-Reranker or specific query-rewrite models) which are faster than general-purpose LLMs.

### 3. Parallelization
*   **Async Execution:** If using decomposition, run sub-query retrievals in parallel rather than sequentially.
*   **Pipeline Overlap:** Start embedding the user query while the LLM is still generating the HyDE document (if using both dense and hybrid search).

### 4. Early Exit / Conditional Transformation
*   **Query Classification:** Use a fast classifier to determine if a query needs transformation.
    *   *Simple/Factual:* Skip HyDE/Expansion; go straight to retrieval.
    *   *Complex/Ambiguous:* Apply HyDE/Step-Back.
*   **Benefit:** Avoids unnecessary LLM calls for simple queries like "What is the capital of France?"

### 5. Optimized Embedding Inference
*   **Batching:** If multiple sub-queries are generated, batch their embedding requests.
*   **Local Embedding:** Use local, optimized embedding servers (e.g., ONNX runtime, TensorRT) to reduce network latency compared to external APIs.

### Summary: Choosing Pre-Retrieval Enhancements for Latency
*   **Low Latency Requirement (<500ms):** Use **Semantic Caching** + **Exact Match**. Skip LLM-based transformations. Use Hybrid Search directly.
*   **Medium Latency (500ms–2s):** Use **Lightweight Query Expansion** or **HyPE** with small local models.
*   **High Accuracy/Low Latency Sensitivity (>2s acceptable):** Use **HyDE** or **Step-Back Prompting** with powerful LLMs for complex analytical tasks.

**Recommendation:** Implement a **Tiered Strategy**:
1.  Check Cache.
2.  Classify Query Complexity.
3.  If Simple: Direct Retrieval.
4.  If Complex: Apply HyDE/Transformation via a fast SLM.
5.  Retrieve & Rank.